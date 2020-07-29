#!/usr/bin/env python

import numpy as np
import sys
import os
import argparse
import pdb

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.offsetbox import TextArea, DrawingArea, OffsetImage, AnnotationBbox
from collections import OrderedDict, defaultdict
from scipy.interpolate import interp1d

from common import rec_dict, get_dict
from get_database import Database, Post, input_args
from simulation_io import get_caps_db, collect_results_db_1d_3d, collect_results_db_3d_3d, \
    collect_results_db_3d_3d_spatial

sys.path.append('/home/pfaller/work/repos/SimVascular/Python/site-packages/')

import sv_1d_simulation as oned


def add_image(db, geo, fig):
    if not os.path.exists(db.get_png(geo)):
        return
    im = plt.imread(db.get_png(geo))
    newax = fig.add_axes([-0.22, 0.3, 0.3, 0.3], anchor='NE')#, zorder=-1
    newax.imshow(im)
    newax.axis('off')


def plot_1d_3d_all(db, opt, geo, res, time):
    # get post-processing constants
    post = Post()

    # get 1d/3d map
    caps = get_caps_db(db, geo)

    fig, ax = plt.subplots(len(post.fields), len(post.models), figsize=(20, 10), dpi=opt['dpi'], sharex='col', sharey='row')

    for i, f in enumerate(post.fields):
        for j, m in enumerate(post.models):
            ax[i, j].grid(True)

            if opt['legend_row'] or i == 0:
                ax[i, j].set_title(geo + ' ' + m + ' ' + f)
            if opt['legend_row'] or i == len(post.fields) - 1:
                ax[i, j].set_xlabel('Time [s]')
            if opt['legend_col'] or j == 0:
                ax[i, j].set_ylabel(f.capitalize() + ' [' + post.units[f] + ']')

            for c in caps.values():
                ax[i, j].plot(time[m + '_all'], res[c][f][m + '_all'] * post.convert[f])

    add_image(db, geo, fig)
    fig.savefig(db.get_post_path(geo, 'all'))
    plt.close(fig)


def plot_1d_3d_cyclic(db, opt, geo, res, time):
    # get post-processing constants
    post = Post()

    # get 1d/3d map
    caps = db.get_surface_names(geo, 'caps')

    fig, ax = plt.subplots(len(post.fields), len(caps), figsize=(opt['w'], opt['h']), dpi=opt['dpi'], sharex=opt['sharex'], sharey=opt['sharey'])

    for i, f in enumerate(post.fields):
        for j, c in enumerate(caps):
            ax[i, j].grid(True)

            if opt['legend_row'] or i == 0:
                ax[i, j].set_title(c)
            if opt['legend_row'] or i == len(post.fields) - 1:
                ax[i, j].set_xlabel('Time [s]')
            if opt['legend_col'] or j == 0:
                ax[i, j].set_ylabel(f.capitalize() + ' [' + post.units[f] + ']')

            res_split = np.split(res[c][f]['1d_all'][:time['step_cycle'] * time['n_cycle']], time['n_cycle'])
            for r in res_split:
                ax[i, j].plot(time['1d'], r * post.convert[f], post.styles['1d'])

            ax[i, j].plot(time['3d'], res[c][f]['3d'] * post.convert[f], post.styles['3d'])

    add_image(db, geo, fig)
    fig.savefig(db.get_post_path(geo, 'cyclic'))
    plt.close(fig)


def plot_1d_3d_caps(db, opt, geo, res, time):
    # get post-processing constants
    post = Post()

    # get 1d/3d map
    caps = get_caps_db(db, geo)
    
    # get cap names
    names = db.get_cap_names(geo)

    # get 0d bcs
    bc_0d = get_dict(db.get_bc_0D_path(geo))
    
    if len(caps) > 50:
        dpi = opt['dpi'] // 4
        sharey = False
    else:
        dpi = opt['dpi']
        sharey = opt['sharey']

    fields = post.fields
    fields.remove('area')

    fig, ax = plt.subplots(len(fields), len(caps), figsize=(opt['w'], opt['h']), dpi=dpi, sharex=opt['sharex'], sharey=sharey)

    for i, f in enumerate(fields):
        for j, (c, br) in enumerate(caps.items()):
            ax[i, j].grid(True)

            if opt['legend_row'] or i == 0:
                ax[i, j].set_title(names[c])
            if opt['legend_row'] or i == len(fields) - 1:
                ax[i, j].set_xlabel('Time [s]')
                ax[i, j].xaxis.set_tick_params(which='both', labelbottom=True)
            if opt['legend_col'] or j == 0:
                ax[i, j].set_ylabel(f.capitalize() + ' [' + post.units[f] + ']')
                ax[i, j].yaxis.set_tick_params(which='both', labelleft=True)

            lg = []
            for m in post.models:
                ax[i, j].plot(time[m], res[br][f][m + '_cap'] * post.convert[f], post.styles[m])
                lg += [m.upper()]

            if f == 'pressure' and bc_0d:
                if br in bc_0d[f]:
                    ax[i, j].plot(bc_0d['time'], bc_0d[f][br] * post.convert[f], 'k--')
                    lg += ['0D BC']

            ax[i, j].legend(lg)

    add_image(db, geo, fig)
    # fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig(db.get_post_path(geo, 'caps'), bbox_inches='tight')
    plt.close(fig)


def plot_1d_3d_interior(db, opt, geo, res, time):
    # get post-processing constants
    post = Post()

    # get 1d/3d map
    caps = get_caps_db(db, geo)
    cap_br = list(caps.values())
    cap_names = list(caps.keys())

    # get cap names
    names = db.get_cap_names(geo)

    if len(res) > 50:
        dpi = opt['dpi'] // 4
        sharey = False
    else:
        dpi = opt['dpi']
        sharey = opt['sharey']

    fig, ax = plt.subplots(len(post.fields), len(res), figsize=(opt['w'], opt['h']), dpi=dpi, sharex='col', sharey=sharey)

    # pick reference time step with highest inflow
    m_ref = '3d'
    t_max = {m_ref: np.argmax(res[0]['flow'][m_ref + '_cap'])}

    # pick closest time step for other models
    for m in post.models:
        if m != m_ref:
            t_max[m] = np.argmin(np.abs(time[m_ref][t_max[m_ref]] - time[m]))

    for i, f in enumerate(post.fields):
        for j, br in enumerate(res.keys()):
            ax[i, j].grid(True)

            if opt['legend_row'] or i == 0:
                if br in cap_br:
                    name = cap_names[cap_br.index(br)]
                else:
                    name = 'branch ' + str(br)
                ax[i, j].set_title(name)
            if opt['legend_row'] or i == len(post.fields) - 1:
                ax[i, j].set_xlabel('Vessel path [1]')
                ax[i, j].xaxis.set_tick_params(which='both', labelbottom=True)
            if opt['legend_col'] or j == 0:
                ax[i, j].set_ylabel(f.capitalize() + ' [' + post.units[f] + ']')
                ax[i, j].yaxis.set_tick_params(which='both', labelleft=True)

            lg = []
            for m in post.models:
                path = res[br][m + '_path']
                ax[i, j].plot(path / path[-1], res[br][f][m + '_int'][:, t_max[m]] * post.convert[f], post.styles[m])
                lg.append(m)

            ax[i, j].legend(lg)
            ax[i, j].set_xlim(left=0)

    add_image(db, geo, fig)
    # fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig(db.get_post_path(geo, 'interior'))
    plt.close(fig)


def plot_1d_3d_paper(db, opt, geo, res, time):
    # get post-processing constants
    post = Post()

    # get 1d/3d map
    caps = db.get_surface_names(geo, 'caps')

    # 0075_1001
    # names = {'inflow': 'Inflow', 'aorta_outflow': 'Aorta', 'Lt-carotid_segs_final': 'Left Carotid',
    #          'rt-carotid_segs_final': 'Right Carotid', 'btrunk_segs_final': 'Right Subclavian',
    #          'subclavian_segs_final': 'Left Subclavian'}
    # myorder = ['inflow', 'aorta_outflow', 'Lt-carotid_segs_final', 'rt-carotid_segs_final', 'subclavian_segs_final', 'btrunk_segs_final']
    # myorder = ['aorta_outflow', 'rt-carotid_segs_final', 'btrunk_segs_final']

    names = {'inflow': 'Aorta', 'R_int_iliac': 'Right Internal Iliac', 'R_ext_iliac': 'Right External Iliac',
             'SMA': 'Superior Mesenteric', 'IMA': 'Inferior Mesenteric'}
    myorder = ['IMA', 'R_int_iliac', 'R_ext_iliac']

    # layout
    nx = 2
    ny = 3

    # plot caps
    fig, ax = plt.subplots(ny, nx, figsize=(7, 6), dpi=opt['dpi'], sharex=True, sharey='col')
    for i, f in enumerate(post.fields):
        for j, c in enumerate(myorder):
            # pos = np.unravel_index(j, (ny, nx))
            pos = (j, i)
            ax[pos].grid(True)
            ax[pos].set_title(names[c])

            if opt['legend_row'] or pos[0] == ny - 1:
                ax[pos].set_xlabel('Time [s]')
                ax[pos].xaxis.set_tick_params(which='both', labelbottom=True)
            if opt['legend_col'] or pos[1] == 0:
                ax[pos].set_ylabel(f.capitalize() + ' [' + post.units[f] + ']')
                ax[pos].yaxis.set_tick_params(which='both', labelleft=True)

            for m in post.models:
                ax[pos].plot(time['3d'], res[c][f][m] * post.convert[f], post.styles[m], color=post.colors[m])
            plt.sca(ax[pos])
            plt.xlim([0, plt.xlim()[1]])

    fig.savefig(db.get_post_path(geo, 'paper_caps'), bbox_inches='tight', pad_inches=0.01)
    plt.close(fig)

    # pick time step
    t_max = np.argmax(res['flow']['inflow']['3d'])

    for i, f in enumerate(post.fields):
        fig, ax = plt.subplots(ny, nx, figsize=(9, 8), dpi=opt['dpi'], sharex=True, sharey=True)
        for j, c in enumerate(myorder):
            pos = np.unravel_index(j, (ny, nx))
            ax[pos].grid(True)
            ax[pos].set_title(names[c])
            ax[pos].set_ylim([0, 110])

            if opt['legend_row'] or pos[0] == ny - 1:
                ax[pos].set_xlabel('Vessel path [1]')
                ax[pos].xaxis.set_tick_params(which='both', labelbottom=True)
            if opt['legend_col'] or pos[1] == 0:
                ax[pos].set_ylabel(f.capitalize() + ' [' + post.units[f] + ']')
                ax[pos].yaxis.set_tick_params(which='both', labelleft=True)

            for m in post.models:
                ax[pos].plot(res['path'][c][m], res[c][f][m + '_int'][t_max] * post.convert[f], post.styles[m], color=post.colors[m])

        fig.savefig(db.get_post_path(geo, 'paper_interior_' + f), bbox_inches='tight', pad_inches=0)
        plt.close(fig)


def calc_error(db, geo, res, time):
    # get post-processing constants
    post = Post()

    # get 1d/3d map
    caps = get_caps_db(db, geo)

    # interpolate 1d to 3d in space and time (allow extrapolation due to round-off errors at bounds)
    interp = lambda x_1d, y_1d, x_3d: interp1d(x_1d, y_1d.T, fill_value='extrapolate')(x_3d)

    # relative difference between two arrays
    rel_diff = lambda a, b: np.abs((a - b) / b)

    # get spatial error
    err = rec_dict()
    for f in post.fields:
        for br in res.keys():
            # retrieve 3d results
            res_3d = res[br][f]['3d_int']

            # map paths to interval [0, 1]
            path_1d = res[br]['1d_path'] / res[br]['1d_path'][-1]
            path_3d = res[br]['3d_path'] / res[br]['3d_path'][-1]

            # interpolate in space and time
            res_1d = interp(path_1d, res[br][f]['1d_int'], path_3d)
            res_1d = interp(time['1d'], res_1d, time['3d'])

            # calculate spatial error (eliminate time dimension)
            if f == 'pressure' or f == 'area':
                diff = rel_diff(res_1d, res_3d)
                err[f]['spatial']['avg'][br] = np.mean(diff, axis=1)
                err[f]['spatial']['max'][br] = np.max(diff, axis=1)
            elif f == 'flow':
                diff = np.abs((res_1d - res_3d).T / np.max(res_3d, axis=1))
                err[f]['spatial']['avg'][br] = np.mean(diff, axis=0)
                err[f]['spatial']['max'][br] = np.max(diff, axis=0)

            err[f]['spatial']['sys'][br] = np.abs(rel_diff(np.max(res_1d, axis=1), np.max(res_3d, axis=1)))
            err[f]['spatial']['dia'][br] = np.abs(rel_diff(np.min(res_1d, axis=1), np.min(res_3d, axis=1)))

    for f in post.fields:
        for m in err[f]['spatial'].keys():
            # get interior error
            for br in res.keys():
                err[f]['int'][m][br] = np.mean(err[f]['spatial'][m][br])

            # get cap error
            for br in caps.values():
                if br == 0:
                    # inlet
                    i_cap = 0
                else:
                    # outlet
                    i_cap = -1
                err[f]['cap'][m][br] = err[f]['spatial'][m][br][i_cap]

            # get error over all branches
            err[f]['int'][m]['all'] = np.mean([err[f]['int'][m][br] for br in res.keys()])
            err[f]['cap'][m]['all'] = np.mean([err[f]['cap'][m][br] for br in caps.values()])

    db.add_1d_3d_comparison(geo, err)


def main(db, geometries):
    # get post-processing constants
    post = Post()

    for geo in geometries:
        print('Comparing geometry ' + geo)

        # read results
        if '3d' in post.models and '1d' in post.models:
            res, time = collect_results_db_1d_3d(db, geo)
        elif '3d' in post.models and '3d_rerun' in post.models:
            res, time = collect_results_db_3d_3d(db, geo)
        else:
            raise ValueError('Unknown combination of models')

        if res is None:
            continue

        # plot options
        opt = {'legend_col': False,
               'legend_row': False,
               'sharex': True,
               'sharey': 'row',
               'dpi': 200,
               'w': 1 * (len(db.get_surface_names(geo)) * 3 + 4),
               'h': 2 * (len(Post().fields) * 1 + 2)}

        # calculate error
        if '3d' in post.models and '1d' in post.models:
            calc_error(db, geo, res, time)

        # generate plots
        print('plotting')
        plot_1d_3d_all(db, opt, geo, res, time)
        plot_1d_3d_caps(db, opt, geo, res, time)
        plot_1d_3d_interior(db, opt, geo, res, time)

        # plot_1d_3d_paper(db, opt, geo, res, time)

        # plot_1d_3d_cyclic(db, opt, geo, res, time)


if __name__ == '__main__':
    descr = 'Plot comparison of xd-results'
    d, g, _ = input_args(descr)
    main(d, g)
