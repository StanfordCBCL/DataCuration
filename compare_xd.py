#!/usr/bin/env python

import numpy as np
import sys
import argparse
import scipy
import pdb

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.offsetbox import TextArea, DrawingArea, OffsetImage, AnnotationBbox
from collections import OrderedDict

from get_database import Database, Post, input_args
from simulation_io import collect_results_db

sys.path.append('/home/pfaller/work/repos/SimVascular/Python/site-packages/')

import sv_1d_simulation as oned

def add_image(db, geo, fig):
    im = plt.imread(db.get_png(geo))
    newax = fig.add_axes([-0.22, 0.3, 0.3, 0.3], anchor='NE')#, zorder=-1
    newax.imshow(im)
    newax.axis('off')


def plot_1d_3d_all(db, opt, geo, res, time):
    # get post-processing constants
    post = Post()

    # get 1d/3d map
    caps = db.get_surface_names(geo, 'caps')

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

            for c in caps:
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
    caps = db.get_surface_names(geo, 'caps')

    fig, ax = plt.subplots(len(post.fields), len(caps), figsize=(opt['w'], opt['h']), dpi=opt['dpi'], sharex=opt['sharex'], sharey=opt['sharey'])

    for i, f in enumerate(post.fields):
        for j, c in enumerate(caps):
            ax[i, j].grid(True)

            if opt['legend_row'] or i == 0:
                ax[i, j].set_title(c)
            if opt['legend_row'] or i == len(post.fields) - 1:
                ax[i, j].set_xlabel('Time [s]')
                ax[i, j].xaxis.set_tick_params(which='both', labelbottom=True)
            if opt['legend_col'] or j == 0:
                ax[i, j].set_ylabel(f.capitalize() + ' [' + post.units[f] + ']')
                ax[i, j].yaxis.set_tick_params(which='both', labelleft=True)

            lg = []
            for m in post.models:
                ax[i, j].plot(time['3d'], res[c][f][m + '_cap'] * post.convert[f], post.styles[m])
                lg.append(m)

            ax[i, j].legend(lg)

    add_image(db, geo, fig)
    # fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig(db.get_post_path(geo, 'caps'))
    plt.close(fig)


def plot_1d_3d_interior(db, opt, geo, res, time):
    # get post-processing constants
    post = Post()

    # get 1d/3d map
    caps = db.get_surface_names(geo, 'caps')

    fig, ax = plt.subplots(len(post.fields), len(caps), figsize=(opt['w'], opt['h']), dpi=opt['dpi'], sharex='col', sharey=opt['sharey'])

    # pick time step
    t_max = np.argmax(res['inflow']['flow']['3d_cap'])

    for i, f in enumerate(post.fields):
        for j, c in enumerate(caps):
            ax[i, j].grid(True)

            if opt['legend_row'] or i == 0:
                ax[i, j].set_title(c)
            if opt['legend_row'] or i == len(post.fields) - 1:
                ax[i, j].set_xlabel('Vessel path [cm]')
                ax[i, j].xaxis.set_tick_params(which='both', labelbottom=True)
            if opt['legend_col'] or j == 0:
                ax[i, j].set_ylabel(f.capitalize() + ' [' + post.units[f] + ']')
                ax[i, j].yaxis.set_tick_params(which='both', labelleft=True)

            lg = []
            for m in post.models:
                ax[i, j].plot(res[c][m + '_path'], res[c][f][m + '_int'][:, t_max] * post.convert[f], post.styles[m])
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


def calc_error(db, opt, geo, res, time):
    # get post-processing constants
    post = Post()

    # get 1d/3d map
    caps = db.get_surface_names(geo, 'caps')

    # remove inflow (prescribed flow)
    caps.remove('inflow')

    # all differences over time
    delta = {}
    for f in post.fields:
        delta[f] = {}
        for c in caps:
            delta[f][c] = {}

            # interpolate 3d results to 1d path
            # interp = scipy.interpolate.interp1d(res['path'][c]['3d'], res[c][f]['3d_int'])
            # res_3d = interp(res['path'][c]['1d'])

            # interpolate 1d results to 3d path
            interp = scipy.interpolate.interp1d(res[c]['1d_path'], res[c][f]['1d_int'].T)
            res_1d = interp(res[c]['3d_path'])

            # difference in interior
            # delta[f][c]['int'] = np.linalg.norm(res_3d - res[c][f]['1d_int'], axis=1) * post.convert[f]
            delta[f][c]['int'] = np.linalg.norm(res_1d - res[c][f]['3d_int'].T, axis=1) * post.convert[f]

            # difference at caps
            delta[f][c]['caps'] = (res[c][f]['3d_cap'] - res[c][f]['1d_cap']) * post.convert[f]

    # mean/max difference over time
    err = {}
    for f in post.fields:
        err[f] = {}
        for c in caps:
            err[f][c] = {}
            for name in delta[f][c].keys():
                err[f][c][name] = {}
                err[f][c][name]['max'] = np.max(np.abs(delta[f][c][name]))
                err[f][c][name]['mean'] = np.mean(np.abs(delta[f][c][name]))

    for f in post.fields:
        err[f]['all'] = {}
        for name in delta[f][c].keys():
            err[f]['all'][name] = {}
            err[f]['all'][name]['max'] = np.max([err[f][c][name]['max'] for c in caps])
            err[f]['all'][name]['mean'] = np.mean([err[f][c][name]['mean'] for c in caps])

    db.add_1d_3d_comparison(geo, err)


def main(db, geometries):
    for geo in geometries:
        print('Comparing geometry ' + geo)

        # read results
        res, time = collect_results_db(db, geo)
        if res is None:
            continue

        # plot options
        opt = {'legend_col': False,
               'legend_row': False,
               'sharex': True,
               'sharey': 'row',
               'dpi': 200,
               'w': 1 * (len(db.get_surface_names(geo)) * 2 + 2),
               'h': 2 * (len(Post().fields) * 1 + 1)}

        # calculate error
        calc_error(db, opt, geo, res, time)

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
