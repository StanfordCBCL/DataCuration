#!/usr/bin/env python

import numpy as np
import argparse
import scipy
import pdb

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.offsetbox import TextArea, DrawingArea, OffsetImage, AnnotationBbox
from collections import OrderedDict

from common import input_args
from get_database import Database, Post


def add_image(db, geo, fig):
    im = plt.imread(db.get_png(geo))
    newax = fig.add_axes([-0.25, 0.3, 0.3, 0.3], anchor='NE')#, zorder=-1
    newax.imshow(im)
    newax.axis('off')


def plot_1d_3d_all(db, opt, geo, res, time):
    # get post-processing constants
    post = Post()

    # get 1d/3d map
    caps = db.get_xd_map(geo)

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

            for c in caps.keys():
                ax[i, j].plot(time[m + '_all'], res[f][c][m + '_all'] * post.convert[f])

    add_image(db, geo, fig)
    fig.savefig(db.get_post_path(geo, 'all'))
    plt.close(fig)


def plot_1d_3d_cyclic(db, opt, geo, res, time):
    # get post-processing constants
    post = Post()

    # get 1d/3d map
    caps = db.get_xd_map(geo)

    fig, ax = plt.subplots(len(post.fields), len(caps), figsize=(opt['w'], opt['h']), dpi=opt['dpi'], sharex=opt['sharex'], sharey=opt['sharey'])

    for i, f in enumerate(post.fields):
        for j, c in enumerate(caps.keys()):
            ax[i, j].grid(True)

            if opt['legend_row'] or i == 0:
                ax[i, j].set_title(c)
            if opt['legend_row'] or i == len(post.fields) - 1:
                ax[i, j].set_xlabel('Time [s]')
            if opt['legend_col'] or j == 0:
                ax[i, j].set_ylabel(f.capitalize() + ' [' + post.units[f] + ']')

            res_split = np.split(res[f][c]['1d_all'][:time['step_cycle'] * time['n_cycle']], time['n_cycle'])
            for r in res_split:
                ax[i, j].plot(time['1d'], r * post.convert[f], post.styles['1d'])

            ax[i, j].plot(time['3d'], res[f][c]['3d'] * post.convert[f], post.styles['3d'])

    add_image(db, geo, fig)
    fig.savefig(db.get_post_path(geo, 'cyclic'))
    plt.close(fig)


def plot_1d_3d_caps(db, opt, geo, res, time):
    # get post-processing constants
    post = Post()

    # get 1d/3d map
    caps = db.get_xd_map(geo)

    fig, ax = plt.subplots(len(post.fields), len(caps), figsize=(opt['w'], opt['h']), dpi=opt['dpi'], sharex=opt['sharex'], sharey=opt['sharey'])

    for i, f in enumerate(post.fields):
        for j, c in enumerate(caps.keys()):
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
                ax[i, j].plot(time['3d'], res[f][c][m] * post.convert[f], post.styles[m])
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
    caps = db.get_xd_map(geo)

    fig, ax = plt.subplots(len(post.fields), len(caps), figsize=(opt['w'], opt['h']), dpi=opt['dpi'], sharex=opt['sharex'], sharey=opt['sharey'])

    # pick time step
    t_max = np.argmax(res['flow']['inflow']['3d'])

    for i, f in enumerate(post.fields):
        for j, c in enumerate(caps.keys()):
            ax[i, j].grid(True)

            if opt['legend_row'] or i == 0:
                ax[i, j].set_title(c)
            if opt['legend_row'] or i == len(post.fields) - 1:
                ax[i, j].set_xlabel('Vessel path [1]')
                ax[i, j].xaxis.set_tick_params(which='both', labelbottom=True)
            if opt['legend_col'] or j == 0:
                ax[i, j].set_ylabel(f.capitalize() + ' [' + post.units[f] + ']')
                ax[i, j].yaxis.set_tick_params(which='both', labelleft=True)

            lg = []
            for m in post.models:
                ax[i, j].plot(res['path'][c][m], res[f][c][m + '_int'][t_max] * post.convert[f], post.styles[m])
                lg.append(m)

            ax[i, j].legend(lg)

    add_image(db, geo, fig)
    # fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig(db.get_post_path(geo, 'interior'))
    plt.close(fig)


def plot_1d_3d_paper(db, opt, geo, res, time):
    # get post-processing constants
    post = Post()

    # get 1d/3d map
    caps = db.get_xd_map(geo)

    names = {'inflow': 'Inflow', 'aorta_outflow': 'Aorta', 'Lt-carotid_segs_final': 'Left Carotid',
             'rt-carotid_segs_final': 'Right Carotid', 'btrunk_segs_final': 'Right Subclavian',
             'subclavian_segs_final': 'Left Subclavian'}
    # myorder = ['inflow', 'aorta_outflow', 'Lt-carotid_segs_final', 'rt-carotid_segs_final', 'subclavian_segs_final', 'btrunk_segs_final']
    myorder = ['aorta_outflow', 'rt-carotid_segs_final', 'btrunk_segs_final']

    # layout
    nx = 2
    ny = 3

    # plot caps
    fig, ax = plt.subplots(ny, nx, figsize=(9, 8), dpi=opt['dpi'], sharex=True, sharey='col')
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
                ax[pos].plot(time['3d'], res[f][c][m] * post.convert[f], post.styles[m], color=post.colors[m])
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
                ax[pos].plot(res['path'][c][m], res[f][c][m + '_int'][t_max] * post.convert[f], post.styles[m], color=post.colors[m])

        fig.savefig(db.get_post_path(geo, 'paper_interior_' + f), bbox_inches='tight', pad_inches=0)
        plt.close(fig)


def calc_error(db, opt, geo, res, time):
    # get post-processing constants
    post = Post()

    # get 1d/3d map
    caps = db.get_xd_map(geo)

    # all differences over time
    delta = {}
    for f in post.fields:
        delta[f] = {}
        for c in caps.keys():
            delta[f][c] = {}

            # interpolate 3d results to 1d path
            interp = scipy.interpolate.interp1d(res['path'][c]['3d'], res[f][c]['3d_int'])
            res_3d = interp(res['path'][c]['1d'])

            # difference in interior
            delta[f][c]['int'] = np.linalg.norm(res_3d - res[f][c]['1d_int'], axis=1) * post.convert[f]

            # difference at caps
            delta[f][c]['caps'] = (res[f][c]['3d'] - res[f][c]['1d']) * post.convert[f]

    # mean/max difference over time
    err = {}
    for f in post.fields:
        err[f] = {}
        for c in caps.keys():
            err[f][c] = {}
            for name in delta[f][c].keys():
                err[f][c][name] = {}
                err[f][c][name]['max'] = np.max(np.abs(delta[f][c][name]))
                err[f][c][name]['mean'] = np.mean(np.abs(delta[f][c][name]))

        for name in delta[f][c].keys():
            err[f]['all'] = {}
            err[f]['all'][name] = {}
            err[f]['all'][name]['max'] = np.max([err[f][c][name]['max'] for c in caps.keys()])
            err[f]['all'][name]['mean'] = np.mean([err[f][c][name]['mean'] for c in caps.keys()])

    db.add_1d_3d_comparison(geo, err)


def main(db, geometries):
    for geo in geometries:
        print('Comparing geometry ' + geo)

        # read results
        res, time = db.get_results_xd(geo)
        if res is None:
            continue

        # plot options
        opt = {'legend_col': False,
               'legend_row': False,
               'sharex': True,
               'sharey': 'row',
               'dpi': 200,
               'w': 2 * (len(db.get_xd_map(geo).keys()) * 2 + 2),
               'h': 2 * (len(Post().fields) * 1 + 1)}

        # generate plots
        calc_error(db, opt, geo, res, time)

        print('plotting')
        # plot_1d_3d_all(db, opt, geo, res, time)
        # plot_1d_3d_caps(db, opt, geo, res, time)
        plot_1d_3d_interior(db, opt, geo, res, time)

        # plot_1d_3d_paper(db, opt, geo, res, time)

        # plot_1d_3d_cyclic(db, opt, geo, res, time)


if __name__ == '__main__':
    descr = 'Plot comparison of xd-results'
    d, g, _ = input_args(descr)
    main(d, g)
