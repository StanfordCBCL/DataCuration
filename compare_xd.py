#!/usr/bin/env python

import numpy as np
import argparse
import pdb

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.offsetbox import TextArea, DrawingArea, OffsetImage, AnnotationBbox

from common import input_args
from get_database import Database, Post


def add_image(db, geo, fig):
    im = plt.imread(db.get_png(geo))
    newax = fig.add_axes([-0.25, 0.3, 0.3, 0.3], anchor='NE')#, zorder=-1
    newax.imshow(im)
    newax.axis('off')


def plot_1d_3d_all(db, geo, res, time):
    # get post-processing constants
    post = Post()

    # get 1d/3d map
    caps = db.get_xd_map(geo)

    fig, ax = plt.subplots(len(post.fields), len(post.models), figsize=(20, 10), dpi=300, sharex='col', sharey='row')

    for i, f in enumerate(post.fields):
        for j, m in enumerate(post.models):
            ax[i, j].set_title(geo + ' ' + m + ' ' + f)
            ax[i, j].grid(True)

            if i == len(post.fields) - 1:
                ax[i, j].set_xlabel('Time [s]')
            if j == 0:
                ax[i, j].set_ylabel(f.capitalize() + ' [' + post.units[f] + ']')

            for c in caps.keys():
                ax[i, j].plot(time[m + '_all'], res[f][c][m + '_all'] * post.convert[f])

    add_image(db, geo, fig)
    fig.savefig(db.get_post_path(geo, 'all'))
    plt.close(fig)


def plot_1d_3d_cyclic(db, geo, res, time):
    # get post-processing constants
    post = Post()

    # get 1d/3d map
    caps = db.get_xd_map(geo)

    fig, ax = plt.subplots(len(post.fields), len(caps), figsize=(50, 10), dpi=300, sharex=True, sharey='row')

    for i, f in enumerate(post.fields):
        for j, c in enumerate(caps.keys()):
            ax[i, j].set_title(c)
            ax[i, j].grid(True)

            if i == len(post.fields) - 1:
                ax[i, j].set_xlabel('Time [s]')
            if j == 0:
                ax[i, j].set_ylabel(f.capitalize() + ' [' + post.units[f] + ']')

            res_split = np.split(res[f][c]['1d_all'][:time['step_cycle'] * time['n_cycle']], time['n_cycle'])
            for r in res_split:
                ax[i, j].plot(time['1d'], r * post.convert[f], post.styles['1d'])

            ax[i, j].plot(time['3d'], res[f][c]['3d'] * post.convert[f], post.styles['3d'])

    add_image(db, geo, fig)
    fig.savefig(db.get_post_path(geo, 'cyclic'))
    plt.close(fig)


def plot_1d_3d_caps(db, geo, res, time):
    # get post-processing constants
    post = Post()

    # get 1d/3d map
    caps = db.get_xd_map(geo)

    fig, ax = plt.subplots(len(post.fields), len(caps), figsize=(50, 10), dpi=300, sharex=True, sharey='row')

    for i, f in enumerate(post.fields):
        for j, c in enumerate(caps.keys()):
            ax[i, j].set_title(c)
            ax[i, j].grid(True)

            if i == len(post.fields) - 1:
                ax[i, j].set_xlabel('Time [s]')
            if j == 0:
                ax[i, j].set_ylabel(f.capitalize() + ' [' + post.units[f] + ']')

            lg = []
            for m in post.models:
                ax[i, j].plot(time['3d'], res[f][c][m] * post.convert[f], post.styles[m])
                lg.append(m)

            ax[i, j].legend(lg)

    add_image(db, geo, fig)
    fig.savefig(db.get_post_path(geo, 'caps'))
    plt.close(fig)


def plot_1d_3d_interior(db, geo, res, time):
    # get post-processing constants
    post = Post()

    # get 1d/3d map
    caps = db.get_xd_map(geo)

    fig, ax = plt.subplots(len(post.fields), len(caps), figsize=(50, 10), dpi=300, sharex=True, sharey='row')

    # pick time step
    t_max = np.argmax(res['flow']['inflow']['3d'])

    for i, f in enumerate(post.fields):
        for j, c in enumerate(caps.keys()):
            ax[i, j].set_title(c)
            ax[i, j].grid(True)

            if i == len(post.fields) - 1:
                ax[i, j].set_xlabel('Vessel path [1]')
            if j == 0:
                ax[i, j].set_ylabel(f.capitalize() + ' [' + post.units[f] + ']')

            lg = []
            for m in post.models:
                ax[i, j].plot(res['path'][c], res[f][c][m + '_int'][t_max] * post.convert[f], post.styles[m] + 'o')
                lg.append(m)

            ax[i, j].legend(lg)

    add_image(db, geo, fig)
    fig.savefig(db.get_post_path(geo, 'interior'))
    plt.close(fig)


def main(db, geometries):
    for geo in geometries:
        print('Comparing geometry ' + geo)

        # read results
        res, time = db.get_results_xd(geo)
        if res is None:
            continue

        # generate plots
        print('plotting')
        plot_1d_3d_all(db, geo, res, time)
        plot_1d_3d_caps(db, geo, res, time)
        # plot_1d_3d_cyclic(db, geo, res, time)
        plot_1d_3d_interior(db, geo, res, time)


if __name__ == '__main__':
    descr = 'Plot comparison of xd-results'
    d, g = input_args(descr)
    main(d, g)
