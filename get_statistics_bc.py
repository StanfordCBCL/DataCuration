#!/usr/bin/env python

import contextlib
import csv
import glob
import io
import os
import pdb
import re
import shutil
import sys
import argparse
import subprocess

import numpy as np
from collections import defaultdict

from vtk.util.numpy_support import numpy_to_vtk as n2v
from vtk.util.numpy_support import vtk_to_numpy as v2n

from get_database import Database, SimVascular, Post, input_args
from simulation_io import get_dict, get_caps_db, collect_results_db_3d_3d
from vtk_functions import read_geo, write_geo, collect_arrays
from compare_1d import plot_1d_3d_caps

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from matplotlib.offsetbox import OffsetImage, AnnotationBbox


def plot_error_spatial(db, geometries):
    # get post-processing constants
    post = Post()
    fields = post.fields
    fields.remove('area')

    # set global plot options
    fig, ax = plt.subplots(len(fields), len(geometries), figsize=(16, 8), dpi=200, sharey=True, sharex=True)#
    plt.rcParams['axes.linewidth'] = 2

    # get 1d/3d map
    for j, geo in enumerate(geometries):
        # get results
        res, time = collect_results_db_3d_3d(db, geo)

        # get caps
        caps = get_caps_db(db, geo)

        for i, f in enumerate(post.fields):
            # plot location
            pos = (i, j)
            err = []
            cycles = []
            for c, br in caps.items():
                # get results for branch at all time steps
                res_br = res[br][f]['3d_rerun_all']

                # get last cycle
                res_last = res[br][f]['3d_rerun_cap']

                # normalize
                if f == 'pressure':
                    norm = np.mean(res_last)
                elif f == 'flow':
                    norm = np.max(res_last) - np.min(res_last)

                # get start and end step of each cardiac cycle
                n_cycle = time['3d_rerun_n_cycle']
                cycle_range = []
                for k in range(1, n_cycle + 1):
                    i_cycle = np.where(time['3d_rerun_i_cycle_' + str(k)])[0]
                    cycle_range += [i_cycle]

                # calculate cycle error
                err_br = []
                for k in range(1, n_cycle):
                    t_prev = cycle_range[k - 1]
                    t_this = cycle_range[k]
                    diff = np.mean(res_br[t_this] - res_br[t_prev])
                    err_br += [np.abs(diff / norm)]
                err += [err_br]
            err = np.array(err).T

            # plot data points
            ax[pos].plot(np.arange(2, len(err) + 2), err, 'o-')

            # print errors
            max_err = np.max(err[-1])
            max_outlet = db.get_cap_names(geo)[list(caps.keys())[np.argmax(err[-1])]]
            print(geo, f[:4], '{:.2e}'.format(max_err * 100) + '%', 'at outlet ' + max_outlet)

            # set plot options
            if i == 0:
                ax[pos].set_title(geo)
            if i == len(fields) - 1:
                ax[pos].set_xlabel('Cardiac cycle')
            if j == 0:
                ax[pos].set_ylabel(f.capitalize() + ' cyclic error')
            ax[pos].grid(True)
            ax[pos].ticklabel_format(axis='y')
            ax[pos].set_yscale('log')
            ax[pos].set_ylim([1.0e-5, 1])
            ax[pos].yaxis.set_major_formatter(mtick.PercentFormatter(1.0, 3))

    # lgd = ax[(0, 0)].legend([], bbox_to_anchor=(-0.2, 0), loc='right') #
    # fig.subplots_adjust(left=0.2)
    # plt.subplots_adjust(left=0.07, right=0.93, wspace=0.25, hspace=0.35)
    plt.subplots_adjust(right=0.8)
    fname = 'outlets.png'
    fpath = os.path.join(db.get_statistics_dir(), fname)
    fig.savefig(fpath, bbox_inches='tight')#, bbox_extra_artists=(lgd,)
    plt.close(fig)


def plot_error_caps(db, geo):
    res, time = collect_results_db_3d_3d(db, geo)

    # plot options
    opt = {'legend_col': False,
           'legend_row': False,
           'sharex': True,
           'sharey': 'row',
           'dpi': 200,
           'w': 1 * (len(db.get_surface_names(geo)) * 3 + 4),
           'h': 2 * (len(Post().fields) * 1 + 2)}

    plot_1d_3d_caps(db, opt, geo, res, time)


def main(db, geometries, params):
    # read bcs
    for geo in geometries:
        f_path = db.get_3d_flow_rerun_bc(geo)
        if not os.path.exists(f_path):
            continue
        print('Plotting ' + geo)

        # plot_error_caps(db, geo)

    geos = []
    for geo in geometries:
        f_path = db.get_3d_flow_rerun_bc(geo)
        if not os.path.exists(f_path):
            continue
        geos += [geo]

    geos = ['0003_0001', '0107_0001', '0067_0001']
    plot_error_spatial(db, geos)


if __name__ == '__main__':
    descr = 'Get 3D-3D statistics'
    d, g, p = input_args(descr)
    main(d, g, p)
