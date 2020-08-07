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

        # number of cardiac cycles
        n_cycle = time['3d_rerun_n_cycle']

        # get caps
        caps = get_caps_db(db, geo)

        for i, f in enumerate(post.fields):
            # plot location
            pos = (i, j)
            err = []
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

            # mean error over all outlets
            err_all = np.mean(err, axis=1)

            # how much does the error decrease in each cycle?
            slope = np.mean(np.diff(np.log10(err_all)))

            # error threshold for a converged solution
            thresh = 1.0e-3
            # pdb.set_trace()

            # how many cycles are needed to reach convergence
            i_conv = np.where(np.all(err < thresh, axis=1))[0]
            if not i_conv.any():
                i_conv = n_cycle + int((np.log10(thresh) - np.log10(err_all[-1]))/slope + 1.0)
            else:
                i_conv = i_conv[0] + 1

            # calculate asymptotic value
            f_conv = norm
            for m in range(10000 - n_cycle):
                f_conv += err_all[-1] * norm * 10**((m + 1) * slope)

            # print errors
            max_err = np.max(err[-1])
            max_outlet = db.get_cap_names(geo)[list(caps.keys())[np.argmax(err[-1])]]
            f_delta = (f_conv - norm) * post.convert[f]
            print(geo, f[:4], '\tdelta to asymptotic ' + '{:2.1e}'.format(f_delta) + ' [' + post.units[f] + ']',
                  '\tconverged in ' + str(i_conv) + ' cycles',
                  '\t{:.2e}'.format(max_err * 100) + '%', ' at outlet ' + max_outlet)

            # plot data points
            ax[pos].plot(np.arange(2, len(err) + 2), err, 'o-')

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
            # ax[pos].set_ylim([0.01, 0.1])
            ax[pos].yaxis.set_major_formatter(mtick.PercentFormatter(1.0, 3))

    plt.subplots_adjust(right=0.8)
    fname = 'outlets2.png'
    fpath = os.path.join(db.get_statistics_dir(), fname)
    fig.savefig(fpath, bbox_inches='tight')
    plt.close(fig)


def main(db, geometries, params):
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
