#!/usr/bin/env python

import os
import sys
import re
import vtk
import argparse
import pdb

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

from collections import defaultdict
from scipy.interpolate import interp1d

from vtk.util.numpy_support import numpy_to_vtk as n2v
from vtk.util.numpy_support import vtk_to_numpy as v2n

sys.path.append('..')
from get_database import Database, Post, input_args
from vtk_functions import read_geo, write_geo, get_all_arrays
from simulation_io import get_dict, get_caps_db, collect_results_db_3d_3d, collect_results_db_1d_3d, collect_results_db_0d
from get_sv_project import coronary_sv_to_oned
from get_statistics_bc import collect_errors

fsize = 20
plt.rcParams.update({'font.size': fsize})
f_out = '/home/pfaller/work/paper/asymptotic'
# plt.style.use('dark_background')


def make_err_plot(db, geo, ax, pos, m, f, p, res_m, errors, time, title_study=''):
    post = Post()
    t = 0

    # get caps
    caps = get_caps_db(db, geo)
    del caps['inflow']

    # get boundary conditions
    bcs = db.get_bcs(geo)['bc']
    bct = db.get_bcs(geo)['bc_type']

    # get time
    n_cycle = time[m + '_n_cycle']

    # plot times
    times_plot = []
    times_all = []
    for k in range(1, n_cycle + 1):
        times_all += [np.where(time[m + '_i_cycle_' + str(k)])[0]]
        times_plot += [np.where(time[m + '_i_cycle_' + str(k)])[0][t]]
    times_all = np.array(times_all)

    # add last time step
    if t == 0:
        times_plot += [np.where(time[m + '_i_cycle_' + str(n_cycle)])[0][-1]]

    cycles = np.arange(len(times_plot))
    if t != 0:
        cycles += 1

    # error threshold for a converged solution
    thresh = 1.0e-2
    e_thresh = 'asymptotic'

    # collect results
    res_m_all = []
    res_m_t = []
    res_m_m = []
    res_0d_t = []
    res_qm_t = []
    for c, br in caps.items():
        res_m_all += [res_m[br][f][m + '_all']]
        res_m_t += [res_m[br][f][m + '_all'][times_plot]]
        # res_0d_t += [interp1d(res_0d[br]['t'], res_0d[br]['p'], fill_value='extrapolate')(time[m][t]).tolist()]
        res_0d_t += [res_m_t[-1][-1]]
        res_m_m += [np.mean(res_m[br][f][m + '_all'][times_all], axis=1)]

        if bct[c] == 'resistance':
            resistance = bcs[c]['R']
        elif bct[c] == 'rcr':
            resistance = bcs[c]['Rd'] + bcs[c]['Rp']
        elif bct[c] == 'coronary':
            cor = coronary_sv_to_oned(bcs[c])
            resistance = cor['Ra1'] + cor['Ra2'] + cor['Rv1']
        res_qm_t += [resistance * np.mean(res_m[br]['flow'][m + '_cap'])]

    # make plot
    xticks = [1, 16]
    x_min = 1
    if p == 'cycle':
        title = 'Solution'
        x_min = 0
        xticks += [0]
        x = time[m + '_all'] / time[m][-1]
        y = np.array(res_m_all).T * post.convert[f]
        ylabel = f.capitalize() + ' [' + post.units[f] + ']'
    elif p == 'cycle_norm':
        title = 'Normalized solution'
        x_min = 0
        xticks += [0]
        x = time[m + '_all'] / time[m][-1]
        y = np.array(res_m_all).T / np.array(res_m_m)[:, -1]
        ylabel = f.capitalize() + ' [-]'
    elif p == 'initial':
        title = 'Initial values'
        x = cycles
        y = np.array(res_m_t).T * post.convert[f]
        ylabel = 'Initial ' + f + ' [' + post.units[f] + ']'
    elif p == 'mean':
        title = 'Mean cycle solution'
        x = cycles[1:]
        y = np.array(res_m_m).T * post.convert[f]
        y /= y[-1]
        xticks += [1]
        # ylabel = 'Mean ' + f + ' [' + post.units[f] + ']'
        ylabel = 'Mean ' + f + ' [-]'
    elif p in 'cyclic':
        title = 'Cyclic error'
        x = cycles[2:]
        y = errors['cyclic'][f]
        xticks += [2]
        ylabel = 'Cyclic ' + f + ' error [-]'
    elif p in 'asymptotic':
        title = 'Asymptotic error'
        x = cycles[1:-1]
        y = errors['asymptotic'][f][:-1]
        xticks += [1]
        ylabel = 'Asymptotic ' + f + ' error [-]'
    else:
        title = ''
        x = np.nan
        y = np.nan

    # converged time step
    conv = np.where(np.all(errors[e_thresh][f] < thresh, axis=1))[0]
    if not conv.any():
        i_conv = -1
    else:
        i_conv = np.min(conv)
    if e_thresh == 'cyclic':
        i_conv += 2
    elif e_thresh == 'asymptotic':
        i_conv += 1
        print(f + ' ' + str(i_conv))
    xticks += [i_conv]

    # plot
    if p == 'initial' or p == 'mean':
        plt.gca().set_prop_cycle(plt.rcParams['axes.prop_cycle'])
        # ax[pos].plot([cycles[0], cycles[-1]], np.vstack((y[-1], y[-1])), '--')
        ax[pos].plot([0, 999], [1, 1], 'k-')
        ax[pos].set_ylim([0, 1.2])
        # ax.plot([cycles[0], cycles[-1]], np.vstack((res_qm_t, res_qm_t)) * post.convert[f], '--')

    ax[pos].plot(x, y, '-')
    ax[pos].axvline(x=i_conv, color='k')
    ax[pos].set_ylabel(ylabel)
    x_eps = 0.5 * np.max(xticks) / 20
    ax[pos].set_xlim([x_min - x_eps, np.max(xticks) + x_eps])
    # ax[pos].set_xlim([0, np.max(x)])
    ax[pos].set_xticks(xticks)
    ax[pos].grid('both')

    if p in ['cyclic', 'asymptotic']:
        ax[pos].set_yscale('log')
        ax[pos].set_ylim([1.0e-4, 1])
        # ax[pos].yaxis.set_major_formatter(mtick.PercentFormatter(1.0, 2))
    if p == e_thresh:
        ax[pos].plot([0, 999], [thresh, thresh], 'k-')
    if pos[0] == 1:
        ax[pos].set_xlabel('Cardiac cycle [-]')
    if title_study:
        ax[pos].set_title(title_study)
    elif pos[0] == 0:
        ax[pos].set_title(title)
    ax[pos].set_prop_cycle(plt.rcParams['axes.prop_cycle'])

    return y


def plot_pressure(study, geo):
    # plot settings
    m = '3d_rerun'
    # m = '1d'
    # m = '0d'
    fields = ['pressure', 'flow']
    # fields = ['pressure']
    # comparisons = ['cycle', 'initial', 'cyclic', 'asymptotic']
    # comparisons = ['cycle', 'mean', 'asymptotic']
    # comparisons = ['cycle', 'mean', 'cyclic', 'asymptotic']
    comparisons = ['cycle', 'mean', 'cyclic', 'asymptotic']

    # get database
    db = Database(study)

    # res_m, time = collect_results_db_1d_3d(db, geo)
    if m == '0d':
        res_m, time = collect_results_db_0d(db, geo)
    elif m == '3d_rerun':
        res_m, time = collect_results_db_3d_3d(db, geo)
    else:
        return

    if res_m is None:
        return

    if os.path.exists(db.get_bc_0D_path(geo, m)):
        res_0d = np.load(db.get_bc_0D_path(geo, m), allow_pickle=True).item()
    else:
        return

    print(geo)
    errors = collect_errors(res_m, res_0d, time, m)

    fig, ax = plt.subplots(len(fields), len(comparisons), figsize=(5 * len(comparisons), 5 * len(fields)), dpi=300)

    c_res = defaultdict(dict)
    for j, f in enumerate(fields):
        # for i, p in zip([0, 1, 2, 2], comparisons):
        for i, p in enumerate(comparisons):
            if len(fields) == 1:
                pos = i
            else:
                pos = (j, i)

            c_res[f][p] = make_err_plot(db, geo, ax, pos, m, f, p, res_m, errors, time)

    fname = 'convergence_' + study + '_' + geo
    if len(fields) == 1:
        fname += '_' + f
    fpath = os.path.join(f_out, fname + '.png')
    # plt.subplots_adjust(right=0.8)
    fig.tight_layout()
    fig.savefig(fpath, bbox_inches='tight')
    plt.close(fig)

    # time constants (analytical)
    tau_ana_dic = db.get_time_constants(geo)
    caps = get_caps_db(db, geo)
    del caps['inflow']
    tau_ana = np.array([tau_ana_dic[c] for c in caps])

    # factor between asymptotic and cyclic error (analytical)
    alpha_ana = 1 / (np.exp(1 / tau_ana) - 1)

    for f in fields:
        # time constants (numerical)
        n_max = 5
        tau_num = 1 / np.mean(-np.diff(np.log(c_res[f]['cyclic'][:n_max]), axis=0), axis=0)

        # factor between asymptotic and cyclic error (numerical)
        alpha_num = c_res[f]['asymptotic'][1:] / c_res[f]['cyclic'][:-1]

        # find convergence dominating outlet
        # i_conv = np.argmin(np.abs(tau_num / tau_ana - 1))
        # cap_conv = list(caps.keys())[i_conv]

        fig, ax = plt.subplots(figsize=(6, 4), dpi=300)
        ax.plot(alpha_num)
        # ax.plot(np.tile(alpha_ana, (alpha_num.shape[0], 1)), 'k')
        ax.plot([0, len(alpha_num)], np.repeat(np.mean(alpha_ana), 2), 'k')
        ax.grid('both')
        fpath = os.path.join(f_out, 'alpha', 'alpha_' + study + '_' + geo + '_' + f + '.png')
        fig.savefig(fpath, bbox_inches='tight')
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(6, 4), dpi=300)
        ax.plot(tau_ana, 'o')
        ax.plot(tau_num, 'x')
        ax.grid('both')
        ax.legend(['analytical', 'numerical'])
        ax.set_xlabel('Outlet')
        ax.set_ylabel('Time constant [s]')
        fpath = os.path.join(f_out, 'tau', 'tau_' + study + '_' + geo + '_' + f + '.png')
        fig.savefig(fpath, bbox_inches='tight')
        plt.close(fig)
        # pdb.set_trace()


def plot_pressure_studies(geo):
    # plot settings
    m = '3d_rerun'
    fields = ['pressure', 'flow']
    studies = ['ini_zero', 'ini_steady', 'ini_1d_quad']
    # studies = ['ini_zero', 'ini_irene', 'ini_1d_quad']
    comparison = 'asymptotic'
    # comparison = 'mean'

    print(geo)
    fig, ax = plt.subplots(len(fields), len(studies), figsize=(8 * len(studies), 5 * len(fields)), dpi=300, sharey=True)

    for j, f in enumerate(fields):
        for i, p in enumerate(studies):
            # get database
            db = Database(p)

            if os.path.exists(db.get_bc_0D_path(geo, m)):
                res_0d = np.load(db.get_bc_0D_path(geo, m), allow_pickle=True).item()
            else:
                return

            res_m, time = collect_results_db_3d_3d(db, geo)
            errors = collect_errors(res_m, res_0d, time, m)

            if len(fields) == 1:
                pos = i
            else:
                pos = (j, i)

            make_err_plot(db, geo, ax, pos, m, f, comparison, res_m, errors, time, title_study=p)

    fname = 'comparison_' + geo
    if len(fields) == 1:
        fname += '_' + f
    fpath = os.path.join(f_out, fname + '.png')
    # plt.subplots_adjust(right=0.8)
    fig.tight_layout()
    fig.savefig(fpath, bbox_inches='tight')
    plt.close(fig)


def main(db, geo):
    for g in geo:
        # plot_pressure_studies(g)
        plot_pressure(db.study, g)


if __name__ == '__main__':
    descr = 'Make plots for 3D-1D-0D paper'
    d, g, _ = input_args(descr)
    main(d, g)
