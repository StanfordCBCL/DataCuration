#!/usr/bin/env python

import numpy as np
import os
import sys
import pdb

from collections import defaultdict
from scipy.interpolate import CubicSpline, interp1d
from scipy.optimize import minimize

import vtk
from vtk.util.numpy_support import vtk_to_numpy as v2n
from vtk.util.numpy_support import numpy_to_vtk as n2v

from get_database import input_args, Database, Post, SimVascular
from vtk_functions import read_geo, write_geo
from get_bc_integrals import integrate_surfaces, integrate_bcs
from simulation_io import get_caps_db, collect_results, collect_results_db_3d_3d, get_dict
from compare_1d import add_image
from get_bcs import get_in_model_units
from get_sv_project import coronary_sv_to_oned
from bc_0d import run_rcr, run_coronary

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick


def get_cycle(f, n_cycle):
    return np.hstack((f, np.tile(f[1:], n_cycle - 1)))


def cont_func(time, value, n_cycle):
    # repeat time for all cycles
    time_cycle = time
    for i in range(n_cycle):
        time_cycle = np.hstack((time_cycle, time[1:] + (i + 1) * time[-1]))

    # repeat value for all cycles
    value_cycle = np.hstack((value, np.tile(value[1:], n_cycle)))

    # create continues function
    fun = lambda t: interp1d(time_cycle, value_cycle)(t)

    # check if function matches origional values
    assert np.sum((fun(time) - value) ** 2) < 1e-16, 'function does not interpolate data'

    return fun, time_cycle, value_cycle


def run_0d_cycles(flow, time, p, distal_pressure, n_step=100, n_rcr=40):
    # number of cycles from rcr time constant
    if 'Rd' in p:
        t_rcr = p['C'] * p['Rd']
    elif 'R1' in p:
        t_rcr = p['C2'] * p['R2']
    else:
        raise ValueError('Unknown boundary conditions')
    n_cycle = np.max([int((t_rcr * n_rcr) // time[-1]), n_rcr])

    # 0d time
    t0 = 0.0  # best to set this as zero
    tf = time[-1] * n_cycle
    number_of_time_steps = n_step * n_cycle + 1
    time_0d = np.linspace(t0, tf, number_of_time_steps)

    # continuous inflow function
    Qfunc, _, _ = cont_func(time, flow, n_cycle)

    # output last cycle
    t_last = time[-1] * (n_cycle - 1)
    i_last = np.arange(n_step * (n_cycle - 1), n_step * n_cycle + 1, 1)
    i_prev = np.arange(n_step * (n_cycle - 2), n_step * (n_cycle - 1) + 1, 1)
    t_out = time_0d[i_last] - t_last

    # run 0d simulation
    if 'Rd' in p:
        p_0d = run_rcr(Qfunc, time_0d, p, distal_pressure)
    elif 'R1' in p:
        _, p_v_time, p_v_pres = cont_func(distal_pressure[:, 0], distal_pressure[:, 1], 3)
        p_0d = run_coronary(Qfunc, time_0d, p, p_v_time, p_v_pres, time[-1])
    else:
        raise ValueError('Unknown boundary conditions')

    # get last cycle
    p_out = p_0d[i_last - 1]

    # check if solution is periodic
    # delta_p = np.abs(np.mean(p_0d[i_last - 1] - p_0d[i_prev - 1]) / np.mean(p_0d[i_last - 1]))
    delta_p = np.abs(p_out[-1] - p_out[0]) / (np.max(p_out) - np.min(p_out))
    assert delta_p < 1.0e-9, 'solution not periodic. diff=' + str(delta_p)

    return t_out, p_out


def compare_0d(db, geo, res, time, m):
    # get boundary conditions
    bc_def, params = db.get_bcs(geo)
    bc_type, err = db.get_bc_type(geo)

    inlet_time = time[m]

    # get outlets
    caps = get_caps_db(db, geo)
    outlets = {}
    for cp, br in caps.items():
        if 'inflow' not in cp:
            outlets[cp] = br

    res_bc = defaultdict(dict)
    for j, (cp, br) in enumerate(outlets.items()):
        # cap bcs
        bc = bc_def['bc'][cp]
        t = bc_type[cp]

        # bc inlet flow
        inlet_flow = res[br]['flow'][m + '_cap']

        # select boundary condition
        p = {}
        if t == 'rcr':
            p['Rp'] = get_in_model_units(params['sim_units'], 'R', bc['Rp'])
            p['C'] = get_in_model_units(params['sim_units'], 'C', bc['C'])
            p['Rd'] = get_in_model_units(params['sim_units'], 'R', bc['Rd'])
            if 'Po' in bc:
                rcr_po = get_in_model_units(params['sim_units'], 'P', bc['Po'])
            else:
                rcr_po = 0.0

            res_bc[br]['t'], res_bc[br]['p'] = run_0d_cycles(inlet_flow, inlet_time, p, rcr_po)
        elif t == 'resistance':
            r_res = get_in_model_units(params['sim_units'], 'R', bc['R'])
            r_po = get_in_model_units(params['sim_units'], 'P', bc['Po'])

            res_bc[br]['t'] = inlet_time
            res_bc[br]['p'] = r_po + r_res * inlet_flow
        elif t == 'coronary':
            if not bc_def['coronary']:
                continue

            cor = coronary_sv_to_oned(bc)
            p['R1'] = get_in_model_units(params['sim_units'], 'R', cor['Ra1'])
            p['R2'] = get_in_model_units(params['sim_units'], 'R', cor['Ra2'])
            p['R3'] = get_in_model_units(params['sim_units'], 'R', cor['Rv1'])
            p['C1'] = get_in_model_units(params['sim_units'], 'C', cor['Ca'])
            p['C2'] = get_in_model_units(params['sim_units'], 'C', cor['Cc'])

            p_v_time = bc_def['coronary'][bc['Pim']][:, 0]
            p_v_pres = bc_def['coronary'][bc['Pim']][:, 1]
            p_v = get_in_model_units(params['sim_units'], 'P', p_v_pres)

            res_bc[br]['t'], res_bc[br]['p'] = run_0d_cycles(inlet_flow, inlet_time, p, np.vstack((p_v_time, p_v)).T)

    return res_bc


def check_bc(db, geo, plot_rerun=True):
    # get post-processing constants
    post = Post()

    # collect results
    if not os.path.exists(db.get_3d_flow_rerun(geo)):
        return
    res, time = collect_results_db_3d_3d(db, geo)
    if res is None:
        return

    print('Plotting ' + geo)

    use_bc = False
    if use_bc:
        rerun_name = '3d_rerun_bc'
    else:
        rerun_name = '3d_rerun'

    if plot_rerun and rerun_name + '_cap' in res[0]['flow']:
        m = rerun_name
    else:
        m = '3d'
    inlet_time = time[m]

    # get 0d results
    res_0d = compare_0d(db, geo, res, time, m)

    # get outlets
    caps = get_caps_db(db, geo)
    outlets = {}
    for cp, br in caps.items():
        if 'inflow' not in cp:
            outlets[cp] = br

    # bounbdary condition types
    bc_type, err = db.get_bc_type(geo)

    # get cap names
    names = db.get_cap_names(geo)

    dpi = 300
    if len(outlets) > 50:
        dpi //= 4

    fig, ax = plt.subplots(1, len(outlets), figsize=(len(outlets) * 3 + 4, 6), dpi=dpi, sharey=True)
    f = 'pressure'

    res_bc = defaultdict(dict)
    errors = []
    for j, (cp, br) in enumerate(outlets.items()):
        t = bc_type[cp]

        # plot settings
        ax[j].grid(True)
        ax[j].set_title(names[cp])
        ax[j].set_xlabel('Time [s]')
        ax[j].xaxis.set_tick_params(which='both', labelbottom=True)
        if j == 0:
            ax[j].set_ylabel(f.capitalize() + ' [' + post.units[f] + ']')
            ax[j].yaxis.set_tick_params(which='both', labelleft=True)

        # plot bcs
        ax[j].plot(inlet_time, res[br]['pressure'][m + '_cap'] * post.convert[f], post.styles[m], color=post.colors[m])
        ax[j].plot(res_0d[br]['t'], res_0d[br]['p'] * post.convert[f], 'k--')

        # legend
        ax[j].legend([rerun_name.upper(), '0D ' + t.upper()])

        # calculate error
        diff = interp1d(res_0d[br]['t'], res_0d[br]['p'], fill_value='extrapolate')(inlet_time) - res[br]['pressure'][m + '_cap']
        err = np.mean(np.abs(diff)) / np.mean(res[br]['pressure'][m + '_cap'])
        errors += [err]

        # save to file
        res_bc['time'] = res_0d[br]['t']
        res_bc['pressure'][br] = res_0d[br]['p']

    max_err = np.max(errors) * 100
    max_outlet = db.get_cap_names(geo)[list(outlets.keys())[np.argmax(errors)]]

    print(geo + ' err=' + '{:05.2f}'.format(max_err) + '% at outlet ' + max_outlet)

    # save figure
    add_image(db, geo, fig)
    if plot_rerun:
        f_out = db.get_post_path(geo, 'bcs')
    else:
        f_out = db.get_bc_comparison_path(geo)
    fig.savefig(f_out, bbox_inches='tight')
    plt.close(fig)

    # save pressure curves
    np.save(db.get_bc_0D_path(geo), res_bc)

    # add error to log
    if m == '3d':
        db.add_bc_err(geo, max_err)


def plot(db, geometries):
    # read all errors
    errors = get_dict(db.get_bc_err_file())

    # color by category
    colors = {'Cerebrovascular': 'k',
              'Coronary': 'r',
              'Aortofemoral': 'm',
              'Pulmonary': 'c',
              'Congenital Heart Disease': 'y',
              'Aorta': 'b',
              'Animal and Misc': '0.75'}

    geo = []
    err = []
    col = []
    for g, e in errors.items():
        if g in geometries:
            geo += [g]
            err += [e]
            col += [colors[db.get_params(g)['deliverable_category']]]

    # sort according to error
    order = np.argsort(err)
    order = np.argsort(geo)

    geo = np.array(geo)[order]
    err = np.array(err)[order]
    col = np.array(col)[order]

    geo_str = '['
    for i in np.where(err > 0.1)[0]:
        geo_str += '\'' + geo[i] + '\', '
    geo_str = geo_str[:-2] + ']'
    # print(geo_str)

    fig1, ax1 = plt.subplots(dpi=400, figsize=(15, 6))
    plt.cla()
    plt.yscale('log')
    ax1.bar(np.arange(len(err)), err, color=col)
    ax1.yaxis.grid(True)
    ax1.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=100, decimals=2))
    ax1.set_ylim(0.01, 100)
    plt.xticks(np.arange(len(err)), geo, rotation='vertical')
    plt.ylabel('Max. outlet pressure error 3D vs. 0D BC')
    fname = os.path.join(db.fpath_gen, 'bc_err.png')
    # plt.legend(list(colors.keys()))
    fig1.savefig(fname, bbox_inches='tight')


def main(db, geometries):
    for geo in geometries:
        check_bc(db, geo)


if __name__ == '__main__':
    descr = 'Check RCR boundary condition of 3d simulation'
    d, g, _ = input_args(descr)
    main(d, g)
    # plot(d, g)
