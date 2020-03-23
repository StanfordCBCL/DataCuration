#!/usr/bin/env python
# coding=utf-8

import argparse
import glob
import os
import csv
import re
import sys
import scipy
import pdb
import vtk

import numpy as np

from common import get_dict
from get_database import input_args
from vtk_functions import read_geo, collect_arrays, get_all_arrays
from get_bc_integrals import get_res_names

from vtk.util.numpy_support import numpy_to_vtk as n2v
from vtk.util.numpy_support import vtk_to_numpy as v2n

sys.path.append('/home/pfaller/work/repos/SimVascular/Python/site-packages/')

import sv_1d_simulation as oned


def read_results_1d(res_dir, params_file=None):
    """
    Read results from oneDSolver and store in dictionary
    Args:
        res_dir: directory containing 1D results
        params_file: optional, path to dictionary of oneDSolver input parameters

    Returns:
    Dictionary sorted as [result field][segment id][time step]
    """
    # requested output fields
    fields_res_1d = ['flow', 'pressure', 'area', 'wss', 'Re']

    # read 1D simulation results
    results_1d = {}
    for field in fields_res_1d:
        # list all output files for field
        result_list_1d = glob.glob(os.path.join(res_dir, '*branch*seg*_' + field + '.dat'))

        # loop segments
        results_1d[field] = {}
        for f_res in result_list_1d:
            with open(f_res) as f:
                reader = csv.reader(f, delimiter=' ')

                # loop nodes
                results_1d_f = []
                for line in reader:
                    results_1d_f.append([float(l) for l in line if l][1:])

            # store results and GroupId
            seg = int(re.findall(r'\d+', f_res)[-1])
            branch = int(re.findall(r'\d+', f_res)[-2])
            if branch not in results_1d[field]:
                results_1d[field][branch] = {}
            results_1d[field][branch][seg] = np.array(results_1d_f)

    # read simulation parameters and add to result dict
    results_1d['params'] = get_dict(params_file)

    return results_1d


def load_results_3d(f_res_3d):
    """
    Read 3d results embedded in centerline and sort according to branch at time step
    """
    # read 1d geometry
    reader = read_geo(f_res_3d)
    res = collect_arrays(reader.GetOutput().GetPointData())

    # names of output arrays
    res_names = get_res_names(reader, ['pressure', 'velocity'])

    # get time steps
    times = np.unique([float(k.split('_')[1]) for k in res_names])

    # get branch ids
    branches = np.unique(res['BranchId']).tolist()
    branches.remove(-1)

    # add time
    out = {'time': times}

    # initilize output arrays [time step, branch]
    for f in res_names:
        name = f.split('_')[0]
        out[name] = {}
        for br in branches:
            ids = res['BranchId'] == br
            out[name][br] = np.zeros((times.shape[0], np.sum(ids)))

    # read branch-wise results from geometry
    for f in res_names:
        name, time = f.split('_')
        for br in branches:
            ids = res['BranchId'] == br
            out[name][br][float(time) == times] = res[f][ids]

    # add area (identical for all time steps)
    out['area'] = {}
    for br in branches:
        ids = res['BranchId'] == br
        out['area'][br] = np.tile(res['area'][ids], (times.shape[0], 1))

    # rename velocity to flow
    out['flow'] = out['velocity']
    del out['velocity']

    return out


def get_time(res_1d, res_3d):
    # time steps
    time = {'3d': res_3d['time'], '3d_all': res_3d['time']}

    n_cycle = 10
    dt = 1e-3
    step_cycle = int(time['3d'][-1] // dt)
    tmax = step_cycle * dt
    time['1d'] = np.arange(0, tmax, dt)
    time['step_cycle'] = step_cycle
    time['n_cycle'] = n_cycle

    # total simulation times
    # time['time_0d_all'] = res_0d['time']
    # time['1d_all'] = np.arange(0, int(time['3d'][-1] // dt * n_cycle) * dt, dt)
    time['1d_all'] = np.arange(0, res_1d['pressure'][0][0].shape[1] + 1)[1:] * dt

    # 3d-time moved to the last full 1d cycle (for interpolation)
    n_cycle_1d = max(1, int(time['1d_all'][-1] // res_3d['time'][-1]))
    time['3d_last_1d'] = res_3d['time'] + (n_cycle_1d - 1) * res_3d['time'][-1]

    return time


def check_consistency(r_oned, res_1d, res_3d):
    n_br_res_1d = len(res_1d['area'].keys())
    n_br_res_3d = len(res_3d['area'].keys())
    n_br_geo_1d = np.unique(v2n(r_oned.GetOutput().GetPointData().GetArray('BranchId'))).shape[0]

    if n_br_res_1d != n_br_res_3d:
        return '1d and 3d results incosistent'

    if r_oned.GetNumberOfCells() + n_br_geo_1d != r_oned.GetNumberOfPoints():
        return '1d model connectivity inconsistent'

    return None


def collect_results(f_res_1d, f_res_3d, f_1d_model, outlets):
    # todo: merge results into 1d model
    if not os.path.exists(f_1d_model):
        print('No 1d model')
        return None, None
    if not os.path.exists(f_res_1d):
        print('No 1d results')
        return None, None
    if not os.path.exists(f_res_3d):
        print('No 3d results')
        return None, None

    # read results
    res_1d = get_dict(f_res_1d)
    res_3d = load_results_3d(f_res_3d)

    # read geometries
    r_cent = read_geo(f_res_3d)
    r_oned = read_geo(f_1d_model)

    # extract point and cell arrays from geometries
    p_oned, c_oned = get_all_arrays(r_oned)
    p_cent, c_cent = get_all_arrays(r_cent)

    # check inpud data for consistency
    err = check_consistency(r_oned, res_1d, res_3d)
    if err:
        print(err)
        return None, None

    # simulation time steps
    time = get_time(res_1d, res_3d)

    # loop outlets
    res = {}
    # for c, br in caps.items():
    for br, c in enumerate(['inflow'] + outlets):
        res[c] = {}

        # 1d-path along branch (real length units)
        path_1d = p_oned['Path'][p_oned['BranchId'] == br]
        path_3d = p_cent['Path'][p_cent['BranchId'] == br]

        # loop result fields
        for f in ['flow', 'pressure', 'area']:
            res[c][f] = {}
            res[c][f]['1d_int'] = []

            # 1d interior results (loop through branch segments)
            res[c]['1d_path'] = []

            assert path_1d.shape[0] == len(res_1d[f][br]) + 1, '1d model and 1d model do not match'

            for seg, res_1d_seg in sorted(res_1d[f][br].items()):
                # 1d results are duplicate at FE-nodes at corners of segments
                if seg == 0:
                    # start with first FE-node
                    i_start = 0
                else:
                    # skip first FE-node (equal to last FE-node of previous segment)
                    i_start = 1

                # generate path for segment FEs, assuming equidistant spacing
                p0 = path_1d[seg]
                p1 = path_1d[seg + 1]
                res[c]['1d_path'] += np.linspace(p0, p1, res_1d_seg.shape[0])[i_start:].tolist()
                res[c][f]['1d_int'] += res_1d_seg[i_start:].tolist()

            res[c]['1d_path'] = np.array(res[c]['1d_path'])
            res[c][f]['1d_int'] = np.array(res[c][f]['1d_int'])

            if c == 'inflow':
                i_cap = 0
            else:
                i_cap = -1

            # 3d interior results
            res[c]['3d_path'] = path_3d
            res[c][f]['3d_int'] = res_3d[f][br].T

            # 1d and 3d cap results
            for m in ['1d', '3d']:
                res[c][f][m + '_cap'] = res[c][f][m + '_int'][i_cap, :]

    # interpolate results to 3d time
    for c in res.keys():
        for f in res[c].keys():
            if 'path' not in f:
                res[c][f]['1d_all'] = res[c][f]['1d_cap']
                res[c][f]['3d_all'] = res[c][f]['3d_cap']
                for m in ['1d_int', '1d_cap']:
                    interp = scipy.interpolate.interp1d(time['1d_all'], res[c][f][m], bounds_error=False)
                    res[c][f][m] = interp(time['3d_last_1d'])

    return res, time


def collect_results_db(db, geo):
    # get paths
    f_res_1d = db.get_1d_flow_path(geo)
    f_res_3d = db.get_3d_flow_path_oned_vtp(geo)
    f_1d_model = db.get_1d_geo(geo)

    # get outlets
    outlets = db.get_outlet_names(geo)

    return collect_results(f_res_1d, f_res_3d, f_1d_model, outlets)


def main(db, geometries):
    for geo in geometries:
        res, time = collect_results_db(db, geo)


if __name__ == '__main__':
    descr = 'Retrieve simulation results'
    d, g, _ = input_args(descr)
    main(d, g)
