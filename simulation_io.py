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
from _collections import defaultdict

from common import get_dict
from get_database import input_args
from vtk_functions import read_geo, collect_arrays, get_all_arrays
from get_bc_integrals import get_res_names

from vtk.util.numpy_support import numpy_to_vtk as n2v
from vtk.util.numpy_support import vtk_to_numpy as v2n

sys.path.append('/home/pfaller/work/repos/SimVascular/Python/site-packages/')

# import sv_1d_simulation as oned


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
    reader = read_geo(f_res_3d).GetOutput()
    res = collect_arrays(reader.GetPointData())

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


def get_time(model, res, time):
    if model == '3d_rerun':
        dt = 4.0e-4
        time[model + '_all'] = res['time'] * dt
    elif '3d' in model:
        time[model] = np.array([0] + res['time'].tolist())
        time[model + '_all'] = time[model]
    elif '1d' in model:
        dt = 1e-3
        time[model + '_all'] = np.arange(0, res['pressure'][0][0].shape[1] + 1)[1:] * dt

    # time steps for last cycle
    if model == '1d' or model == '3d_rerun':
        n_cycle_1d = max(1, int(time[model + '_all'][-1] // time['3d'][-1]))
        t_1d_first = time['3d'][-1] * (n_cycle_1d - 1)
        time[model + '_last_cycle_i'] = time[model + '_all'] >= t_1d_first
        time[model] = time[model + '_all'][time[model + '_last_cycle_i']] - t_1d_first


def check_consistency(r_oned, res_1d, res_3d):
    n_br_res_1d = len(res_1d['area'].keys())
    n_br_res_3d = len(res_3d['area'].keys())
    n_br_geo_1d = np.unique(v2n(r_oned.GetOutput().GetPointData().GetArray('BranchId'))).shape[0]

    if n_br_res_1d != n_br_res_3d:
        return '1d and 3d results incosistent'

    if r_oned.GetNumberOfCells() + n_br_geo_1d != r_oned.GetNumberOfPoints():
        return '1d model connectivity inconsistent'

    return None


def get_caps(f_outlet, f_centerline):
    """
    Create dictionary: cap name -> BranchId
    """
    caps = {'inflow': 0}

    # read ordered outlet names from file
    outlet_names = []
    with open(f_outlet) as file:
        for line in file:
            outlet_names += line.splitlines()

    # read centerline
    cent = read_geo(f_centerline).GetOutput()
    branch_id = v2n(cent.GetPointData().GetArray('BranchId'))

    # find outlets and store name and branch id
    ids = vtk.vtkIdList()
    i_outlet = 0
    for i in range(1, cent.GetNumberOfPoints()):
        cent.GetPointCells(i, ids)
        if ids.GetNumberOfIds() == 1:
            caps[outlet_names[i_outlet]] = branch_id[i]
            i_outlet += 1

    return caps


def collect_results(model, res, time, f_res, f_outlet, centerline=None, res3d=None):
    # read results
    # todo: store 1d results in vtp as well
    if '1d' in model:
        res_in = get_dict(f_res)
        f_geo = centerline
        f_cap = res3d
    elif '3d' in model:
        res_in = load_results_3d(f_res)
        f_geo = f_res
        f_cap = f_res
    else:
        raise ValueError('Model ' + model + ' not recognized')

    # read geometry
    geo = read_geo(f_geo)

    # extract point and cell arrays from geometry
    arrays, _ = get_all_arrays(geo)

    # get cap->branch map
    caps = get_caps(f_outlet, f_cap)

    # simulation time steps
    get_time(model, res_in, time)

    # loop outlets
    for c, br in caps.items():
        # 1d-path along branch (real length units)
        branch_path = arrays['Path'][arrays['BranchId'] == br]

        # loop result fields
        for f in ['flow', 'pressure', 'area']:
            if '1d' in model:
                # loop 1d segments
                path_1d = []
                int_1d = []
                for seg, res_1d_seg in sorted(res_in[f][br].items()):
                    # 1d results are duplicate at FE-nodes at corners of segments
                    if seg == 0:
                        # start with first FE-node
                        i_start = 0
                    else:
                        # skip first FE-node (equal to last FE-node of previous segment)
                        i_start = 1

                    # generate path for segment FEs, assuming equidistant spacing
                    p0 = branch_path[seg]
                    p1 = branch_path[seg + 1]
                    path_1d += np.linspace(p0, p1, res_1d_seg.shape[0])[i_start:].tolist()
                    int_1d += res_1d_seg[i_start:].tolist()

                res[c]['1d_path'] = np.array(path_1d)
                res[c][f]['1d_int'] = np.array(int_1d)

            elif '3d' in model:
                res[c][model + '_path'] = branch_path
                res[c][f][model + '_int'] = res_in[f][br].T

            # copy last time step at t=0
            if model == '3d':
                res[c][f][model + '_int'] = np.tile(res[c][f][model + '_int'], (1, 2))[:, res[c][f][model + '_int'].shape[1] - 1:]

            # cap results
            if c == 'inflow':
                i_cap = 0
            else:
                i_cap = -1
            res[c][f][model + '_cap'] = res[c][f][model + '_int'][i_cap, :]

    # get last cycle
    for c in res.keys():
        for f in res[c].keys():
            if 'path' not in f:
                res[c][f][model + '_all'] = res[c][f][model + '_cap']

                if model + '_last_cycle_i' in time:
                    res[c][f][model + '_int'] = res[c][f][model + '_int'][:, time[model + '_last_cycle_i']]
                    res[c][f][model + '_cap'] = res[c][f][model + '_cap'][time[model + '_last_cycle_i']]


def collect_results_db_1d_3d(db, geo):
    # initialzie results dict
    res = defaultdict(lambda: defaultdict(dict))
    time = {}

    # get paths
    f_res_1d = db.get_1d_flow_path(geo)
    f_res_3d = db.get_3d_flow(geo)
    f_outlet = db.get_centerline_outlet_path(geo)
    f_oned = db.get_1d_geo(geo)

    # collect results
    collect_results('3d', res, time, f_res_3d, f_outlet)
    collect_results('1d', res, time, f_res_1d, f_outlet, f_oned, f_res_3d)

    return res, time


def collect_results_db_3d_3d(db, geo):
    # initialzie results dict
    res = defaultdict(lambda: defaultdict(dict))
    time = {}

    # get paths
    f_res_3d_osmsc = db.get_3d_flow(geo)
    f_res_3d_rerun = db.get_3d_flow_rerun(geo)
    f_outlet = db.get_centerline_outlet_path(geo)

    # collect results
    collect_results('3d', res, time, f_res_3d_osmsc, f_outlet)
    collect_results('3d_rerun', res, time, f_res_3d_rerun, f_outlet)

    return res, time


def main(db, geometries):
    for geo in geometries:
        res, time = collect_results_db(db, geo)


if __name__ == '__main__':
    descr = 'Retrieve simulation results'
    d, g, _ = input_args(descr)
    main(d, g)
