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
from scipy.interpolate import interp1d

import numpy as np
from collections import defaultdict, OrderedDict

from common import get_dict
from get_database import input_args
from vtk_functions import read_geo, write_geo, collect_arrays, get_all_arrays, ClosestPoints
from get_bc_integrals import get_res_names
from vtk_to_xdmf import write_xdmf
from postproc import map_meshes

from vtk.util.numpy_support import numpy_to_vtk as n2v
from vtk.util.numpy_support import vtk_to_numpy as v2n

sys.path.append('/home/pfaller/work/repos/SimVascular/Python/site-packages/')


# import sv_1d_simulation as oned
# from mesh import get_connectivity


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
        results_1d[field] = defaultdict(dict)
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
            results_1d[field][branch][seg] = np.array(results_1d_f)

    # read simulation parameters and add to result dict
    results_1d['params'] = get_dict(params_file)

    return results_1d


def write_results_1d(f_res_1d, f_res_3d, f_geo_1d, f_cent, f_out, t_in):
    # read results
    res_1d = get_dict(f_res_1d)
    res_3d = load_results_3d(f_res_3d)

    # read geometry
    geo_cent = read_geo(f_cent).GetOutput()
    geo_1d = read_geo(f_geo_1d).GetOutput()

    # get time information
    time = {}
    get_time('3d', res_3d, time, t_in=t_in)
    get_time('1d', res_1d, time, t_in=t_in)

    # write results to centerline
    arrays = map_1d_to_centerline(geo_cent, geo_1d, res_1d, time)

    if '.xdmf' in f_out:
        write_xdmf(geo_cent, arrays, f_out)
    else:
        for f, a in arrays[str(time['1d'][-1])]['point'].items():
            out_array = n2v(a)
            out_array.SetName(f)
            geo_cent.GetPointData().AddArray(out_array)
        write_geo(f_out, geo_cent)


def map_1d_to_centerline(geo_cent, geo_1d, res_1d, time):
    # assemble output dict
    rec_dd = lambda: defaultdict(rec_dd)
    arrays = rec_dd()

    # extract point arrays from geometry
    arrays_cent, _ = get_all_arrays(geo_cent)
    arrays_1d, _ = get_all_arrays(geo_1d)

    # add centerline arrays
    for name, data in arrays_cent.items():
        arrays['always']['point'][name] = data

    # fields to export
    fields_res_1d = ['flow', 'pressure', 'area', 'wss', 'Re']

    # time steps to export
    i_export = np.where(time['1d_last_cycle_i'])[0]

    # number of time steps
    n_t = res_1d[fields_res_1d[0]][0][0].shape[1]

    # centerline points
    points = v2n(geo_cent.GetPoints().GetData())

    # loop all result fields
    for f in fields_res_1d:
        array_f = np.zeros((arrays_cent['Path'].shape[0], n_t))

        n_outlet = np.zeros(arrays_cent['Path'].shape[0])

        # loop all branches
        for br in res_1d[f].keys():
            # results of this branch
            res_br = res_1d[f][br]

            # get centerline path
            path_cent = arrays_cent['Path'][arrays_cent['BranchId'] == br]

            # get 1d path
            path_1d_geo = arrays_1d['Path'][arrays_1d['BranchId'] == br]

            # map results to branches
            path_1d_res, f_res = res_1d_to_path(path_1d_geo, res_br)

            # map 1d results to centerline using paths
            f_cent = interp1d(path_1d_res, f_res.T, fill_value='extrapolate')(path_cent).T

            # store results of this path
            array_f[arrays_cent['BranchId'] == br] = f_cent

            # add upstream part of branch within junction
            if br == 0:
                continue

            # first point of branch
            ip = np.where(arrays_cent['BranchId'] == br)[0][0]

            # centerline that passes through branch (first occurence)
            cid = np.where(arrays_cent['CenterlineId'][ip])[0][0]

            # id of upstream junction
            jc = arrays_cent['BifurcationId'][ip - 1]

            # centerline within junction
            jc_cent = np.where(np.logical_and(arrays_cent['BifurcationId'] == jc, arrays_cent['CenterlineId'][:, cid]))[
                0]

            # length of centerline within junction
            jc_path = np.append(0, np.cumsum(np.linalg.norm(np.diff(points[jc_cent], axis=0), axis=1)))
            jc_path /= jc_path[-1]

            # results at upstream branch
            res_br_u = res_1d[f][arrays_cent['BranchId'][jc_cent[0] - 1]]

            # results at beginning and end of centerline within junction
            f0 = res_br_u[sorted(res_br_u.keys())[-1]][-1]
            f1 = res_br[0][0]

            # map 1d results to centerline using paths
            array_f[jc_cent] += interp1d([0, 1], np.vstack((f0, f1)).T, fill_value='extrapolate')(jc_path).T

            # count number of outlets of this junction
            n_outlet[jc_cent] += 1

        # normalize by number of outlets
        array_f[n_outlet > 0] = (array_f[n_outlet > 0].T / n_outlet[n_outlet > 0]).T

        # assemble time steps
        for i, t in enumerate(i_export):
            arrays[str(time['1d'][i])]['point'][f] = array_f[:, t]

    return arrays


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


def get_time(model, res, time, dt_3d=0, t_in=0):
    if '3d_rerun' in model:
        time[model + '_all'] = res['time'] * dt_3d
    elif '3d' in model:
        time[model] = np.array([0] + res['time'].tolist())
        time[model + '_all'] = time[model]
    elif '1d' in model:
        dt = 1e-3
        time[model + '_all'] = np.arange(0, res['pressure'][0][0].shape[1] + 1)[1:] * dt

    # time steps for last cycle
    if '3d' in time and (model == '1d' or '3d_rerun' in model):
        # how many full cycles where completed?
        n_cycle = max(1, int(time[model + '_all'][-1] // time['3d'][-1]))
        time[model + '_n_cycle'] = n_cycle

        # first and last time step in cycle
        t_end = t_in
        t_first = t_end * (n_cycle - 1)
        t_last = t_end * n_cycle

        # tolerance (<< time step) to prevent errors due to time step round-off
        eps = 1.0e-12

        # select last cycle and shift time to start from zero
        time[model + '_last_cycle_i'] = np.logical_and(time[model + '_all'] >= t_first, time[model + '_all'] <= t_last)
        time[model] = time[model + '_all'][time[model + '_last_cycle_i']] - t_first
        for i in np.arange(1, n_cycle + 1):
            t_first = t_end * (i - 1)
            t_last = t_end * i
            bound0 = time[model + '_all'] > t_first + eps
            bound1 = time[model + '_all'] <= t_last + eps
            time[model + '_i_cycle_' + str(i)] = np.logical_and(bound0, bound1)
            time[model + '_cycle_' + str(i)] = time[model + '_all'][time[model + '_i_cycle_' + str(i)]] - t_first


def check_consistency(r_oned, res_1d, res_3d):
    n_br_res_1d = len(res_1d['area'].keys())
    n_br_res_3d = len(res_3d['area'].keys())
    n_br_geo_1d = np.unique(v2n(r_oned.GetOutput().GetPointData().GetArray('BranchId'))).shape[0]

    if n_br_res_1d != n_br_res_3d:
        return '1d and 3d results incosistent'

    if r_oned.GetNumberOfCells() + n_br_geo_1d != r_oned.GetNumberOfPoints():
        return '1d model connectivity inconsistent'

    return None


def get_branches(arrays):
    """
    Get list of branch IDs from point arrays
    """
    branches = np.unique(arrays['BranchId']).astype(int).tolist()
    if -1 in branches:
        branches.remove(-1)
    return branches


def get_caps_db(db, geo, f_surf=None):
    """
    Get caps for OSMSC models
    """
    return get_caps(db.get_centerline_outlet_path(geo), db.get_centerline_path(geo), f_surf)


def get_caps(f_outlet, f_centerline, f_surf=None):
    """
    Map outlet names to centerline branch id
    Args:
        f_outlet: ordered list of outlet names (created during centerline extraction)
        f_centerline: centerline geometry (.vtp)

    Returns:
        dictionary {cap name: BranchId}
    """
    caps = OrderedDict()
    caps['inflow'] = 0

    # read ordered outlet names from file
    outlet_names = []
    with open(f_outlet) as file:
        for line in file:
            outlet_names += line.splitlines()

    # read centerline
    cent = read_geo(f_centerline).GetOutput()
    branch_id = v2n(cent.GetPointData().GetArray('BranchId'))

    # find outlets and store outlet name and BranchId
    ids = vtk.vtkIdList()
    i_outlet = 0

    # closest surface points
    if f_surf:
        # transfer surface ids
        surf = read_geo(f_surf).GetOutput()
        cell_to_point = vtk.vtkCellDataToPointData()
        cell_to_point.SetInputData(surf)
        cell_to_point.Update()
        face_id = v2n(cell_to_point.GetOutput().GetPointData().GetArray('BC_FaceID'))
        cp = ClosestPoints(f_surf)
        br_to_bcface = OrderedDict()
        br_to_bcface[0] = face_id[cp.search([list(cent.GetPoint(0))])[0]]

    # loop all centerline points
    for i in range(1, cent.GetNumberOfPoints()):
        cent.GetPointCells(i, ids)

        # check if cap
        if ids.GetNumberOfIds() == 1:
            # this works since the points are numbered according to the order of outlets
            caps[outlet_names[i_outlet]] = branch_id[i]

            # find closest surface point
            if f_surf:
                i_point = cp.search([list(cent.GetPoint(i))])[0]
                br_to_bcface[branch_id[i]] = face_id[i_point]

            i_outlet += 1

    if f_surf:
        return caps, br_to_bcface
    else:
        return caps


def res_1d_to_path(path, res):
    path_1d = []
    int_1d = []
    for seg, res_1d_seg in sorted(res.items()):
        # 1d results are duplicate at FE-nodes at corners of segments
        if seg == 0:
            # start with first FE-node
            i_start = 0
        else:
            # skip first FE-node (equal to last FE-node of previous segment)
            i_start = 1

        # generate path for segment FEs, assuming equidistant spacing
        p0 = path[seg]
        p1 = path[seg + 1]
        path_1d += np.linspace(p0, p1, res_1d_seg.shape[0])[i_start:].tolist()
        int_1d += res_1d_seg[i_start:].tolist()

    return np.array(path_1d), np.array(int_1d)


def res_1d_to_jc():
    return


def collect_results(model, res, time, f_res, centerline=None, dt_3d=0, t_in=0, caps=None):
    # read results
    # todo: store 1d results in vtp as well
    if '1d' in model:
        res_in = get_dict(f_res)
        f_geo = centerline
    elif '3d_rerun_bc' in model:
        res_in = get_dict(f_res)
        f_geo = centerline
    elif '3d' in model:
        res_in = load_results_3d(f_res)
        f_geo = f_res
    else:
        raise ValueError('Model ' + model + ' not recognized')

    # read geometry
    geo = read_geo(f_geo)

    # extract point and cell arrays from geometry
    arrays, _ = get_all_arrays(geo.GetOutput())

    # get branches
    branches = get_branches(arrays)

    # simulation time steps
    get_time(model, res_in, time, dt_3d, t_in)

    # loop outlets
    for br in branches:
        # 1d-path along branch (real length units)
        branch_path = arrays['Path'][arrays['BranchId'] == br]

        # loop result fields
        for f in ['flow', 'pressure', 'area']:
            if '1d' in model:
                res[br]['1d_path'], res[br][f]['1d_int'] = res_1d_to_path(branch_path, res_in[f][br])
            elif 'bc' in model:
                if br not in caps.keys():
                    continue
                res_bc = res_in[f][:, caps[br] - 1]
                if br > 0 and f == 'flow':
                    res_bc *= -1
                res[br][f][model + '_cap'] = res_bc
            elif '3d' in model:
                res[br][model + '_path'] = branch_path
                res[br][f][model + '_int'] = res_in[f][br].T

            # copy last time step at t=0
            if model == '3d':
                res[br][f][model + '_int'] = np.tile(res[br][f][model + '_int'], (1, 2))[:,
                                             res[br][f][model + '_int'].shape[1] - 1:]

            if 'bc' not in model:
                if br == 0:
                    # inlet
                    i_cap = 0
                else:
                    # outlet
                    i_cap = -1

                # extract cap results
                res[br][f][model + '_cap'] = res[br][f][model + '_int'][i_cap, :]

    # get last cycle
    for br in res.keys():
        if 'bc' in model and br not in caps.keys():
            continue
        for f in res[br].keys():
            if 'path' not in f:
                res[br][f][model + '_all'] = res[br][f][model + '_cap']

                if model + '_last_cycle_i' in time:
                    if 'bc' not in model:
                        res[br][f][model + '_int'] = res[br][f][model + '_int'][:, time[model + '_last_cycle_i']]
                    res[br][f][model + '_cap'] = res[br][f][model + '_cap'][time[model + '_last_cycle_i']]


def collect_results_spatial(model, res, time, f_res, dt_3d=0, t_in=0):
    geo = read_geo(f_res).GetOutput()

    # fields to export
    fields = ['pressure', 'velocity']

    # get all result array names
    res_names = get_res_names(geo, fields)

    # extract all point arrays
    arrays, _ = get_all_arrays(geo)

    # sort results according to GlobalNodeID
    mask = map_meshes(arrays['GlobalNodeID'], np.arange(1, geo.GetNumberOfPoints() + 1))

    # get time steps
    times = np.unique([float(k.split('_')[1]) for k in res_names])

    # simulation time steps
    get_time(model, {'time': times}, time, dt_3d, t_in)

    # initialize results
    res[model]['pressure'] = np.zeros((times.shape[0], geo.GetNumberOfPoints()))
    res[model]['velocity'] = np.zeros((times.shape[0], geo.GetNumberOfPoints(), 3))

    # extract results
    for f in res_names:
        n, t = f.split('_')
        res[model][n][float(t) == times] = arrays[f][mask]

    # extract periodic cycle
    # if model + '_last_cycle_i' in time:
    #     for n in fields:
    #         res[model][n] = res[model][n][time[model + '_last_cycle_i']]


def collect_results_db_1d_3d(db, geo):
    # initialzie results dict
    res = defaultdict(lambda: defaultdict(dict))
    time = {}

    # get paths
    f_res_1d = db.get_1d_flow_path(geo)
    f_res_3d = db.get_3d_flow(geo)
    f_oned = db.get_1d_geo(geo)

    if not os.path.exists(f_res_1d) or not os.path.exists(f_res_3d):
        return None, None

    time_inflow, _ = db.get_inflow_smooth(geo)

    # collect results
    collect_results('3d', res, time, f_res_3d)
    collect_results('1d', res, time, f_res_1d, f_oned, t_in=time_inflow[-1])

    return res, time


def collect_results_db_3d_3d(db, geo, bc=False):
    # initialzie results dict
    res = defaultdict(lambda: defaultdict(dict))
    time = {}

    # get paths
    f_res_3d_osmsc = db.get_3d_flow(geo)
    if bc:
        f_res_3d_rerun = db.get_3d_flow_rerun_bc(geo)
    else:
        f_res_3d_rerun = db.get_3d_flow_rerun(geo)

    if not os.path.exists(f_res_3d_osmsc):
        return None, None

    time_inflow, _ = db.get_inflow_smooth(geo)

    if time_inflow is None:
        return None, None

    # collect osmsc results
    collect_results('3d', res, time, f_res_3d_osmsc)

    if not os.path.exists(f_res_3d_rerun):
        return res, time

    # collect rerun results
    if bc:
        f_cent = db.get_centerline_path(geo)
        f_surf = db.get_surfaces(geo, 'all_exterior')
        _, br_to_bcface = get_caps_db(db, geo, f_surf=f_surf)
        collect_results('3d_rerun_bc', res, time, f_res_3d_rerun, centerline=f_cent, dt_3d=db.get_3d_timestep(geo),
                        t_in=time_inflow[-1], caps=br_to_bcface)
    else:
        collect_results('3d_rerun', res, time, f_res_3d_rerun, dt_3d=db.get_3d_timestep(geo), t_in=time_inflow[-1])

    return res, time


def collect_results_db_3d_3d_spatial(db, geo):
    # initialzie results dict
    res = defaultdict(lambda: defaultdict(dict))
    time = {}

    # get paths
    f_res_3d_osmsc = db.get_volume(geo)
    f_res_3d_rerun = db.get_res_3d_vol_rerun(geo)

    if not os.path.exists(f_res_3d_osmsc) or not os.path.exists(f_res_3d_rerun):
        return None, None

    time_inflow, _ = db.get_inflow_smooth(geo)

    if time_inflow is None:
        return None, None

    # collect results
    collect_results_spatial('3d', res, time, f_res_3d_osmsc)
    collect_results_spatial('3d_rerun', res, time, f_res_3d_rerun, dt_3d=db.get_3d_timestep(geo), t_in=time_inflow[-1])

    return res, time


def export_1d_xmdf(db, geo):
    f_geo_1d = db.get_1d_geo(geo)
    f_res_1d = db.get_1d_flow_path(geo)
    f_res_3d = db.get_3d_flow(geo)
    f_geo = db.get_centerline_path(geo)
    # f_out = db.get_1d_flow_path_xdmf(geo)
    f_out = db.get_1d_flow_path_vtp(geo)

    time_inflow, _ = db.get_inflow_smooth(geo)

    write_results_1d(f_res_1d, f_res_3d, f_geo_1d, f_geo, f_out, t_in=time_inflow[-1])


def main(db, geometries):
    for geo in geometries:
        print('Processing ' + geo)

        if not os.path.exists(db.get_1d_flow_path(geo)):
            continue

        export_1d_xmdf(db, geo)


if __name__ == '__main__':
    descr = 'Retrieve simulation results'
    d, g, _ = input_args(descr)
    main(d, g)
