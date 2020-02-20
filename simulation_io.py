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

import numpy as np

from common import get_dict
from get_database import input_args
from vtk_functions import read_geo

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
        result_list_1d = glob.glob(os.path.join(res_dir, '*Group*Seg*_' + field + '.dat'))

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
            group = int(re.findall(r'\d+', f_res)[-2])
            results_1d[field][group] = np.array(results_1d_f)

    # read simulation parameters and add to result dict
    results_1d['params'] = get_dict(params_file)

    return results_1d


def load_results_3d(f_res_3d):
    res_3d = get_dict(f_res_3d)
    res_3d['flow'] = res_3d['velocity']
    del res_3d['velocity']
    return res_3d


class Geometry(object):
    def __init__(self, fpath):
        reader, pts, cls = read_geo(fpath)
        self.reader = reader

        self.p_arrays = {}
        for i in range(pts.GetNumberOfArrays()):
            self.p_arrays[pts.GetArrayName(i)] = v2n(pts.GetArray(i))

        self.c_arrays = {}
        for i in range(cls.GetNumberOfArrays()):
            self.c_arrays[cls.GetArrayName(i)] = v2n(cls.GetArray(i))

        self.groups = np.unique(self.c_arrays['GroupIds'])

    def cell(self, i):
        return self.reader.GetOutput().GetCell(i)

    def cell_points(self, i):
        cell = self.cell(i)
        return [cell.GetPointIds().GetId(i) for i in range(cell.GetNumberOfPoints())]

    def group_point_ids(self):
        point_ids = {}
        for g in self.groups:
            ids = []
            for c in self.group_cell_id(g):
                ids += self.cell_points(c)
            point_ids[g] = np.unique(ids)
        return point_ids

    def get_group_array(self, name):
        res = {}
        for g in self.groups:
            res[g] = self.p_arrays[name][self.group_point_ids()[g]]
        return res


class OneDModel(Geometry):
    def __init__(self, f_geo, f_caps):
        super().__init__(f_geo)
        
        self.caps = self.get_map(f_caps)

    def group_cell_id(self, g):
        # get all ids
        return np.where(self.c_arrays['GroupIds'] == g)[0]

    def group_seg_ids(self):
        seg_ids = {}
        for g in self.groups:
            seg_ids[g] = self.c_arrays['seg_id'][self.group_cell_id(g)]
        return seg_ids

    def get_seg_array(self, name):
        res = {}
        for s in self.c_arrays['seg_id']:
            i = np.where(self.c_arrays['seg_id'] == s)[0]
            res[s] = []
            for p in self.cell_points(i):
                res[s] += [self.p_arrays[name][p]]
        return res

    def get_map(self, fpath):
        # add inlet segment id
        caps = {'inflow': {}}
        caps['inflow']['cap'] = 0

        # add outlet segment id
        with open(fpath) as f:
            for line in csv.reader(f, delimiter=' '):
                caps[line[0]] = {}
                caps[line[0]]['cap'] = int(line[2])

        # add GroupIds
        for c, v in caps.items():
            caps[c]['GroupId'] = self.c_arrays['GroupIds'][self.c_arrays['seg_id'] == v['cap']][0]

        # add SegIds
        for c, v in caps.items():
            caps[c]['SegId'] = self.c_arrays['seg_id'][self.group_seg_ids()[caps[c]['GroupId']]]

        return caps
    

class Centerline(Geometry):
    def __init__(self, fpath):
        super().__init__(fpath)

    def group_cell_id(self, g):
        # get only first id
        return [np.where(self.c_arrays['GroupIds'] == g)[0][0]]

    def path(self):
        points = v2n(self.reader.GetOutput().GetPoints().GetData())
        paths = {}
        for g, ids in self.group_point_ids().items():
            paths[g] = np.cumsum(np.insert(np.linalg.norm(np.diff(points[ids], axis=0), axis=1), 0, 0))
        return paths

    def branch(self):
        branches = {}
        for g in self.groups:
            branches[g] = np.where(1 - self.p_arrays['CenterlineSectionBifurcation'][self.group_point_ids()[g]])[0]
        return branches


    def InteriorGroups(self):
        return self.c_arrays['GroupIds'][self.IntegrationCells()]

    def IntegrationCells(self):
        branch_cells = np.where(1 - self.c_arrays['Blanking'])[0]
        _, ids = np.unique(self.c_arrays['GroupIds'][branch_cells], return_index=True)
        return branch_cells[ids]

    def InletCells(self):
        integrate = self.IntegrationCells()
        return integrate[integrate == 0]

    def OutletCells(self):
        branch_cells = np.where(1 - self.c_arrays['Blanking'])[0]
        _, ids, counts = np.unique(self.c_arrays['GroupIds'][branch_cells], return_index=True, return_counts=True)
        return branch_cells[ids[counts == 1]]

    def BranchPoints(self):
        return np.where(1 - self.p_arrays['CenterlineSectionBifurcation'])


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
    time['1d_all'] = np.arange(0, res_1d['pressure'][0].shape[1] + 1)[1:] * dt

    # 3d-time moved to the last full 1d cycle (for interpolation)
    n_cycle_1d = max(1, int(time['1d_all'][-1] // res_3d['time'][-1]))
    time['3d_last_1d'] = res_3d['time'] + (n_cycle_1d - 1) * res_3d['time'][-1]

    return time


def check_consistency(model_1d, centerline, res_1d, res_3d):
    # if model_1d.reader.GetNumberOfCells() + 1 != model_1d.reader.GetNumberOfPoints():
    #     return '1d model connectivity inconsistent'

    if not np.array_equal(centerline.InteriorGroups(), model_1d.groups):
        return '1d model and 3d centerline inconsistent'

    if model_1d.c_arrays['seg_id'].shape[0] != len([*res_1d['flow']]):
        return '1d model and 1d results incosistent'

    if centerline.InteriorGroups().tolist() != [*res_3d['flow']]:
        return '3d centerline and 3d results inconsistent'

    return None


def collect_results(f_res_1d, f_res_3d, f_1d_model, f_centerline, f_groupid):
    if not os.path.exists(f_centerline):
        print('No centerline')
        return None, None
    if not os.path.exists(f_1d_model):
        print('No 1d model')
        return None, None
    if not os.path.exists(f_res_1d):
        print('No 1d results')
        return None, None
    if not os.path.exists(f_res_3d):
        print('No 3d results')
        return None, None
    if not os.path.exists(f_groupid):
        print('No outlet file')
        return None, None

    res_1d = get_dict(f_res_1d)
    res_3d = load_results_3d(f_res_3d)

    model_1d = OneDModel(f_1d_model, f_groupid)
    centerline = Centerline(f_centerline)

    err = check_consistency(model_1d, centerline, res_1d, res_3d)
    if err:
        print(err)
        return None

    # simulation time steps
    time = get_time(res_1d, res_3d)

    # 1d-coordinates for points along centerline
    path_1d = model_1d.get_seg_array('path')
    branch = centerline.branch()
    area = centerline.get_group_array('CenterlineSectionArea')

    res = {}
    for c, v in model_1d.caps.items():
        res[c] = {}
        for f in ['flow', 'pressure', 'area']:
            res[c][f] = {}
            res[c][f]['1d_int'] = []

            # 1d interior results (loop through group segments)
            res[c]['1d_path'] = []
            for i, s in enumerate(v['SegId']):
                # 1d results are duplicate at FE-nodes at corners of segments
                if i == 0:
                    i_start = 0
                else:
                    i_start = 1

                # generate path for segment FEs, assuming equidistant spacing
                p0 = path_1d[s][0]
                p1 = path_1d[s][1]
                res[c]['1d_path'] += np.linspace(p0, p1, res_1d[f][s].shape[0])[i_start:].tolist()
                res[c][f]['1d_int'] += res_1d[f][s][i_start:].tolist()

            res[c]['1d_path'] = np.array(res[c]['1d_path'])
            res[c][f]['1d_int'] = np.array(res[c][f]['1d_int'])

            if c == 'inflow':
                i_cap = 0
            else:
                i_cap = -1

            # 3d interior results
            g = v['GroupId']
            res[c]['3d_path'] = centerline.path()[g][branch[g]]
            res[c][f]['3d_int'] = res_3d[f][g].T[branch[g]]

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
    f_res_1d = db.get_1d_flow_path(geo)
    f_res_3d = db.get_3d_flow_path(geo)
    f_1d_model = db.get_1d_geo(geo)
    f_centerline = db.get_centerline_path(geo)
    f_groupid = db.get_groupid_path(geo)

    return collect_results(f_res_1d, f_res_3d, f_1d_model, f_centerline, f_groupid)


def main(db, geometries):
    for geo in geometries:
        res, time = collect_results_db(db, geo)


if __name__ == '__main__':
    descr = 'Retrieve simulation results'
    d, g, _ = input_args(descr)
    main(d, g)
