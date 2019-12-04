#!/usr/bin/env python

import os
import shutil
import glob
import subprocess
import csv

import numpy as np
from collections import OrderedDict

from get_bcs import get_bcs


class Database:
    def __init__(self):
        # folder for tcl files with boundary conditions
        self.fpath_bc = '/home/pfaller/work/osmsc/VMR_tcl_repository_scripts/repos_ready_cpm_scripts'

        # folder for simulation files
        self.fpath_sim = '/home/pfaller/work/osmsc/data_uploaded'

        # folder where generated data is saved
        self.fpath_gen = '/home/pfaller/work/osmsc/data_generated'

        # folder where simulation is run
        self.fpath_solve = '/home/pfaller/work/osmsc/simulation'

        # derived paths
        self.fpath_save = os.path.join(self.fpath_gen, 'database', 'database')

        # fields to extract
        self.res_fields = ['velocity', 'pressure']

        # initialize database containing all model information
        # todo: use advanced database that could be connected to webserver, like mysql
        self.database = {}

        # load from hdd
        self.load()

    def build_osmsc(self):
        # fields in database
        fields = ['image', 'model', 'results_rest', 'results_exercise',
                  'results_exercise_light', 'results_exercise_medium', 'results_exercise_heavy']

        # some geometries don't have results
        geo_no_results = ['0004', '0100', '0109', '0113', '0115', '0116', '0117', '0121', '0135', '0136', '0143',
                          '0159', '0168', '0169', '0170', '0171', '0177', '0178', '0179', '0180', '0181', '0182']
        geo_exercise_all = ['']
        geo_exercise_single = ['']

        # build defaults
        for geo in self.get_geometries():
            self.database[geo] = {}

            # default
            for f in fields:
                self.database[geo][f] = False

            # every geometry has
            self.database[geo]['image'] = True
            self.database[geo]['model'] = True

            # (almost) every geometry has
            if geo not in geo_no_results:
                self.database[geo]['results_rest'] = True

    def save(self):
        np.save(self.fpath_save, self.database)

    def load(self):
        self.database = np.load(self.fpath_save + '.npy', allow_pickle=True).item()
        return self.database

    def add(self):
        # todo: this is where new entries are added
        raise ValueError('not implemented')

    def has(self, geo, field):
        return self.database[geo][field]

    def get_geometries(self):
        geometries = os.listdir(self.fpath_sim)
        geometries.sort()
        return geometries

    def get_bcs(self, geo):
        tcl, tcl_bc = self.get_tcl_paths(geo)
        if os.path.exists(tcl) and os.path.exists(tcl_bc):
            return get_bcs(tcl, tcl_bc)
        else:
            return None, None

    def get_tcl_paths(self, geo):
        geo_bc = geo.split('_')[0] + '_' + str(int(geo.split('_')[1]) - 1).zfill(4)
        return os.path.join(self.fpath_bc, geo_bc + '.tcl'), os.path.join(self.fpath_bc, geo_bc + '-bc.tcl')

    def get_bc_flow_path(self, geo):
        return os.path.join(self.fpath_gen, 'bc_flow', geo + '.npy')

    def get_1d_flow_path(self, geo):
        return os.path.join(self.fpath_gen, '1d_flow', geo + '.npy')

    def get_flow_path(self, geo):
        return os.path.join(self.fpath_gen, 'flow', geo + '.flow')

    def get_groupid_path(self, geo):
        return os.path.join(self.get_solve_dir_1d(geo), 'outletface_groupid.dat')

    def get_flow(self, geo):
        return os.path.join(self.fpath_gen, 'flow', geo + '.flow')

    def get_solve_dir(self, geo):
        fsolve = os.path.join(self.fpath_solve, geo)
        os.makedirs(fsolve, exist_ok=True)
        return fsolve

    def get_solve_dir_1d(self, geo):
        fsolve = os.path.join(self.fpath_solve, geo, '1d')
        os.makedirs(fsolve, exist_ok=True)
        return fsolve

    def get_surface_dir(self, geo):
        return os.path.join(self.fpath_gen, 'surfaces', geo)

    def get_surfaces_upload(self, geo):
        surfaces = glob.glob(os.path.join(self.fpath_sim, geo, 'extras', 'mesh-surfaces', '*.vtp'))
        surfaces.append(os.path.join(self.fpath_sim, geo, 'extras', 'mesh-surfaces', 'extras', 'all_exterior.vtp'))
        return surfaces

    def add_cap_ordered(self, caps, keys_ordered, keys_left, c):
        keys = sorted(caps.keys())
        for k in keys:
            if c in k.lower() and k in keys_left:
                keys_ordered.append(k)
                keys_left.remove(k)

    def get_3d_1d_map(self, geo):
        # add inlet GroupId
        caps = {'inflow': {}}
        caps['inflow']['GroupId'] = 0

        # add outlet GroupIds
        with open(self.get_groupid_path(geo)) as f:
            for line in csv.reader(f, delimiter=' '):
                caps[line[0]] = {}
                caps[line[0]]['GroupId'] = int(line[2])

        # add inlet/oulet BC_FaceIDs
        bcs, _ = self.get_bcs(geo)
        for k in caps.keys():
            caps[k]['BC_FaceID'] = int(bcs['preid'][k])

        # nicely ordered cap names for output
        keys_left = sorted(caps.keys())
        keys_ordered = []
        self.add_cap_ordered(caps, keys_ordered, keys_left, 'inflow')
        self.add_cap_ordered(caps, keys_ordered, keys_left, 'aorta')
        self.add_cap_ordered(caps, keys_ordered, keys_left, 'p_')
        self.add_cap_ordered(caps, keys_ordered, keys_left, 'd_')
        self.add_cap_ordered(caps, keys_ordered, keys_left, 'left')
        self.add_cap_ordered(caps, keys_ordered, keys_left, 'right')
        self.add_cap_ordered(caps, keys_ordered, keys_left, 'l_')
        self.add_cap_ordered(caps, keys_ordered, keys_left, 'r_')
        keys_ordered += keys_left

        caps_ordered = OrderedDict()
        for k in keys_ordered:
            caps_ordered[k] = caps[k]

        return caps_ordered

    def get_surfaces(self, geo, surf='all'):
        fdir = self.get_surface_dir(geo)
        surfaces_all = glob.glob(os.path.join(fdir, '*.vtp'))
        if surf == 'all':
            surfaces = surfaces_all
        elif surf == 'inflow':
            surfaces = os.path.join(fdir, 'inflow.vtp')
        elif surf == 'all_exterior':
            surfaces = os.path.join(fdir, 'all_exterior.vtp')
        elif surf == 'outlets' or surf == 'caps':
            if surf == 'outlets':
                exclude = ['all_exterior', 'wall', 'inflow']
            elif surf == 'caps':
                exclude = ['all_exterior', 'wall']
            surfaces = [x for x in surfaces_all if not any(e in x for e in exclude)]
        else:
            print('Unknown surface option ' + surf)
            surfaces = []
        return surfaces

    def get_volume(self, geo):
        return os.path.join(self.fpath_sim, geo, 'results', geo + '_sim_results_in_cm.vtu')

    def get_outlet_names(self, geo):
        outlets = self.get_surfaces(geo, 'outlets')
        outlets = [os.path.splitext(os.path.basename(s))[0] for s in outlets]
        outlets.sort()
        return outlets

    def copy_files(self, geo):
        # define paths
        fpath_surf = os.path.join(self.fpath_solve, 'mesh-complete', 'mesh-surfaces')

        # create simulation folder
        os.makedirs(fpath_surf)

        # copy generic solver settings
        shutil.copy('solver.inp', self.fpath_solve)

        # copy geometry
        for f in glob.glob(os.path.join(self.fpath_gen, 'surfaces', geo, '*.vtp')):
            shutil.copy(f, fpath_surf)

        # copy volume mesh
        # todo: copy without results
        fpath_res = os.path.join(self.fpath_sim, geo, 'results', geo + '_sim_results_in_cm.vtu')
        shutil.copy(fpath_res, os.path.join(self.fpath_solve, 'mesh-complete'))

    def read_results(self, geo, fpath):
        if os.path.exists(fpath):
            res = np.load(fpath, allow_pickle=True).item()
        else:
            print('no results')
            return None

        if 'pressure' not in res or len(res['pressure']) == 0:
            print('results empty')
            return None

        return res

    def get_results_xd(self, geo):
        # get post-processing object
        post = Post()

        # read results
        res_1d = self.read_results(geo, self.get_1d_flow_path(geo))
        if res_1d is None:
            return None, None

        res_3d = self.read_results(geo, self.get_bc_flow_path(geo))
        if res_3d is None:
            return None, None

        # get map between 3d BC_FaceID and 1d GroupId for all inlet/outlets
        caps = self.get_3d_1d_map(geo)

        # rename 3d results
        res_3d['flow'] = res_3d['velocity']
        del res_3d['velocity']

        # collect results
        res = {}
        n_t = []
        for f in post.fields:
            res[f] = {}
            for k, v in caps.items():
                # get first or last element in 1d segment
                if k == 'inflow':
                    i_1d = 0
                else:
                    i_1d = -1

                res[f][k] = {}
                res[f][k]['1d'] = res_1d[f][v['GroupId']][i_1d]
                res[f][k]['3d'] = res_3d[f][:, v['BC_FaceID'] - 1]
                n_t.append(res[f][k]['1d'].shape[0])

        # check if all 1d solutions have the same number of time steps
        n_t = np.unique(np.array(n_t))
        if n_t.shape[0] > 1:
            print('conflicting number of time steps: ' + repr(n_t))
            return None, None

        # time
        time = {'3d': res_3d['time']}

        n_cycle = 10
        dt = 1e-3
        step_cycle = int(time['3d'][-1] / dt)
        tmax = step_cycle * dt
        time['1d'] = np.arange(0, tmax, dt)
        time['step_cycle'] = step_cycle
        time['n_max'] = step_cycle * n_cycle
        time['n_cycle'] = n_cycle

        return res, time


class SimVascular:
    """
    simvascular object to handle external calls
    """
    def __init__(self):
        self.svpre = '/usr/local/sv/svsolver/2019-02-07/svpre'
        self.svsolver = '/usr/local/sv/svsolver/2019-02-07/svsolver'
        self.onedsolver = '/home/pfaller/work/repos/oneDSolver/build/bin/OneDSolver'

    def run_pre(self, pre_folder, pre_file):
        subprocess.run([self.svpre, pre_file], cwd=pre_folder)

    def run_solver(self, run_folder, run_file='solver.inp'):
        subprocess.run([self.svsolver, run_file], cwd=run_folder)

    def run_solver_1d(self, run_folder, run_file='solver.inp'):
        subprocess.run([self.onedsolver, run_file], cwd=run_folder)


class Post:
    def __init__(self):
        db = Database()
        self.fields = ['pressure', 'flow']
        self.units = {'pressure': 'mmHg', 'flow': 'l/h'}
        self.models = ['1d', '3d']
        self.styles = {'1d': '-', '3d': '--'}

        cgs2mmhg = 7.50062e-4
        mlps2lph = 60 / 1000
        self.convert = {'pressure': cgs2mmhg, 'flow': mlps2lph}