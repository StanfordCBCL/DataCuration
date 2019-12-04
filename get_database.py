#!/usr/bin/env python

import os
import shutil
import glob
import subprocess

import numpy as np

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

    def get_flow_path(self, geo):
        return os.path.join(self.fpath_gen, 'flow', geo + '.flow')

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
