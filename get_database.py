#!/usr/bin/env python

import os
import sys
import shutil
import glob
import subprocess
import csv
import re
import argparse
import pdb

import numpy as np
from collections import OrderedDict

from vtk.util.numpy_support import numpy_to_vtk as n2v
from vtk.util.numpy_support import vtk_to_numpy as v2n

from get_bcs import get_bcs, get_params, get_in_model_units
from vtk_functions import read_geo
from common import get_dict


def input_args(description):
    """
    Handles input arguments to scripts
    Args:
        description: script description (hgelp string)

    Returns:
        database: Database object for study
        geometries: list of geometries to evaluate
    """
    # parse input arguments
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('study', help='study name')
    parser.add_argument('-g', '--geo', help='individual geometry or subset name')
    param = parser.parse_args()

    # get model database
    database = Database(param.study)

    # choose geometries to evaluate
    if param.geo in database.get_geometries():
        geometries = [param.geo]
    elif param.geo is None:
        geometries = database.get_geometries()
    elif param.geo == '-1':
        geometries = reversed(database.get_geometries())
    elif param.geo[-1] == ':':
        geo_all = database.get_geometries()
        geo_first = geo_all.index(param.geo[:-1])
        geometries = geo_all[geo_first:]
    elif param.geo[-3:] == ':-1':
        geo_all = database.get_geometries()
        geo_all.reverse()
        geo_first = geo_all.index(param.geo[:-3])
        geometries = geo_all[geo_first:]
    else:
        geometries = database.get_geometries_select(param.geo)

    return database, geometries, param


class Database:
    def __init__(self, study=''):
        # study name, if any
        self.study = study

        # folder for tcl files with boundary conditions
        self.fpath_bc = '/home/pfaller/work/osmsc/VMR_tcl_repository_scripts/repos_ready_cpm_scripts'

        # self.fpath_need_sim = '/home/pfaller/work/osmsc/VMR_tcl_repository_scripts/need_sim_cpm_scripts'
        self.fpath_need_sim = '/home/pfaller/work/osmsc/VMR_tcl_repository_scripts/released_cpm_scripts'

        # folder for simulation files
        self.fpath_sim = '/home/pfaller/work/osmsc/data_uploaded'

        # folder where generated data is saved
        self.fpath_gen = '/home/pfaller/work/osmsc/data_generated'

        # folder for paths and segmentations
        self.fpath_seg_path = '/home/pfaller/work/osmsc/data_additional/models/'

        # folder for simulation studies
        self.fpath_studies = '/home/pfaller/work/osmsc/studies'

        # folder containing model images
        self.fpath_png = '/home/pfaller/work/osmsc/data_png'

        # folder for simulation studies
        self.fpath_study = os.path.join(self.fpath_studies, self.study)

        # folder where simulation is run
        self.fpath_solve = os.path.join(self.fpath_study, 'simulation')

        # derived paths
        self.fpath_save = os.path.join(self.fpath_gen, 'database', 'database')

        # fields to extract
        self.res_fields = ['velocity', 'pressure']

        # svproject object
        self.svproj = SVProject()

    def is_excluded(self, geo):
        # excluded geometries by Nathan
        imaging = ['0001', '0020', '0044']
        animals = ['0066', '0067', '0068', '0069', '0070', '0071', '0072', '0073', '0074']
        single_vessel = ['0158', '0164', '0165']

        exclude_nathan = imaging + animals + single_vessel

        # # excluded models by martin (say rest but are exercise)
        # exclude_martin = ['0063_2001', '0064_2001', '0065_2001', '0075_2001', '0076_2001', '0080_1001', '0081_1001',
        #                   '0082_1001', '0083_2002', '0084_1001', '0086_1001', '0107_1001', '0111_1001']

        if geo[:4] in exclude_nathan:  # or geo in exclude_martin:
            return True
        else:
            return False

    def exclude_geometries(self, geometries):
        return [g for g in geometries if not self.is_excluded(g)]

    def get_geometries(self):
        geometries = os.listdir(self.fpath_sim)
        geometries.sort()
        return geometries

    def get_geometries_select(self, name):
        if name == 'paper':
            # pick all geometries in rest state
            geometries = []
            for geo in self.get_geometries():
                params = self.get_params(geo)
                if params['sim_physio_state'] == 'rest':
                    geometries += [geo]

            # exclude geometries
            geometries = self.exclude_geometries(geometries)
        elif name == 'fix_surf_id':
            geometries = ['0140_2001', '0144_1001', '0147_1001', '0160_6001', '0161_0001', '0162_3001', '0163_0001']
        elif name == 'fix_surf_discr':
            geometries = ['0069_0001', '0164_0001']
        elif name == 'fix_surf_displacement':
            geometries = ['0065_0001', '0065_1001', '0065_2001', '0065_3001', '0065_4001', '0078_0001', '0079_0001',
                          '0091_0001', '0091_2001', '0092_0001', '0108_0001', '0154_0001', '0154_1001', '0165_0001',
                          '0166_0001', '0183_1002', '0187_0002']
        elif name == 'fix_inlet_node':
            geometries = ['0080_0001', '0082_0001', '0083_2002', '0084_1001', '0088_1001', '0112_1001', '0134_0002']
        elif name == 'fix_surf_orientation':
            geometries = ['0069_0001']
        elif name == 'bifurcation_outlet':
            geometries = ['0080_0001', '0082_0001', '0083_2002', '0084_1001', '0088_1001', '0112_1001', '0134_0002']
        elif name == 'bifurcation_inlet':
            geometries = ['0065_1001', '0076_1001', '0081_0001', '0081_1001', '0086_0001', '0086_1001', '0089_1001',
                          '0148_1001', '0155_0001', '0162_3001']
        elif name == 'resistance':
            geometries = []
            for geo in self.get_geometries():
                name, err = self.get_bc_type(geo)
                if name == 'resistance':
                    geometries += [geo]
        elif 'units' in name:
            geometries = []
            for geo in self.get_geometries():
                _, params = self.get_bcs(geo)
                if params is None:
                    continue
                _, part, unit = name.split('_')
                if part == 's' and params['sim_units'] == unit:
                    geometries += [geo]
                elif part == 'm' and params['model_units'] == unit:
                    geometries += [geo]
            print(geometries)
            for geo in geometries:
                _, params = self.get_bcs(geo)

                print(get_in_model_units(params['sim_units'], 'viscosity', float(params['sim_viscosity'])))
            sys.exit(1)
        elif name == 'coronary':
            geometries = []
            for geo in self.get_geometries():
                bc_type, err = self.get_bc_type(geo)
                if bc_type is not None and 'coronary' in bc_type.values():
                    geometries += [geo]
        elif name in ['aorta', 'aortofemoral', 'pulmonary', 'cerebrovascular', 'coronary']:
            geometries = []
            for geo in self.get_geometries():
                _, params = self.get_bcs(geo)
                if params is not None and params['deliverable_category'].lower() == name:
                    geometries += [geo]

        else:
            raise Exception('Unknown selection ' + name)
        return geometries

    def get_params(self, geo):
        # try to find params in these folders
        paths = [self.fpath_bc, self.fpath_need_sim]
        for p in paths:
            for o in [-1, 0]:
                tcl, _ = get_tcl_paths(p, geo, o)
                if os.path.exists(tcl):
                    return get_params(tcl)

        return None

    def get_bcs(self, geo):
        # try two different offsets of tcl name vs geo name
        # todo: find out why there are several variants
        for o in [-1, 0]:
            tcl, tcl_bc = get_tcl_paths(self.fpath_bc, geo, o)
            if os.path.exists(tcl) and os.path.exists(tcl_bc):
                return get_bcs(tcl, tcl_bc)
        return None, None

    def get_bc_type(self, geo):
        bc_def, _ = self.get_bcs(geo)

        if bc_def is None:
            return None, 'boundary conditions not found'

        outlets = self.get_outlet_names(geo)

        bc_type = {}
        for s in outlets:
            if s in bc_def['bc']:
                bc = bc_def['bc'][s]
            else:
                return None, 'boundary conditions do not exist for surface ' + s

            if 'Rp' in bc and 'C' in bc and 'Rd' in bc:
                bc_type[s] = 'rcr'
            elif 'R' in bc and 'Po' in bc:
                bc_type[s] = 'resistance'
            elif 'COR' in bc:
                bc_type[s] = 'coronary'
            else:
                pdb.set_trace()
                return None, 'boundary conditions not implemented'

        return bc_type, False

    def has_loop(self, geo):
        # todo: find automatic way to check for loop
        loop = ['0001_0001', '0106_0001', '0188_0001']
        return geo in loop

    def get_png(self, geo):
        return os.path.join(self.fpath_png, 'OSMSC' + geo + '_sim.png')

    def get_img(self, geo):
        return exists(os.path.join(self.fpath_sim, geo, 'image_data', 'vti', 'OSMSC' + geo[:4] + '-cm.vti'))

    def get_surface_dir(self, geo):
        return os.path.join(self.fpath_gen, 'surfaces', geo)

    def get_sv_meshes(self, geo):
        fdir = os.path.join(self.fpath_gen, 'sv_meshes', geo)
        fdir_caps = os.path.join(fdir, 'caps')
        os.makedirs(fdir, exist_ok=True)
        os.makedirs(fdir_caps, exist_ok=True)
        return fdir

    def get_sv_surface(self, geo):
        return exists(os.path.join(self.get_sv_meshes(geo), geo + '.vtp'))

    def get_sv_surface_path(self, geo):
        return os.path.join(self.fpath_gen, 'surfaces_sv', geo + '.vtp')

    def get_bc_flow_path(self, geo):
        return os.path.join(self.fpath_gen, 'bc_flow', geo + '.npy')

    def get_3d_flow(self, geo):
        return os.path.join(self.fpath_gen, '3d_flow', geo + '.vtp')

    def get_3d_flow_rerun(self, geo):
        return self.gen_file('3d_flow', geo, 'vtp')

    def get_sv_flow_path(self, geo, model):
        return os.path.join(self.get_svproj_dir(geo), self.svproj.dir['flow'], 'inflow_' + model + '.flow')

    def get_sv_flow_path_rel(self, geo, model):
        sim_dir = os.path.join(self.get_svproj_dir(geo), self.svproj.dir['simulations'], geo)
        return os.path.relpath(self.get_sv_flow_path(geo, model), sim_dir)

    def get_centerline_path(self, geo):
        return os.path.join(self.fpath_gen, 'centerlines', geo + '.vtp')

    def get_centerline_vmtk_path(self, geo):
        return os.path.join(self.fpath_gen, 'centerlines_vmtk', geo + '.vtp')

    def get_centerline_outlet_path(self, geo):
        return os.path.join(self.fpath_gen, 'centerlines', 'outlets_' + geo)

    def get_surfaces_grouped_path(self, geo):
        return os.path.join(self.fpath_gen, 'surfaces_grouped', geo + '.vtp')

    def get_surfaces_cut_path(self, geo):
        return os.path.join(self.fpath_gen, 'surfaces_cut', geo + '.vtu')

    def get_surfaces_grouped_path_oned(self, geo):
        return os.path.join(self.fpath_gen, 'surfaces_grouped_oned', geo + '.vtp')

    def get_centerline_section_path(self, geo):
        return os.path.join(self.fpath_gen, 'centerlines_sections', geo + '.vtp')

    def get_section_path(self, geo):
        return os.path.join(self.fpath_gen, 'sections', geo + '.vtp')

    def get_bifurcation_path(self, geo):
        return os.path.join(self.fpath_gen, 'bifurcation_pressure', geo + '.vtp')

    def get_initial_conditions(self, geo):
        return os.path.join(self.get_sv_meshes(geo), 'initial.vtu')

    def get_sv_initial_conditions(self, geo):
        return os.path.join(self.get_svproj_dir(geo), self.svproj.dir['simulations'], geo, 'mesh-complete', 'initial.vtu')

    def gen_dir(self, name):
        fdir = os.path.join(self.fpath_study, name)
        os.makedirs(fdir, exist_ok=True)
        return fdir

    def gen_file(self, name, geo, ext='npy'):
        fdir = self.gen_dir(name)
        return os.path.join(fdir, geo + '.' + ext)

    def get_0d_flow_path(self, geo):
        return self.gen_file('0d_flow', geo)

    def get_1d_flow_path(self, geo):
        return self.gen_file('1d_flow', geo)

    def get_1d_flow_path_xdmf(self, geo):
        return self.gen_file('1d_flow', geo, 'xdmf')

    def get_post_path(self, geo, name):
        return self.gen_file('1d_3d_comparison', geo + '_' + name, 'png')

    def get_groupid_path(self, geo):
        return os.path.join(self.get_solve_dir_1d(geo), 'outletface_groupid.dat')

    def get_statistics_dir(self):
        return self.gen_dir('statistics')

    def get_solve_dir(self, geo):
        fsolve = os.path.join(self.fpath_solve, geo)
        os.makedirs(fsolve, exist_ok=True)
        return fsolve

    def get_solve_dir_0d(self, geo):
        fsolve = os.path.join(self.get_solve_dir(geo), '0d')
        os.makedirs(fsolve, exist_ok=True)
        return fsolve

    def get_solve_dir_1d(self, geo):
        fsolve = os.path.join(self.get_solve_dir(geo), '1d')
        os.makedirs(fsolve, exist_ok=True)
        return fsolve

    def get_solve_dir_3d(self, geo):
        # fsolve = os.path.join(self.get_solve_dir(geo), '3d')
        fsolve = os.path.join(self.get_svproj_dir(geo), self.svproj.dir['simulations'], geo)
        os.makedirs(fsolve, exist_ok=True)
        return fsolve

    def get_solve_dir_3d_perigee(self, geo):
        fsolve = os.path.join(self.get_solve_dir(geo), '3d_perigee')
        os.makedirs(fsolve, exist_ok=True)
        return fsolve

    def get_svproj_dir(self, geo):
        fdir = os.path.join(self.fpath_gen, 'svprojects', geo)
        os.makedirs(fdir, exist_ok=True)
        return fdir

    def get_svproj_file(self, geo):
        fdir = self.get_svproj_dir(geo)
        return os.path.join(fdir, '.svproj')

    def get_svpre_file(self, geo, solver):
        name = geo
        if solver == 'perigee':
            name += '_perigee'
        return os.path.join(self.get_solve_dir_3d(geo), name + '.svpre')

    def get_solver_file(self, geo):
        return os.path.join(self.get_solve_dir_3d(geo), 'solver.inp')

    def get_svproj_mdl_file(self, geo):
        return os.path.join(self.get_svproj_dir(geo), self.svproj.dir['models'], geo + '.mdl')

    def get_svproj_sjb_file(self, geo):
        return os.path.join(self.get_svproj_dir(geo), self.svproj.dir['simulations'], geo + '.sjb')

    def add_dict(self, dict_file, geo, add):
        dict_db = get_dict(dict_file)
        dict_db[geo] = add
        np.save(dict_file, dict_db)

    def get_log_file_1d(self):
        return os.path.join(self.fpath_solve, 'log_1d.npy')

    def add_log_file_1d(self, geo, log):
        self.add_dict(self.get_log_file_1d(), geo, log)

    def get_1d_3d_comparison(self):
        return os.path.join(os.path.dirname(self.get_post_path('', '')), '1d_3d_comparison.npy')

    def add_1d_3d_comparison(self, geo, err):
        self.add_dict(self.get_1d_3d_comparison(), geo, err)

    def get_3d_3d_comparison(self):
        return os.path.join(os.path.dirname(self.get_post_path('', '')), '3d_3d_comparison.npy')

    def add_3d_3d_comparison(self, geo, err):
        self.add_dict(self.get_3d_3d_comparison(), geo, err)

    def get_1d_geo(self, geo):
        return os.path.join(self.get_solve_dir_1d(geo), geo + '.vtp')

    def get_1d_params(self, geo):
        return os.path.join(self.get_solve_dir_1d(geo), 'parameters.npy')

    def get_seg_path(self, geo):
        return os.path.join(self.fpath_seg_path, geo)

    def get_cap_names(self, geo):
        caps = self.get_surface_names(geo, 'caps')

        bc_def, _ = self.get_bcs(geo)

        names = {}
        for c, n in bc_def['spname'].items():
            if isinstance(n, list):
                names[c] = ' '.join(n).lower().capitalize()

        for c in caps:
            if c not in names:
                names[c] = c.replace('_', ' ').lower().capitalize()

        return names

    # todo: adapt to units?
    def get_path_file(self, geo):
        return os.path.join(self.get_seg_path(geo), geo + '-cm.paths')

    # todo: adapt to units?
    def get_seg_dir(self, geo):
        return os.path.join(self.get_seg_path(geo), geo + '_groups-cm')

    def get_inflow(self, geo):
        # read inflow conditions
        flow = np.load(self.get_bc_flow_path(geo), allow_pickle=True).item()

        # read 3d boundary conditions
        bc_def, _ = self.get_bcs(geo)

        # extract inflow data
        time = flow['time']
        inflow = flow['velocity'][:, int(bc_def['preid']['inflow']) - 1]

        return time, inflow

    def get_inflow_smooth(self, geo):
        m = np.loadtxt(self.get_sv_flow_path(geo, '3d'))
        return m[:, 0], m[:, 1]

    def get_surfaces_upload(self, geo):
        surfaces = glob.glob(os.path.join(self.fpath_sim, geo, 'extras', 'mesh-surfaces', '*.vtp'))
        surfaces.append(os.path.join(self.fpath_sim, geo, 'extras', 'mesh-surfaces', 'extras', 'all_exterior.vtp'))
        return surfaces

    def add_cap_ordered(self, caps, keys_ordered, keys_left, c):
        for k in sorted(caps):
            if c in k.lower() and k in keys_left:
                keys_ordered.append(k)
                keys_left.remove(k)

    def get_surfaces(self, geo, surf='all'):
        fdir = self.get_surface_dir(geo)
        surfaces_all = glob.glob(os.path.join(fdir, '*.vtp'))
        if surf == 'all':
            surfaces = surfaces_all
        elif surf == 'outlets' or surf == 'caps':
            exclude = ['all_exterior', 'wall', 'stent']
            if surf == 'outlets':
                exclude += ['inflow']
            surfaces = [x for x in surfaces_all if not any(e in x for e in exclude)]
        elif surf in self.get_surface_names(geo):
            surfaces = os.path.join(fdir, surf + '.vtp')
        else:
            print('Unknown surface option ' + surf)
            surfaces = []
        return surfaces

    def get_surface_names(self, geo, surf='all'):
        surfaces = self.get_surfaces(geo, surf)
        surfaces = [surfaces] if isinstance(surfaces, str) else surfaces
        surfaces = [os.path.splitext(os.path.basename(s))[0] for s in surfaces]
        surfaces.sort()

        if surf == 'caps':
            # nicely ordered cap names for output
            surfaces = sorted(surfaces)
            caps = surfaces.copy()
            keys_left = surfaces.copy()
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
            surfaces = keys_ordered

        return surfaces

    def get_surface_ids(self, geo, surf='all'):
        surfaces = self.get_surface_names(geo, surf)
        bc_def, _ = self.get_bcs(geo)
        ids = []
        for s in surfaces:
            ids += [int(float(bc_def['spid'][s]))]
        ids.sort()
        return np.array(ids)

    def get_volume(self, geo):
        return os.path.join(self.fpath_sim, geo, 'results', geo + '_sim_results_in_cm.vtu')

    def get_volume_mesh(self, geo):
        return os.path.join(self.get_sv_meshes(geo), geo + '.vtu')

    def get_res_3d_vol_rerun(self, geo):
        return os.path.join(self.fpath_study, '3d_flow', geo, geo + '.vtu')

    def get_res_3d_surf_rerun(self, geo):
        return os.path.join(self.fpath_study, '3d_flow', geo, geo + '.vtp')

    def get_outlet_names(self, geo):
        bc_def, _ = self.get_bcs(geo)
        if bc_def is None:
            return None
        names = [k for k, v in sorted(bc_def['preid'].items(), key=lambda kv: kv[1])]

        names_out = []
        for n in names:
            if 'wall' not in n and 'inflow' not in n and 'stent' not in n:
                names_out += [n]
        return names_out

    def count_inlets(self, geo):
        n_inlet = 0
        for s in self.get_surface_names(geo):
            if 'inflow' in s:
                n_inlet += 1
        return n_inlet

    def read_results(self, fpath):
        if os.path.exists(fpath):
            res = np.load(fpath, allow_pickle=True).item()
        else:
            print('no results in ' + fpath)
            return None

        if 'pressure' not in res or len(res['pressure']) == 0:
            print('results empty in ' + fpath)
            return None

        return res

    def get_constants(self, geo):
        # get simulation parameters
        _, params = self.get_bcs(geo)

        if 'sim_density' in params:
            constants = {'density': float(params['sim_density']), 'viscosity': float(params['sim_viscosity'])}
        else:
            return None

        # convert units
        for name, val in constants.items():
            constants[name] = get_in_model_units(params['sim_units'], name, val)

        return constants

    def get_3d_timestep(self, geo):
        # get model parameters
        _, params = self.get_bcs(geo)

        # read inflow conditions
        time, inflow = self.get_inflow(geo)

        # number of time steps
        numstep = int(float(params['sim_steps_per_cycle']))

        # time step
        return time[-1] / numstep


class SimVascular:
    """
    simvascular object to handle external calls
    """

    def __init__(self):
        self.svpre = '/usr/local/sv/svsolver/2019-02-07/svpre'
        self.svsolver = '/usr/local/sv/svsolver/2019-02-07/svsolver'
        self.svpost = '/home/pfaller/work/repos/svSolver/build/svSolver-build/bin/svpost'
        self.onedsolver = '/home/pfaller/work/repos/svOneDSolver/build_superlu/bin/OneDSolver'
        self.sv = '/home/pfaller/work/repos/SimVascular/build/SimVascular-build/sv'
        self.sv_legacy_io = '/home/pfaller/work/repos/SimVascularLegacyIO/build/SimVascular-build/sv'
        # self.sv_debug = '/home/pfaller/work/repos/SimVascular/build_debug/SimVascular-build/sv'
        self.sv_debug = '/home/pfaller/work/repos/SimVascular/build_debug/SimVascular-build/bin/simvascular'
        self.perigee = '/home/pfaller/work/repos/PERIGEE/tools/sv_file_converter/build'

    def run_pre(self, pre_folder, pre_file):
        subprocess.run([self.svpre, pre_file], cwd=pre_folder)

    def run_solver(self, run_folder, run_file='solver.inp'):
        subprocess.run([self.svsolver, run_file], cwd=run_folder)

    def run_post(self, run_folder, args):
        subprocess.run([self.svpost] + args, cwd=run_folder, stdout=open(os.devnull, "w"))
        # run_command(run_folder, [self.svpost, args])

    def run_solver_1d(self, run_folder, run_file='solver.inp'):
        run_command(run_folder, [self.onedsolver, run_file])  # 'mpirun', '-np', '4',
        return ' ', True

    def run_python(self, command):
        return subprocess.run([self.sv, ' --python -- '] + command)

    def run_python_legacyio(self, command):
        p = subprocess.Popen([self.sv_legacy_io, ' --python -- '] + command, stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE)
        m = ''
        for s in [self.sv_legacy_io, ' --python -- '] + command:
            m += s + ' '
        # print(m)
        return p.communicate()

    def run_python_debug(self, command):
        command = [self.sv, ' --python -- '] + command
        out_str = ''
        for c in command:
            out_str += c + ' '
        print(out_str)
        # return subprocess.run(['gdb', self.sv_debug])

    def run_perigee_sv_converter(self):
        run_command(run_folder, [self.onedsolver, run_file])


def get_tcl_paths(fpath, geo, offset):
    assert len(geo) == 9 and geo[4] == '_' and is_int(geo[:4]) and is_int(geo[5:]), geo + ' not in OSMSC format'
    ids = geo.split('_')
    geo_bc = ids[0] + '_' + str(int(ids[1]) + offset).zfill(4)
    return os.path.join(fpath, geo_bc + '.tcl'), os.path.join(fpath, geo_bc + '-bc.tcl')


class SVProject:
    def __init__(self):
        self.dir = {'images': 'Images', 'paths': 'Paths', 'segmentations': 'Segmentations', 'models': 'Models',
                    'meshes': 'Meshes', 'simulations': 'Simulations', 'flow': 'flow-files'}
        self.t = '    '


class Post:
    def __init__(self):
        self.fields = ['pressure', 'flow', 'area']
        self.units = {'pressure': 'mmHg', 'flow': 'l/h', 'area': 'mm^2'}
        self.styles = {'3d': '-', '3d_rerun': '-', '1d': '-', '0d': ':'}
        self.colors = {'3d': 'C0', '3d_rerun': 'C1', '1d': 'C1', 'r': 'C2'}

        self.cgs2mmhg = 7.50062e-4
        self.mlps2lph = 60 / 1000
        self.convert = {'pressure': self.cgs2mmhg, 'flow': self.mlps2lph, 'area': 100}

        # sets the plot order
        # self.models = ['3d', '1d']
        self.models = ['3d', '3d_rerun']


def run_command(run_folder, command):
    process = subprocess.Popen(command, cwd=run_folder, stdout=subprocess.PIPE, universal_newlines=True)
    while True:
        output = process.stdout.readline()
        if output == '' and process.poll() is not None:
            break
        if output:
            print(output.strip())
    rc = process.poll()
    return rc


def is_int(s):
    try:
        int(s)
        return True
    except ValueError:
        return False


def exists(fpath):
    if os.path.exists(fpath):
        return fpath
    else:
        return None
