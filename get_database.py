#!/usr/bin/env python

import os
import shutil
import glob
import subprocess
import csv
import re
import argparse
import pdb

import numpy as np
import scipy.interpolate
from collections import OrderedDict

from vtk.util.numpy_support import numpy_to_vtk as n2v
from vtk.util.numpy_support import vtk_to_numpy as v2n

from get_bcs import get_bcs
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
    elif param.geo[-1] == ':':
        geo_all = database.get_geometries()
        geo_first = geo_all.index(param.geo[:-1])
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

        # initialize database containing all model information
        # todo: use advanced database that could be connected to webserver, like mysql
        self.database = {}

        # load from hdd
        self.load()

        # svproject object
        self.svproj = SVProject()

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

    def exclude_geometries(self, geometries_in):
        # excluded geometries by Nathan
        imaging = ['0001', '0020', '0044']
        animals = ['0066', '0067', '0068', '0069', '0070', '0071', '0072', '0073', '0074']
        single_vessel = ['0158', '0164', '0165']

        # temporary: wrong integration due to displaced volumetric geometry
        error = ['0091', '0092', '0154']

        exclude = imaging + animals + single_vessel + error

        geometries = []
        for g in geometries_in:
            if g[:4] not in exclude:
                geometries += [g]

        return geometries

    def get_geometries(self):
        geometries = os.listdir(self.fpath_sim)
        geometries.sort()
        return geometries

    def get_geometries_select(self, name):
        if name == 'paper':
            # pick alpl geometries in rest state
            geometries = []
            for geo in self.get_geometries():
                _, params = self.get_bcs(geo)
                if params is not None and params['sim_physio_state'] == 'rest':
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
        else:
            raise Exception('Unknown selection ' + name)
        return geometries

    def get_bcs(self, geo):
        # try two different offsets of tcl name vs geo name
        # todo: find out why there are several variants
        for o in [-1, 0, -1001]:
            tcl, tcl_bc = self.get_tcl_paths(geo, o)
            if os.path.exists(tcl) and os.path.exists(tcl_bc):
                return get_bcs(tcl, tcl_bc)
        return None, None

    def get_png(self, geo):
        return os.path.join(self.fpath_png, 'OSMSC' + geo + '_sim.png')

    def get_img(self, geo):
        return exists(os.path.join(self.fpath_sim, geo, 'image_data', 'vti', 'OSMSC' + geo[:4] + '-cm.vti'))

    def get_tcl_paths(self, geo, offset):
        assert len(geo) == 9 and geo[4] == '_' and is_int(geo[:4]) and is_int(geo[5:]), geo + ' not in OSMSC format'
        ids = geo.split('_')
        geo_bc = ids[0] + '_' + str(int(ids[1]) + offset).zfill(4)
        return os.path.join(self.fpath_bc, geo_bc + '.tcl'), os.path.join(self.fpath_bc, geo_bc + '-bc.tcl')

    def get_surface_dir(self, geo):
        return os.path.join(self.fpath_gen, 'surfaces', geo)

    def get_sv_surface(self, geo):
        return exists(os.path.join(self.fpath_gen, 'surfaces_sv', geo + '.vtp'))

    def get_bc_flow_path(self, geo):
        return os.path.join(self.fpath_gen, 'bc_flow', geo + '.npy')

    def get_3d_flow_path(self, geo):
        return os.path.join(self.fpath_gen, '3d_flow', geo + '.npy')

    def get_3d_flow_path_oned(self, geo):
        return os.path.join(self.fpath_gen, '3d_flow_oned', geo + '.npy')

    def get_3d_flow_path_oned_vtp(self, geo):
        return os.path.join(self.fpath_gen, '3d_flow_oned', geo + '.vtp')

    def get_3d_flow_path_old(self, geo):
        return os.path.join(self.fpath_gen, '3d_flow_no_exclusion', geo + '.npy')

    def get_centerline_path(self, geo):
        return os.path.join(self.fpath_gen, 'centerlines', geo + '.vtp')

    def get_centerline_outlet_path(self, geo):
        return os.path.join(self.fpath_gen, 'centerlines', 'outlets_' + geo)

    def get_surfaces_grouped_path(self, geo):
        return os.path.join(self.fpath_gen, 'surfaces_grouped', geo + '.vtp')

    def get_surfaces_cut_path(self, geo):
        return os.path.join(self.fpath_gen, 'surfaces_cut', geo + '.vtu')

    def get_surfaces_grouped_path_oned(self, geo):
        return os.path.join(self.fpath_gen, 'surfaces_grouped_oned', geo + '.vtp')

    def get_centerline_path_1d(self, geo):
        return os.path.join(self.fpath_gen, 'centerlines_from_1d', geo + '.vtp')

    def get_centerline_path_oned(self, geo):
        return os.path.join(self.fpath_gen, 'centerlines_oned', geo + '.vtp')

    def get_centerline_path_raw(self, geo):
        return os.path.join(self.fpath_gen, 'centerlines_raw', geo + '.vtp')

    def get_centerline_section_path(self, geo):
        return os.path.join(self.fpath_gen, 'centerlines_sections', geo + '.vtp')

    def get_section_path(self, geo):
        return os.path.join(self.fpath_gen, 'sections', geo + '.vtp')

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
        fsolve = os.path.join(self.get_solve_dir(geo), '3d')
        os.makedirs(fsolve, exist_ok=True)
        return fsolve

    def get_svproj_dir(self, geo):
        fdir = os.path.join(self.fpath_gen, 'svprojects', geo)
        os.makedirs(fdir, exist_ok=True)
        return fdir

    def get_svproj_file(self, geo):
        fdir = self.get_svproj_dir(geo)
        return os.path.join(fdir, '.svproj')

    def get_svproj_mdl_file(self, geo):
        return os.path.join(self.get_svproj_dir(geo), self.svproj.dir['models'], geo + '.mdl')

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

    def get_1d_geo(self, geo):
        return os.path.join(self.get_solve_dir_1d(geo), geo + '.vtp')

    def get_1d_params(self, geo):
        return os.path.join(self.get_solve_dir_1d(geo), 'parameters.npy')

    def get_seg_path(self, geo):
        return os.path.join(self.fpath_seg_path, geo)

    # todo: adapt to units?
    def get_path_file(self, geo):
        return os.path.join(self.get_seg_path(geo), geo + '-cm.paths')

    # todo: adapt to units?
    def get_seg_dir(self, geo):
        return os.path.join(self.get_seg_path(geo), geo + '_groups-cm', '*')

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
        return os.path.join(self.fpath_gen, 'volumes', geo + '.vtu')

    def get_outlet_names(self, geo):
        return self.get_surface_names(geo, 'outlets')

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

        constants = {'density': float(params['sim_density']), 'viscosity': float(params['sim_viscosity'])}

        # no conversion for units cgs
        if params['sim_units'] == 'cm':
            pass
        # convert cgm to cgs
        elif params['sim_units'] == 'mm':
            constants['density'] *= 1000
            constants['viscosity'] *= 10
        else:
            raise ValueError('Unknown units ' + units)

        return constants

    def copy_files(self, geo):
        # define paths
        fpath_surf = os.path.join(self.get_solve_dir_3d(geo), 'mesh-complete', 'mesh-surfaces')

        # create simulation folder
        os.makedirs(fpath_surf, exist_ok=True)

        # copy generic solver settings
        shutil.copy('solver.inp', self.get_solve_dir_3d(geo))

        # copy geometry
        for f in glob.glob(os.path.join(self.fpath_gen, 'surfaces', geo, '*.vtp')):
            shutil.copy(f, fpath_surf)

        # copy volume mesh
        # todo: copy without results
        fpath_res = os.path.join(self.fpath_sim, geo, 'results', geo + '_sim_results_in_cm.vtu')
        shutil.copy(fpath_res, os.path.join(self.get_solve_dir_3d(geo), 'mesh-complete'))


class SimVascular:
    """
    simvascular object to handle external calls
    """
    def __init__(self):
        self.svpre = '/usr/local/sv/svsolver/2019-02-07/svpre'
        self.svsolver = '/usr/local/sv/svsolver/2019-02-07/svsolver'
        self.onedsolver = '/home/pfaller/work/repos/oneDSolver/build/bin/OneDSolver'
        self.sv = '/home/pfaller/work/repos/SimVascular/build/SimVascular-build/sv'
        self.sv_legacy_io = '/home/pfaller/work/repos/SimVascularLegacyIO/build/SimVascular-build/sv'
        # self.sv_debug = '/home/pfaller/work/repos/SimVascular/build_debug/SimVascular-build/sv'
        self.sv_debug = '/home/pfaller/work/repos/SimVascular/build_debug/SimVascular-build/bin/simvascular'

    def run_pre(self, pre_folder, pre_file):
        subprocess.run([self.svpre, pre_file], cwd=pre_folder)

    def run_solver(self, run_folder, run_file='solver.inp'):
        subprocess.run([self.svsolver, run_file], cwd=run_folder)

    def run_solver_1d(self, run_folder, run_file='solver.inp'):
        run_command(run_folder, [self.onedsolver, run_file])
        return ' ', True

    def run_python(self, command):
        return subprocess.run([self.sv, ' --python -- '] + command)

    def run_python_legacyio(self, command):
        p = subprocess.Popen([self.sv_legacy_io, ' --python -- '] + command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return p.communicate()

    def run_python_debug(self, command):
        command = [self.sv, ' --python -- '] + command
        out_str = ''
        for c in command:
            out_str += c + ' '
        print(out_str)
        # return subprocess.run(['gdb', self.sv_debug])

class SVProject:
    def __init__(self):
        self.dir = {'images': 'Images', 'paths': 'Paths', 'segmentations': 'Segmentations', 'models': 'Models',
                    'meshes': 'Meshes', 'simulations': 'Simulations'}
        self.t = '    '


class Post:
    def __init__(self):
        self.fields = ['pressure', 'flow', 'area']#
        self.units = {'pressure': 'mmHg', 'flow': 'l/h', 'area': 'mm^2'}
        self.styles = {'3d': '-', '1d': '--', '0d': ':'}
        self.colors = {'3d': 'C0', '1d': 'C1', 'r': 'C2'}

        self.cgs2mmhg = 7.50062e-4
        self.mlps2lph = 60 / 1000
        self.convert = {'pressure': self.cgs2mmhg, 'flow': self.mlps2lph, 'area': 100}

        # sets the plot order
        self.models = ['3d', '1d'] #'0d',


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
