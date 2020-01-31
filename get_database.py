#!/usr/bin/env python

import os
import shutil
import glob
import subprocess
import csv
import re
import pdb

import numpy as np
import scipy.interpolate
from collections import OrderedDict

from vtk.util.numpy_support import numpy_to_vtk as n2v
from vtk.util.numpy_support import vtk_to_numpy as v2n

from get_bcs import get_bcs
from vtk_functions import read_geo
from simulation_io import get_dict


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
        exclude = imaging + animals + single_vessel

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

    def get_centerline_path(self, geo):
        return os.path.join(self.fpath_gen, 'centerlines', geo + '.vtp')

    def get_centerline_path_1d(self, geo):
        return os.path.join(self.fpath_gen, 'centerlines_from_1d', geo + '.vtp')

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

    def get_surfaces_upload(self, geo):
        surfaces = glob.glob(os.path.join(self.fpath_sim, geo, 'extras', 'mesh-surfaces', '*.vtp'))
        surfaces.append(os.path.join(self.fpath_sim, geo, 'extras', 'mesh-surfaces', 'extras', 'all_exterior.vtp'))
        return surfaces

    def add_cap_ordered(self, caps, keys_ordered, keys_left, c):
        for k in sorted(caps.keys()):
            if c in k.lower() and k in keys_left:
                keys_ordered.append(k)
                keys_left.remove(k)

    def read_centerline(self, fpath):
        reader_1d, nodes_1d, cells_1d = read_geo(fpath)
        group = v2n(cells_1d.GetArray('group'))
        seg_id = v2n(cells_1d.GetArray('seg_id'))
        point_id = v2n(nodes_1d.GetArray('point_id'))
        path = v2n(nodes_1d.GetArray('path'))

        grp = np.unique(group)
        grp_nodes = {}
        for g in grp:
            points_group = []
            for c in np.where(group == g)[0]:
                for i in range(2):
                    points_group += [reader_1d.GetOutput().GetCell(c).GetPointId(i)]
            grp_nodes[g] = np.unique(points_group)

        return group, seg_id, grp_nodes, path

    def get_xd_map(self, geo):
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

        # add inlet/oulet SegIds
        group, seg_id, grp_nodes, path = self.read_centerline(os.path.join(self.get_solve_dir_1d(geo), geo + '.vtp'))
        for k in caps.keys():
            g = group[seg_id == caps[k]['GroupId']][0]
            # caps[k]['SegId'] = seg_id[group == g]
            caps[k]['SegId'] = grp_nodes[g]
            caps[k]['BranchId'] = g
            caps[k]['path_1d'] = path[caps[k]['SegId']]
            if not k == 'inflow':
                caps[k]['path_1d'][0] = 0

        group, seg_id, grp_nodes, path = self.read_centerline(self.get_centerline_path_1d(geo))
        for k in caps.keys():
            # caps[k]['SegId_cent'] = seg_id[group == caps[k]['BranchId']]
            caps[k]['SegId_cent'] = grp_nodes[caps[k]['BranchId']]
            caps[k]['path_3d'] = path[caps[k]['SegId_cent']]
            if not k == 'inflow':
                caps[k]['path_3d'][0] = 0

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

    def get_results_xd(self, geo):
        # get post-processing object
        post = Post()

        # read results
        # res_0d = self.read_results(geo, self.get_0d_flow_path(geo))
        # if res_0d is None:
        #     return None, None

        res_1d = self.read_results(self.get_1d_flow_path(geo))
        if res_1d is None:
            return None, None

        res_3d_caps = self.read_results(self.get_bc_flow_path(geo))
        if res_3d_caps is None:
            return None, None

        res_3d_interior = self.read_results(self.get_3d_flow_path(geo))
        if res_3d_interior is None:
            return None, None

        # get map between 3d BC_FaceID and 1d GroupId for all inlet/outlets
        xd_map = self.get_xd_map(geo)

        # rename 3d results
        res_3d_caps['flow'] = res_3d_caps['velocity']
        res_3d_interior['flow'] = res_3d_interior['velocity']
        del res_3d_caps['velocity']
        del res_3d_interior['velocity']

        # time steps
        time = {'3d': res_3d_caps['time'],
                '3d_all': res_3d_caps['time']}

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
        n_cycle_1d = max(1, int(time['1d_all'][-1] // res_3d_caps['time'][-1]))
        time_3d_last_1d = res_3d_caps['time'] + (n_cycle_1d - 1) * res_3d_caps['time'][-1]

        # collect results
        res = {'path': {}}
        for f in post.fields:
            res[f] = {}
            for k, v in xd_map.items():

                # get last element in 1d segment
                i_1d = -1
                s_3d = 1
                if k == 'inflow':
                    # get first 1d segment
                    i_1d = 0

                    if f == 'flow':
                        # reverse flow direction so that all caps have positive flow (looks nicer)
                        s_3d = -1

                # assemble simulation results
                res[f][k] = {}
                res['path'][k] = {}

                if res_1d is not None:
                    res['path'][k]['1d'] = np.array([0])

                    # res['path'][k]['3d'] = res_3d_interior['path'][v['SegId_cent']]
                    res[f][k]['1d_all'] = res_1d[f][v['GroupId']][i_1d]

                    # interpolate 1d results to 3d time steps of last cycle
                    interp = scipy.interpolate.interp1d(time['1d_all'], res[f][k]['1d_all'], bounds_error=False)
                    res[f][k]['1d'] = interp(time_3d_last_1d)

                    # interpolate 1d branch in time to 3d
                    # res[f][k]['1d_int'] = np.zeros((time_3d_last_1d.shape[0], v['SegId'].shape[0]))
                    # assert res[f][k]['1d_int'].shape == res[f][k]['3d_int'].shape, 'size of 1d and 3d results do not agree'

                    # add inlet FE
                    res_inflow = res_1d[f][v['SegId'][1] - 1][0]
                    interp = scipy.interpolate.interp1d(time['1d_all'], res_inflow, bounds_error=False)
                    res[f][k]['1d_int'] = interp(time_3d_last_1d).reshape(-1, 1)

                    # add all FE of segments and their coordinates
                    for i, s in enumerate(v['SegId'][1:]):
                        # always exclude first element (identical with last element of previous segment)
                        res_seg = res_1d[f][s - 1][1:]

                        # generate path for segment FEs, assuming equidistant spacing
                        path_1d = np.linspace(v['path_1d'][i], v['path_1d'][i+1], res_seg.shape[0] + 1)[1:]

                        # append paths of all segments
                        res['path'][k]['1d'] = np.hstack((res['path'][k]['1d'], path_1d))

                        # interpolate results to 3D time steps
                        interp = scipy.interpolate.interp1d(time['1d_all'], res_seg, bounds_error=False)
                        res[f][k]['1d_int'] = np.hstack((res[f][k]['1d_int'], interp(time_3d_last_1d).T))
                    # if k=='rt_carotid' and f=='flow':
                    #     pdb.set_trace()

                if not time['1d_all'].shape[0] == res[f][k]['1d_all'].shape[0]:
                    print('time steps not matching results for 1d results ' + f + ' in GroupId ' + repr(v['GroupId']))
                    pdb.set_trace()
                    return None, None

                res[f][k]['3d'] = res_3d_caps[f][:, v['BC_FaceID'] - 1] * s_3d
                res[f][k]['3d_all'] = res[f][k]['3d']

                # indicator for branching
                is_vessel = np.array(1 - res_3d_interior['is_branch'][v['SegId_cent'] - 1], dtype=bool)
                if not is_vessel[1]:
                    is_vessel[0] = False

                # read interior results
                # res['path'][k]['3d'] = res_3d_interior['path'][v['SegId_cent']]
                # res['path'][k]['3d'] = v['path_3d'][is_branch]
                res['path'][k]['3d'] = v['path_3d'][is_vessel]
                res[f][k]['3d_int'] = res_3d_interior[f][:, v['SegId_cent'] - 1][:, is_vessel]

                # replace cap integrals
                res[f][k]['3d_int'][:, i_1d] = res[f][k]['3d']

                # if v['SegId'] in res_0d[f]:
                #     res[f][k]['0d_all'] = res_0d[f][v['SegId']]
                # else:
                #     res[f][k]['0d_all'] = np.zeros(time['time_0d_all'].shape)
                # # interpolate 0d results to 3d time steps of last cycle
                # interp = scipy.interpolate.interp1d(time['time_0d_all'], res[f][k]['0d_all'])
                #
                # # 3d-time moved to the last full 1d cycle (for interpolation)
                # n_cycle_0d = int(time['time_0d_all'][-1] // res_3d['time'][-1])
                # time_3d_last_0d = res_3d['time'] + (n_cycle_0d - 1) * res_3d['time'][-1]
                # res[f][k]['0d'] = interp(time_3d_last_0d)

        return res, time


class SimVascular:
    """
    simvascular object to handle external calls
    """
    def __init__(self):
        self.svpre = '/usr/local/sv/svsolver/2019-02-07/svpre'
        self.svsolver = '/usr/local/sv/svsolver/2019-02-07/svsolver'
        self.onedsolver = '/home/pfaller/work/repos/oneDSolver/build/bin/OneDSolver'
        self.sv = '/home/pfaller/work/repos/SimVascular/build/SimVascular-build/sv'

    def run_pre(self, pre_folder, pre_file):
        subprocess.run([self.svpre, pre_file], cwd=pre_folder)

    def run_solver(self, run_folder, run_file='solver.inp'):
        subprocess.run([self.svsolver, run_file], cwd=run_folder)

    def run_solver_1d(self, run_folder, run_file='solver.inp'):
        run_command(run_folder, [self.onedsolver, run_file])
        return ' ', True

    def run_python(self, command):
        return subprocess.run([self.sv, ' --python -- '] + command)

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
