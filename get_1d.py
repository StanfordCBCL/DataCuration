#!/usr/bin/env python

import sv
import glob
import io
import os
import pdb
import re
import shutil
import sys

import numpy as np

from get_database import Database, SimVascular, Post, input_args
from get_sv_project import write_bc, write_inflow
from simulation_io import read_results_1d, get_caps_db

sys.path.append('/home/pfaller/work/repos/SimVascular/Python/site-packages/')

import sv_1d_simulation as oned


def get_params(db, geo):
    # store parameters for export
    params = {}
    
    params['fpath_1d'] = db.get_solve_dir_1d(geo)
    fpath_geo = os.path.join(params['fpath_1d'], 'geometry')

    # reference pressure (= initial pressure?)
    if os.path.exists(db.get_bc_flow_path(geo)):
        res_3d = db.read_results(db.get_bc_flow_path(geo))
        params['pref'] = res_3d['pressure'][-1, 0]
    else:
        raise RuntimeError('3d results do not exist')

    if db.has_loop(geo):
        raise RuntimeError('3d geometry contains a loop')

    # copy surface model to folder if it exists
    if os.path.exists(db.get_surfaces(geo, 'all_exterior')):
        shutil.copy2(db.get_surfaces(geo, 'all_exterior'), fpath_geo)
    else:
        raise RuntimeError('3d geometry does not exist')

    # check if geometry has no or multiple inlets
    n_inlet = db.count_inlets(geo)
    if n_inlet == 0:
        raise RuntimeError('3d geometry has no inlet')
    elif n_inlet > 1:
        raise RuntimeError('3d geometry has multiple inlets (' + repr(n_inlet) + ')')

    # assume pre-computed centerlines from SimVascular C++
    params['centerlines_input_file'] = db.get_centerline_path(geo)
    if not os.path.exists(params['centerlines_input_file']):
        raise RuntimeError('centerline does not exist')

    # get boundary conditions
    params['bc_def'] = db.get_bcs(geo)
    if params['bc_def'] is None:
        raise RuntimeError('no boundary conditions')

    # write outlet boundary conditions to file if they exist
    params['bc_types'], err = write_bc(params['fpath_1d'], db, geo, model='1d')
    if err:
        raise RuntimeError(err)

    # get inflow
    time, _ = db.get_inflow_smooth(geo)
    if time is None:
        raise RuntimeError('inflow does not exist')
    params['inflow_input_file'] = db.get_sv_flow_path(geo, '1d')

    # get outlets
    caps = get_caps_db(db, geo)
    del caps['inflow']
    params['outlets'] = list(caps.keys())

    # number of cycles to run
    params['n_cycle'] = 50

    # sub-segment size
    params['seg_min_num'] = 1
    params['seg_size'] = 999

    # FEM size
    params['min_num_elems'] = 10
    params['element_size'] = 0.1

    # mesh adaptive?
    params['seg_size_adaptive'] = True

    # set simulation time as end of 3d simulation
    params['save_data_freq'] = 1
    params['dt'] = 1.0e-3

    # run all cycles
    params['num_dts'] = int(time[-1] * params['n_cycle'] / params['dt'] + 1.0)

    return params


def generate_1d_api(db, geo):
    # get parameters
    params_db = get_params(db, geo)

    # create 1d simulation
    oned_simulation = sv.simulation.OneDimensional()

    # Create 1D simulation parameters
    params = sv.simulation.OneDimensionalParameters()

    # Mesh parameters
    mesh_params = params.MeshParameters()

    # Model parameters
    model_params = params.ModelParameters()
    model_params.name = geo
    model_params.inlet_face_names = ['inflow']
    model_params.outlet_face_names = params_db['outlets']
    model_params.centerlines_file_name = params_db['centerlines_input_file']

    # Fluid properties
    fluid_props = params.FluidProperties()

    # Set wall properties
    material = params.WallProperties.OlufsenMaterial()

    # Set boundary conditions
    bcs = params.BoundaryConditions()

    # add inlet
    bcs.add_velocities(face_name='inlet', file_name=params_db['inflow_input_file'])

    # add outlets
    for cp in params_db['outlets']:
        t = params_db['bc_def']['bc_type'][cp]
        bc = params_db['bc_def']['bc'][cp]
        if 'Po' in bc and bc['Po'] != 0.0:
            raise ValueError('RCR reference pressure not implemented')
        if t == 'rcr':
            bcs.add_rcr(face_name=cp, Rp=bc['Rp'], C=bc['C'], Rd=bc['Rd'])
        elif t == 'resistance':
            bcs.add_resistance(face_name=cp, resistance=bc['R'])
        elif t == 'coronary':
            raise ValueError('Coronary BC not implemented')

    # Set solution parameters
    solution_params = params.Solution()
    solution_params.time_step = params_db['dt']
    solution_params.num_time_steps = params_db['num_dts']
    solution_params.save_frequency = params_db['save_data_freq']

    # Write a 1D solver input file
    oned_simulation.write_input_file(model=model_params, mesh=mesh_params, fluid=fluid_props,
                                     material=material, boundary_conditions=bcs, solution=solution_params,
                                     directory=params_db['fpath_1d'])
    return None


def generate_1d(db, geo):
    # get parameters
    params = get_params(db, geo)

    # set simulation paths
    fpath_geo = os.path.join(params['fpath_1d'], 'geometry')
    fpath_surf = os.path.join(fpath_geo, 'surfaces')

    # copy outlet surfaces
    for f in db.get_surfaces(geo, 'caps'):
        shutil.copy2(f, fpath_surf)

    # copy outlet names
    fpath_outlets = os.path.join(params['fpath_1d'], 'outlets')
    shutil.copy(db.get_centerline_outlet_path(geo), fpath_outlets)

    # write inflow
    write_inflow(db, geo, '1d')

    # try:
    if True:
        oned.run(boundary_surfaces_directory=fpath_surf,
                 centerlines_input_file=params['centerlines_input_file'],
                 centerlines_output_file=None,
                 compute_centerlines=False,
                 compute_mesh=True,
                 density=params['bc_def']['params']['sim_density'],
                 element_size=params['element_size'],
                 inlet_face_input_file='inflow.vtp',
                 inflow_input_file=db.get_sv_flow_path(geo, '1d'),
                 linear_material_ehr=1e15,
                 linear_material_pressure=params['pref'],
                 material_model=None,
                 mesh_output_file=geo + '.vtp',
                 min_num_elements=params['min_num_elems'],
                 model_name=geo,
                 num_time_steps=params['num_dts'],
                 olufsen_material_k1=None,
                 olufsen_material_k2=None,
                 olufsen_material_k3=None,
                 olufsen_material_exp=2.0,
                 olufsen_material_pressure=params['pref'],
                 outflow_bc_input_file=params['fpath_1d'],
                 outflow_bc_type=params['bc_types'],
                 outlet_face_names_input_file=fpath_outlets,
                 output_directory=params['fpath_1d'],
                 seg_min_num=params['seg_min_num'],
                 seg_size=params['seg_size'],
                 seg_size_adaptive=params['seg_size_adaptive'],
                 solver_output_file=geo + '.inp',
                 save_data_frequency=params['save_data_freq'],
                 surface_model=os.path.join(fpath_geo, 'all_exterior.vtp'),
                 time_step=params['dt'],
                 uniform_bc=True,
                 units='cm',
                 viscosity=params['bc_def']['params']['sim_viscosity'],
                 wall_properties_input_file=None,
                 wall_properties_output_file=None,
                 write_mesh_file=True,
                 write_solver_file=True)
    # except Exception as e:
    #     return repr(e)

    return None


def main(db, geometries):
    # simvascular object
    sv = SimVascular()

    for geo in geometries:
        print('Running geometry ' + geo)

        # if os.path.exists(db.get_1d_flow_path(geo)):
        #     print('  skipping')
        #     continue

        # generate oneDSolver input file and check if successful
        # msg = generate_1d(db, geo)
        msg = generate_1d_api(db, geo)

        # msg = None
        # if False:
        if not msg:
            # run oneDSolver
            # sv.run_solver_1d(db.get_solve_dir_1d(geo), geo + '.inp')
            sv.run_solver_1d(db.get_solve_dir_1d(geo), 'solver.in')
            # with open(os.path.join(db.get_solve_dir_1d(geo), 'solver.log'), 'w+') as f:
            #     f.write(out_solve)

            # extract results
            res_dir = db.get_solve_dir_1d(geo)
            params_file = db.get_1d_params(geo)
            results_1d = read_results_1d(res_dir, params_file)

            # save results
            if results_1d['flow']:
                # save results in dict
                np.save(db.get_1d_flow_path(geo), results_1d)

                # remove 1d output files
                for f in glob.glob(os.path.join(res_dir, geo + 'branch*seg*_*.dat')):
                    os.remove(f)

                msg = 'success'
            else:
                msg = 'unconverged'
        else:
            print('  skipping (1d model creation failed)\n  ' + msg)

        # store errors in file
        db.add_log_file_1d(geo, msg)


if __name__ == '__main__':
    descr = 'Automatically create, run, and post-process 1d-simulations'
    d, g, _ = input_args(descr)
    main(d, g)
