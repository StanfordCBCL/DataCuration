#!/usr/bin/env python

import contextlib
import csv
import glob
import io
import os
import pdb
import re
import shutil
import sys
import argparse

import numpy as np

from get_database import Database, SimVascular, Post, input_args
from get_sim import write_bc
from simulation_io import read_results_1d

sys.path.append('/home/pfaller/work/repos/SimVascular/Python/site-packages/')

import sv_1d_simulation as oned


def generate_1d(db, geo):
    # use all options, set to None if using defaults (in cgs units)

    # number of cycles to run
    n_cycle = 10

    # sub-segment size
    seg_min_num = 2
    seg_size = 999

    # FEM size
    min_num_elems = 10
    element_size = 0.1

    # mesh adaptive?
    seg_size_adaptive = True

    # set simulation paths and create folders
    fpath_1d = db.get_solve_dir_1d(geo)
    fpath_geo = os.path.join(fpath_1d, 'geometry')
    fpath_surf = os.path.join(fpath_geo, 'surfaces')
    os.makedirs(fpath_geo, exist_ok=True)
    os.makedirs(fpath_surf, exist_ok=True)

    if os.path.exists(db.get_bc_flow_path(geo)):
        res_3d = db.read_results(db.get_bc_flow_path(geo))
    else:
        return '3d results do not exist'

    # copy surface model to folder if it exists
    if os.path.exists(db.get_surfaces(geo, 'all_exterior')):
        shutil.copy2(db.get_surfaces(geo, 'all_exterior'), fpath_geo)
    else:
        return '3d geometry does not exist'

    # check if geometry has no or multiple inlets
    n_inlet = db.count_inlets(geo)
    if n_inlet == 0:
        return '3d geometry has no inlet'
    elif n_inlet > 1:
        return '3d geometry has multiple inlets (' + repr(n_inlet) + ')'

    # assume pre-computed centerlines from SimVascular C++ (don't use python vmtk interfance)
    centerlines_input_file = db.get_centerline_path(geo)
    if not os.path.exists(centerlines_input_file):
        return 'centerline does not exist'

    # write outlet boundary conditions to file if they exist
    bc_types, err = write_bc(fpath_1d, db, geo)
    if err:
        return err

    # get simulation constants
    constants = db.get_constants(geo)

    # reference pressure (= initial pressure?)
    pref = res_3d['pressure'][-1, 0]

    # copy cap surfaces to simulation folder
    for f in db.get_surfaces(geo, 'caps'):
        shutil.copy2(f, fpath_surf)

    # copy outlet names
    fpath_outlets = os.path.join(fpath_1d, 'outlets')
    shutil.copy(db.get_centerline_outlet_path(geo), fpath_outlets)

    # read inflow conditions
    flow = np.load(db.get_bc_flow_path(geo), allow_pickle=True).item()

    # read 3d boundary conditions
    bc_def, _ = db.get_bcs(geo)

    # extract inflow data
    time = flow['time']
    inflow = flow['velocity'][:, int(bc_def['preid']['inflow']) - 1]

    # insert last 3d time step as 1d initial condition (periodic solution)
    time = np.insert(time, 0, 0)
    inflow = np.insert(inflow, 0, inflow[-1])

    # save inflow file. sign reverse as compared to 3d simulation (inflow is positive)
    if geo == '0069_0001':
        inflow *= -1
    np.savetxt(os.path.join(fpath_1d, 'inflow.flow'), np.vstack((time, - inflow)).T)

    # set simulation time as end of 3d simulation
    save_data_freq = 1
    dt = 1e-3

    # run all cycles
    num_dts = int(flow['time'][-1] * n_cycle / dt + 1.0)

    try:
    # if True:
        oned.run(boundary_surfaces_directory=fpath_surf,
                 centerlines_input_file=centerlines_input_file,
                 centerlines_output_file=None,
                 compute_centerlines=False,
                 compute_mesh=True,
                 density=constants['density'],
                 element_size=element_size,
                 inlet_face_input_file='inflow.vtp',
                 inflow_input_file=os.path.join(fpath_1d, 'inflow.flow'),
                 linear_material_ehr=1e15,
                 linear_material_pressure=pref,
                 material_model=None,
                 mesh_output_file=geo + '.vtp',
                 min_num_elements=min_num_elems,
                 model_name=geo,
                 num_time_steps=num_dts,
                 olufsen_material_k1=None,
                 olufsen_material_k2=None,
                 olufsen_material_k3=None,
                 olufsen_material_exp=None,
                 olufsen_material_pressure=pref,
                 outflow_bc_input_file=fpath_1d,
                 outflow_bc_type=bc_types,
                 outlet_face_names_input_file=fpath_outlets,
                 output_directory=fpath_1d,
                 seg_min_num=seg_min_num,
                 seg_size=seg_size,
                 seg_size_adaptive=seg_size_adaptive,
                 solver_output_file=geo + '.inp',
                 save_data_frequency=save_data_freq,
                 surface_model=os.path.join(fpath_geo, 'all_exterior.vtp'),
                 time_step=dt,
                 uniform_bc=True,
                 units='cm',
                 viscosity=constants['viscosity'],
                 wall_properties_input_file=None,
                 wall_properties_output_file=None,
                 write_mesh_file=True,
                 write_solver_file=True)
    except Exception as e:
        return repr(e)

    return None


def main(db, geometries):
    # simvascular object
    sv = SimVascular()

    for geo in geometries:
        print('Running geometry ' + geo)

        # if os.path.exists(os.path.join(db.get_solve_dir_1d(geo), geo + '.vtp')):
        #     print('  skipping')
        #     continue

        # generate oneDSolver input file and check if successful
        msg = generate_1d(db, geo)

        # msg = None
        if not msg:
            # run oneDSolver
            sv.run_solver_1d(db.get_solve_dir_1d(geo), geo + '.inp')
            # with open(os.path.join(db.get_solve_dir_1d(geo), 'solver.log'), 'w+') as f:
            #     f.write(out_solve)

            # extract results
            res_dir = db.get_solve_dir_1d(geo)
            params_file = db.get_1d_params(geo)
            results_1d = read_results_1d(res_dir, params_file)

            # save results
            if results_1d['flow']:
                np.save(db.get_1d_flow_path(geo), results_1d)
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
