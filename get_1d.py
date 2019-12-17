#!/usr/bin/env python

import numpy as np
import sys
import os
import shutil
import glob
import csv
import re
import pdb
import contextlib, io

from get_database import Database, SimVascular, Post
from get_sim import write_bc

sys.path.append('/home/pfaller/work/repos/SimVascular/Python/site-packages/')

import sv_1d_simulation as oned


def read_results_1d(fpath_1d, geo):
    # requested output fields
    fields_res_1d = ['flow', 'pressure']

    # read 1D simulation results
    results_1d = {}
    for field in fields_res_1d:
        # list all output files for field
        result_list_1d = glob.glob(os.path.join(fpath_1d, geo + 'Group*Seg*_' + field + '.dat'))

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

    return results_1d


def generate_1d(db, geo):
    # number of cycles to run
    n_cycle = 10

    # sub-segment size
    seg_min_num = 1
    seg_size = 0.1

    # FEM size
    min_num_elems = 1
    element_size = 0.05

    # mesh adaptive?
    seg_size_adaptive = True

    # set simulation paths and create folders
    fpath_1d = db.get_solve_dir_1d(geo)
    fpath_geo = os.path.join(fpath_1d, 'geometry')
    fpath_surf = os.path.join(fpath_geo, 'surfaces')
    os.makedirs(fpath_geo, exist_ok=True)
    os.makedirs(fpath_surf, exist_ok=True)

    # write outlet boundary conditions to file if they exist
    fpath_outlet_bcs = os.path.join(fpath_1d, 'rcrt.dat')
    if not write_bc(fpath_outlet_bcs, db, geo):
        return False, 'non-existing boundary conditions'

    # copy surface model to folder if it exists
    if os.path.exists(db.get_surfaces(geo, 'all_exterior')):
        shutil.copy2(db.get_surfaces(geo, 'all_exterior'), fpath_geo)
    else:
        return False, 'non-existing 3d geometry'

    if os.path.exists(db.get_bc_flow_path(geo)):
        res_3d = db.read_results(geo, db.get_bc_flow_path(geo))
    else:
        return False, 'no 3d results'

    # reference pressure (= initial pressure?)
    pref = res_3d['pressure'][0, 0]

    # copy cap surfaces to simulation folder
    for f in db.get_surfaces(geo, 'caps'):
        shutil.copy2(f, fpath_surf)

    # compute centerlines only if they don't already exist
    centerlines_output_file = os.path.join(fpath_1d, 'centerlines.vtp')
    compute_centerlines = not os.path.exists(centerlines_output_file)
    if compute_centerlines:
        centerlines_input_file = None
    else:
        centerlines_input_file = centerlines_output_file

    # write outlet names to file
    fpath_outlets = os.path.join(fpath_1d, 'outlets')
    with open(fpath_outlets, 'w+') as f:
        for s in db.get_outlet_names(geo):
            f.write(s + '\n')

    # read inflow conditions
    flow = np.load(db.get_bc_flow_path(geo), allow_pickle=True).item()

    # read 3d boundary conditions
    bc_def, _ = db.get_bcs(geo)

    # extract inflow data
    time = flow['time']
    inflow = flow['velocity'][:, int(bc_def['preid']['inflow']) - 1]

    # repeat cycles (one more than n_cycle to guarantee inflow data in case of round-off errors with time steps)
    for i in range(n_cycle):
        time = np.append(time, flow['time'] + time[-1])
    inflow = np.tile(inflow, n_cycle + 1)

    # insert last 3d time step as 1d initial condition (periodic solution)
    time = np.insert(time, 0, 0)
    inflow = np.insert(inflow, 0, inflow[-1])

    # save inflow file. sign reverse as compared to 3d simulation (inflow is positive)
    np.savetxt(os.path.join(fpath_1d, 'inflow.flow'), np.vstack((time, - inflow)).T)

    # set simulation time as end of 3d simulation
    dt = 1e-3
    num_dts = int(flow['time'][-1] * n_cycle / dt + 1.0)
    save_data_freq = 1

    # if True:
    try:
        f_io = io.StringIO()
        # with contextlib.redirect_stdout(f_io):
        #     # use all options, set to None if using defaults (in cgs units)
        oned.run(boundary_surfaces_directory=fpath_surf,
                 centerlines_input_file=centerlines_input_file,
                 centerlines_output_file=centerlines_output_file,
                 compute_centerlines=compute_centerlines,
                 compute_mesh=True,
                 density=None,
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
                 outflow_bc_input_file=fpath_outlet_bcs,
                 outflow_bc_type='rcr',
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
                 units=None,
                 viscosity=None,
                 wall_properties_input_file=None,
                 wall_properties_output_file=None,
                 write_mesh_file=True,
                 write_solver_file=True)
    except (KeyError, ZeroDivisionError, RuntimeError) as e:
        # todo KeyError: log geometries with multiple inlets
        # todo ZeroDivisionError:
        #   0070_0001: Cannot read cell data array "GlobalElementID" from PointData in piece 0.  The data array in the element may be too short.
        # todo RuntimeError:
        #   0098_0001: Inlet group id is not 0 or number of centerlines is not equal to the number of outlets
        return False, repr(e)

    return True, f_io.getvalue()


def main():
    db = Database()
    sv = SimVascular()

    # for geo in ['0006_0001']:
    # for geo in ['0071_0001']:
    # for geo in ['0091_0001']:
    for geo in db.get_geometries():
        print('Running geometry ' + geo)

        # output path for 1d results
        fpath_out = os.path.join(db.fpath_gen, '1d_flow', geo)

        # if os.path.exists(fpath_out + '.npy'):
        #     print('  skipping (1d solution alread exists)')
        #     continue

        # generate oneDSolver input file and check if successful
        success_gen, out_gen = generate_1d(db, geo)
        with open(os.path.join(db.get_solve_dir_1d(geo), 'generate_1d.log'), 'w+') as f:
            f.write(out_gen)

        if success_gen:
            # run oneDSolver
            out_solve, success_solve = sv.run_solver_1d(db.get_solve_dir_1d(geo), geo + '.inp')
            with open(os.path.join(db.get_solve_dir_1d(geo), 'solver.log'), 'w+') as f:
                f.write(out_solve)

            # extract results
            results_1d = read_results_1d(db.get_solve_dir_1d(geo), geo)
            np.save(fpath_out, results_1d)
        else:
            print('  skipping (1d model creation failed)')


if __name__ == '__main__':
    main()
