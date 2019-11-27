#!/usr/bin/env python

import numpy as np
import sys
import os
import shutil
import pdb

from get_database import Database, SimVascular
from get_sim import write_bc

sys.path.append('/home/pfaller/work/repos/SimVascular/Python/site-packages/')

import sv_1d_simulation as oned

db = Database()
sv = SimVascular()

element_size = 0.1
min_num_elems = 10
time_step = 5e-5
num_time_steps = 2e3
save_data_freq = 20

# for geo in db.get_geometries():
for geo in ['0075_1001']:
# for geo in ['0110_0001']:
    # set simulation paths
    fpath_1d = db.get_solve_dir_1d(geo)
    fpath_geo = os.path.join(fpath_1d, 'geometry')
    fpath_surf = os.path.join(fpath_geo, 'surfaces')
    os.makedirs(fpath_geo, exist_ok=True)
    os.makedirs(fpath_surf, exist_ok=True)

    # copy geometry
    for f in db.get_surfaces(geo, 'caps'):
        shutil.copy2(f, fpath_surf)
    shutil.copy2(db.get_surfaces(geo, 'all_exterior'), fpath_geo)

    # write outlet names to file
    fpath_outlets = os.path.join(fpath_1d, 'outlets')
    outlets = db.get_surfaces(geo, 'outlets')
    outlets = [os.path.splitext(os.path.basename(s))[0] for s in outlets]
    outlets.sort()
    with open(fpath_outlets, 'w+') as f:
        for s in outlets:
            f.write(s + '\n')

    # write outlet boundary conditions to file
    fpath_oulet_bcs = os.path.join(fpath_1d, 'rcrt.dat')
    write_bc(fpath_oulet_bcs, db, geo)

    # pdb.set_trace()
    oned.run(boundary_surfaces_directory=fpath_surf,
             centerlines_input_file=None,
             centerlines_output_file=os.path.join(fpath_1d, 'centerlines.vtp'),
             compute_centerlines=True,
             compute_mesh=True,
             density=None,
             element_size=element_size,
             inlet_face_input_file='inflow.vtp',
             inflow_input_file=db.get_flow(geo),
             linear_material_ehr=None,
             linear_material_pressure=None,
             material_model=None,
             mesh_output_file='mesh.vtp',
             min_num_elements=min_num_elems,
             model_name=geo,
             num_time_steps=num_time_steps,
             olufsen_material_k1=None,
             olufsen_material_k2=None,
             olufsen_material_k3=None,
             olufsen_material_exp=None,
             olufsen_material_pressure=None,
             outflow_bc_input_file=fpath_oulet_bcs,
             outflow_bc_type='rcr',
             outlet_face_names_input_file=fpath_outlets,
             output_directory=fpath_1d,
             solver_output_file='solver.inp',
             save_data_frequency=save_data_freq,
             surface_model=os.path.join(fpath_geo, 'all_exterior.vtp'),
             time_step=time_step,
             uniform_bc=True,
             units=None,
             viscosity=None,
             wall_properties_input_file=None,
             wall_properties_output_file=None,
             write_mesh_file=True,
             write_solver_file=True)

    sv.run_solver_id(fpath_1d)
