#!/usr/bin/env python

import numpy as np
import os
import pdb


# create input file for svpre
def write_pre(fpath_solve, bc_def, geo):
    fname_pre = os.path.join(fpath_solve, 'sim.svpre')
    with open(fname_pre, 'w+') as f:

        # enter debug mode
        # f.write('verbose true\n')

        # write volume mesh
        f.write('mesh_and_adjncy_vtu ' + os.path.join('mesh-complete', geo + '_sim_results_in_cm.vtu') + '\n')

        # write surface mesh
        # f.write('set_surface_id_vtp ' + os.path.join('mesh-complete', 'all_exterior.vtp 1') + '\n')

        fpath_surf = os.path.join('mesh-complete', 'mesh-surfaces')

        # write surfaces (sort according to surface ID for readability)
        for k, v in sorted(bc_def['spid'].items(), key=lambda kv: kv[1]):
            if int(v) > 0:
                f_surf = os.path.join(fpath_surf, k + '.vtp')

                # check if mesh file exists
                f_surf_full = os.path.join(fpath_solve, f_surf)
                assert os.path.exists(f_surf_full), 'file ' + f_surf + ' does not exist'

                f.write('set_surface_id_vtp ' + f_surf + ' ' + repr(int(v)) + '\n')

        # write inlet bc
        f.write('prescribed_velocities_vtp ' + os.path.join(fpath_surf, 'inflow.vtp') + '\n')
        f.write('bct_analytical_shape ' + bc_def['bc']['inflow']['type'] + '\n')

        # write default parameters
        f.write('fluid_density 1.06\n')
        f.write('fluid_viscosity 0.04\n')

        # TODO: replace by pressure/velocity fields
        f.write('initial_pressure 0\n')
        f.write('initial_velocity 0.0001 0.0001 0.0001\n')

        # request outputs
        f.write('write_geombc geombc.dat.1\n')
        f.write('write_restart restart.0.1\n')
        f.write('bct_write_dat bct.dat\n')
        f.write('bct_write_vtp bct.vtp\n')

    # write start file
    fname_start = os.path.join(fpath_solve, 'numstart.dat')
    with open(fname_start, 'w+') as f:
        f.write('0')

    return fname_pre


def write_value(db, geo, bc, name):
    return str(db.get_in_model_units(geo, name[0], float(bc[name])))


def write_bc(fdir, db, geo):
    # get boundary conditions
    bc_def, _ = db.get_bcs(geo)

    # check if bc-file exists
    if not bc_def:
        return None, 'boundary conditions do not exist'

    # get outlet names
    outlets = db.get_outlet_names(geo)

    # get type of bcs
    bc_type, err = db.get_bc_type(geo)
    if err:
        return None, err

    # write bc-file
    fname = os.path.join(fdir, bc_type + '.dat')
    f = open(fname, 'w+')

    if bc_type == 'rcr':
        keyword = 'newbcface'
        f.write(keyword + '\n')

    # write boundary conditions
    for s in outlets:
        bc = bc_def['bc'][s]
        if bc_type == 'rcr':
            f.write(keyword + '\n')
            f.write(s + '\n')
            f.write(write_value(db, geo, bc, 'Rp') + '\n')
            f.write(write_value(db, geo, bc, 'C') + '\n')
            f.write(write_value(db, geo, bc, 'Rd') + '\n')
            # not sure what this does???
            f.write('0.0 0\n')
            f.write('1.0 0\n')
        elif bc_type == 'resistance':
            f.write(s + ' ')
            f.write(write_value(db, geo, bc, 'R') + ' ')
            f.write(write_value(db, geo, bc, 'Po') + '\n')
        else:
            return None, 'boundary conditions not implemented'

    f.close()

    return fname, False
