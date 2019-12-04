#!/usr/bin/env python

import numpy as np
import os


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


def write_value(bc, name, units):
    symbol = name[0]

    # no conversion for units cgs
    if units == 'cm':
        bc_str = float(bc[name])

    # convert cgm to cgs
    elif units == 'mm':
        if symbol == 'R':
            bc_str = float(bc[name]) * 1e4
        elif symbol == 'C':
            bc_str = float(bc[name]) * 1e-4
        else:
            raise ValueError('Unknown boundary condition symbol ' + name)
    else:
        raise ValueError('Unknown units ' + units)
    return str(bc_str)


# write boundary conditions
def write_bc(fname, db, geo):
    # get boundary conditions
    bc_def, units = db.get_bcs(geo)
    if not bc_def:
        return False

    # get outlet names
    outlets = db.get_outlet_names(geo)

    # write bc-file
    f = open(fname, 'w+')

    keyword = 'newbcface'
    # not sure if this is right???
    f.write(keyword + '\n')

    # write boundary conditions
    for s in outlets:
        bc = bc_def['bc'][s]
        if 'Rp' in bc and 'C' in bc and 'Rd' in bc:
            f.write(keyword + '\n')
            f.write(s + '\n')
            f.write(write_value(bc, 'Rp', units) + '\n')
            f.write(write_value(bc, 'C', units) + '\n')
            f.write(write_value(bc, 'Rd', units) + '\n')
        else:
            # todo: what's up with other boundary conditions?
            print('boundary condition not implemented')
            return False

        # not sure what this does???
        f.write('0.0 0\n')
        f.write('1.0 0\n')

    f.close()

    return True
