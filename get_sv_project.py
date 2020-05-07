#!/usr/bin/env python

import pdb
import os
import shutil
import glob
import matplotlib.cm as cm
from collections import OrderedDict, defaultdict

import numpy as np

from get_database import input_args, Database, SVProject, SimVascular
from get_bcs import get_in_model_units


def write_svproj_file(db, geo):
    t = str(db.svproj.t)
    proj_head = ['<?xml version="1.0" encoding="UTF-8"?>',
                 '<projectDescription version="1.0">']
    proj_end = ['</projectDescription>']

    with open(db.get_svproj_file(geo), 'w+') as f:
        # write header
        for s in proj_head:
            f.write(s + '\n')

        # write images/segmentations
        for k, s in db.svproj.dir.items():
            f.write(t + '<' + k + ' folder_name="' + s + '"')
            if k == 'images':
                img = os.path.basename(db.get_img(geo))
                f.write('>\n')
                f.write(t*2 + '<image name="' + os.path.splitext(img)[0] + '" in_project="yes" path="' + img + '"/>\n')
                f.write(t + '</' + k + '>\n')
            # elif k == 'segmentations':
            else:
                f.write('/>\n')

        # write end
        for s in proj_end:
            f.write(s + '\n')


def write_model(db, geo):
    t = str(db.svproj.t)
    model_head = ['<?xml version="1.0" encoding="UTF-8" ?>',
                  '<format version="1.0" />',
                  '<model type="PolyData">',
                  t + '<timestep id="0">',
                  t*2 + '<model_element type="PolyData" num_sampling="0">']
    model_end = [t*3 + '<blend_radii />',
                 t*3 + '<blend_param blend_iters="2" sub_blend_iters="3" cstr_smooth_iters="2" lap_smooth_iters="50" '
                       'subdivision_iters="1" decimation="0.01" />',
                 t*2 + '</model_element>',
                 t + '</timestep>',
                 '</model>']

    # read boundary conditions
    bc_def, _ = db.get_bcs(geo)
    bc_def['spid']['wall'] = 0

    # get cap names
    caps = db.get_surface_names(geo, 'caps')
    caps += ['wall']

    # sort caps according to face id
    ids = np.array([repr(int(float(bc_def['spid'][c]))) for c in caps])
    order = np.argsort(ids)
    caps = np.array(caps)[order]
    ids = ids[order]

    # display colors for caps
    colors = cm.jet(np.linspace(0, 1, len(caps)))

    # write model file
    with open(db.get_svproj_mdl_file(geo), 'w+') as f:
        # write header
        for s in model_head:
            f.write(s + '\n')

        # write faces
        f.write(t*3 + '<faces>\n')
        for i, c in enumerate(caps):
            c_str = t*4 + '<face id="' + ids[i] + '" name="' + c + '" type='
            if c == 'wall':
                c_str += '"wall"'
            else:
                c_str += '"cap"'
            for j in range(3):
                c_str += ' color' + repr(j + 1) + '="' + repr(colors[i, j]) + '"'
            f.write(c_str + ' visible="true" opacity="1" />\n')
        f.write(t*3 + '</faces>\n')

        # write end
        for s in model_end:
            f.write(s + '\n')


def write_path_segmentation(db, geo):
    # SimVascular instance
    sv = SimVascular()

    # get paths
    p = OrderedDict()
    p['f_path_in'] = db.get_path_file(geo)
    p['f_path_out'] = os.path.join(db.get_svproj_dir(geo), db.svproj.dir['paths'])

    seg_dir = db.get_seg_dir(geo)
    segments = glob.glob(os.path.join(seg_dir, '*'))

    err_seg = ''
    for s in segments:
        p['f_seg_in'] = s
        p['f_seg_out'] = os.path.join(db.get_svproj_dir(geo), db.svproj.dir['segmentations'])

        if '.tcl' in s:
            continue

        # assemble call string
        sv_string = [os.path.join(os.getcwd(), 'sv_get_path_segmentation.py')]
        for v in p.values():
            sv_string += [v]

        err = sv.run_python_legacyio(sv_string)[1]
        if err:
            err_seg += os.path.basename(s).split('.')[0] + '\n'

    # execute SimVascular-Python
    return err_seg


def write_inflow(db, geo, model):
    # read inflow conditions
    time, inflow = db.get_inflow(geo)

    # reverse flow for this geometry (wrong surface normals, too lazy to fix)
    if geo == '0069_0001':
        inflow *= -1

    # reverse flow for svOneDSolver
    if model == '1d':
        inflow *= -1

    # save inflow file
    np.savetxt(db.get_sv_flow_path(geo, model), np.vstack((time, inflow)).T)

    return len(inflow), time[-1]


def write_pre(db, geo):
    """
    Create input file for svpre
    """
    # get boundary conditions
    bc_def, params = db.get_bcs(geo)

    # read inflow conditions
    time, inflow = db.get_inflow(geo)

    with open(db.get_svpre_file(geo), 'w+') as f:

        # enter debug mode
        # f.write('verbose true\n')

        # write volume mesh
        f.write('mesh_and_adjncy_vtu ' + os.path.join('mesh-complete', geo + '_sim_results_in_cm.vtu') + '\n\n')

        # write surface mesh
        # f.write('set_surface_id_vtp ' + os.path.join('mesh-complete', 'all_exterior.vtp 1') + '\n')

        fpath_surf = os.path.join('mesh-complete', 'mesh-surfaces')

        # write surfaces (sort according to surface ID for readability)
        f.write('set_surface_id_vtp ' + os.path.join(fpath_surf, 'wall.vtp') + ' 0\n')
        for k, v in sorted(bc_def['spid'].items(), key=lambda kv: kv[1]):
            if int(v) > 0:
                f_surf = os.path.join(fpath_surf, k + '.vtp')

                # check if mesh file exists
                f_surf_full = os.path.join(db.get_solve_dir_3d(geo), f_surf)
                # assert os.path.exists(f_surf_full), 'file ' + f_surf + ' does not exist'

                f.write('set_surface_id_vtp ' + f_surf + ' ' + repr(int(v)) + '\n')
        f.write('\n')

        # write inlet bc
        f_inflow = os.path.join(fpath_surf, 'inflow.vtp')
        f.write('prescribed_velocities_vtp ' + f_inflow + '\n\n')

        # generate inflow
        f.write('bct_analytical_shape ' + bc_def['bc']['inflow']['type'] + '\n')
        f.write('bct_period ' + str(time[-1]) + '\n')
        f.write('bct_point_number ' + str(len(inflow)) + '\n')
        f.write('bct_fourier_mode_number 10\n')
        f.write('bct_create ' + f_inflow + ' ' + db.get_sv_flow_path(geo, '3d') + '\n')
        f.write('bct_write_dat bct.dat\n')
        f.write('bct_write_vtp bct.vtp\n\n')

        # write default parameters
        # todo: get from tcl
        f.write('fluid_density 1.06\n')
        f.write('fluid_viscosity 0.04\n\n')

        # no slip boundary condition
        f.write('noslip_vtp mesh-complete/mesh-surfaces/wall.vtp\n\n')

        # reference pressure
        for cap, bc in bc_def['bc'].items():
            if cap == 'inflow' or cap == 'wall':
                continue
            if 'Po' in bc:
                p = str(bc['Po'])
            else:
                p = '0.0'
            f.write('pressure_vtp ' + os.path.join(fpath_surf, cap + '.vtp') + ' ' + p + '\n')
        f.write('\n')

        # todo: replace by pressure/velocity fields
        # f.write('initial_pressure 0\n')
        # f.write('initial_velocity 0.0001 0.0001 0.0001\n\n')

        # set OSMSC results as initial condition
        f.write('read_pressure_velocity_vtu ' + db.get_volume(geo) + '\n\n')

        # request outputs
        f.write('write_geombc geombc.dat.1\n')
        f.write('write_restart restart.0.1\n')
        f.write('write_numstart 0\n\n')

    # write start file
    fname_start = os.path.join(db.get_solve_dir_3d(geo), 'numstart.dat')
    with open(fname_start, 'w+') as f:
        f.write('0')


def write_solver(db, geo):
    # get boundary conditions
    bc_def, params = db.get_bcs(geo)
    bc_type, err = db.get_bc_type(geo)

    # read inflow conditions
    time, inflow = db.get_inflow(geo)

    # time step
    dt = 4.0e-4

    # number of cycles
    n_cycle = 3

    with open(db.get_solver_file(geo), 'w+') as f:
        # write default parameters
        # todo: get from tcl
        f.write('Density: 1.06\n')
        f.write('Viscosity: 0.04\n\n')

        # time step
        f.write('Number of Timesteps: ' + str(int(n_cycle * time[-1] / dt)) + '\n')
        f.write('Time Step Size: ' + str(dt) + '\n\n')

        # output
        f.write('Number of Timesteps between Restarts: 10\n')
        f.write('Number of Force Surfaces: 1\n')
        f.write('Surface ID\'s for Force Calculation: 0\n')
        f.write('Force Calculation Method: Velocity Based\n')
        f.write('Print Average Solution: True\n')
        f.write('Print Error Indicators: True\n\n')

        f.write('Time Varying Boundary Conditions From File: True\n\n')

        f.write('Step Construction: 0 1 0 1\n\n')

        # collect faces for each boundary condition type
        bc_ids = defaultdict(list)
        for cap, bc in bc_type.items():
            bc_ids[bc] += [int(bc_def['spid'][cap])]

        names = {'rcr': 'RCR', 'resistance': 'Resistance', 'coronary': 'Coronary'}
        for t, v in bc_ids.items():
            f.write('Number of ' + names[t] + ' Surfaces: ' + str(len(v)) + '\n')
            f.write('List of ' + names[t] + ' Surfaces: ' + str(v).replace(',', '')[1:-1] + '\n')

            if t == 'rcr':
                f.write('RCR Values From File: True\n\n')
            elif t == 'resistance':
                f.write('Resistance Values: ')
                for cap, bc in bc_type.items():
                    if bc == 'resistance':
                        f.write(str(bc_def['bc'][cap]['R']) + ' ')
                f.write('\n\n')
            elif t == 'coronary':
                raise ValueError('Coronary BCs not implemented')

        f.write('Pressure Coupling: Implicit\n')
        f.write('Number of Coupled Surfaces: ' + str(len(bc_def['bc']) - 2) + '\n\n')

        f.write('Backflow Stabilization Coefficient: 0.2\n')

        # nonlinear solver
        f.write('Residual Control: True\n')
        f.write('Residual Criteria: 0.01\n')
        f.write('Minimum Required Iterations: 3\n')

        # linear solver
        f.write('svLS Type: NS\n')
        f.write('Number of Krylov Vectors per GMRES Sweep: 100\n')
        f.write('Number of Solves per Left-hand-side Formation: 1\n')

        f.write('Tolerance on Momentum Equations: 0.01\n')
        f.write('Tolerance on Continuity Equations: 0.01\n')
        f.write('Tolerance on svLS NS Solver: 0.01\n')

        f.write('Maximum Number of Iterations for svLS NS Solver: 5\n')
        f.write('Maximum Number of Iterations for svLS Momentum Loop: 2\n')
        f.write('Maximum Number of Iterations for svLS Continuity Loop: 400\n')

        f.write('Time Integration Rule: Second Order\n')
        f.write('Time Integration Rho Infinity: 0.5\n')

        f.write('Flow Advection Form: Convective\n')

        f.write('Quadrature Rule on Interior: 2\n')
        f.write('Quadrature Rule on Boundary: 3\n')


def copy_files(db, geo):
    # define paths
    fpath_surf = os.path.join(db.get_solve_dir_3d(geo), 'mesh-complete', 'mesh-surfaces')

    # create simulation folder
    os.makedirs(fpath_surf, exist_ok=True)

    # copy geometry
    for f in glob.glob(os.path.join(db.fpath_gen, 'surfaces', geo, '*.vtp')):
        shutil.copy(f, fpath_surf)

    # copy volume mesh
    # todo: copy without results
    fpath_res = os.path.join(db.fpath_sim, geo, 'results', geo + '_sim_results_in_cm.vtu')
    shutil.copy(fpath_res, os.path.join(db.get_solve_dir_3d(geo), 'mesh-complete'))

    # copy inflow
    # inflow_src = os.path.join(db.fpath_gen, 'surfaces', geo, 'inflow.vtp')
    # inflow_trg = os.path.join(db.get_solve_dir_3d(geo), 'bct.vtp')
    # shutil.copy(inflow_src, inflow_trg)


def write_value(params, geo, bc, name):
    return str(get_in_model_units(params['sim_units'], name[0], float(bc[name])))


def write_bc(fdir, db, geo, write_face=True):
    # get boundary conditions
    bc_def, params = db.get_bcs(geo)

    # check if bc-file exists
    if not bc_def:
        return None, 'boundary conditions do not exist'

    # get type of bcs
    bc_type, err = db.get_bc_type(geo)
    if err:
        return None, err

    # get outlet names
    outlets = db.get_outlet_names(geo)

    # names expected by SimVascular for different boundary conditions
    bc_file_names = {'rcr': 'rcrt.dat', 'resistance': 'resistance.dat', 'coronary': 'cort.dat'}

    # keyword to indicate a new boundary condition
    keyword = '2'

    # create bc-files for every bc type
    u_bc_types = list(set(bc_type.values()))
    files = {}
    fnames = []
    for t in u_bc_types:
        if t in bc_file_names:
            fname = os.path.join(fdir, bc_file_names[t])
            files[t] = open(fname, 'w+')
            fnames += [fname]

            # write keyword for new faces in first line
            if t == 'rcr' or t == 'coronary':
                files[t].write(keyword + '\n')
        else:
            return None, 'boundary condition not implemented (' + t + ')'

    # write boundary conditions
    for s in outlets:
        bc = bc_def['bc'][s]
        t = bc_type[s]
        f = files[t]
        if t == 'rcr':
            f.write(keyword + '\n')
            if write_face:
                f.write(s + '\n')
            f.write(write_value(params, geo, bc, 'Rp') + '\n')
            f.write(write_value(params, geo, bc, 'C') + '\n')
            f.write(write_value(params, geo, bc, 'Rd') + '\n')
            if 'Po' in bc:
                if bc['Po'] != 0.0:
                    return None, 'RCR with Po unequal zero'
            # not sure what this does???
            f.write('0.0 0\n')
            f.write('1.0 0\n')
        elif t == 'resistance':
            f.write(s + ' ')
            f.write(write_value(params, geo, bc, 'R') + ' ')
            f.write(write_value(params, geo, bc, 'Po') + '\n')
        elif t == 'coronary':
            f.write(keyword + '\n')
            f.write(s + '\n')
            f.write(write_value(params, geo, bc, 'q0') + '\n')
            f.write(write_value(params, geo, bc, 'q1') + '\n')
            f.write(write_value(params, geo, bc, 'q2') + '\n')
            f.write(write_value(params, geo, bc, 'p0') + '\n')
            f.write(write_value(params, geo, bc, 'p1') + '\n')
            f.write(write_value(params, geo, bc, 'p2') + '\n')
            f.write(write_value(params, geo, bc, 'b0') + '\n')
            f.write(write_value(params, geo, bc, 'b1') + '\n')
            f.write(write_value(params, geo, bc, 'b2') + '\n')
            f.write(write_value(params, geo, bc, 'dQinidT') + '\n')
            f.write(write_value(params, geo, bc, 'dPinidT') + '\n')

            print(bc)
            # write time and pressure pairs
            for m in bc_def['coronary'][bc['Pim']]:
                f.write(str(m[0]) + ' ' + str(get_in_model_units(params['sim_units'], 'P', m[1])) + '\n')

    # close all opened files
    for t in u_bc_types:
        files[t].close()

    return fnames, False


def copy_file(db, geo, src, trg_dir):
    trg = os.path.join(db.get_svproj_dir(geo), db.svproj.dir[trg_dir], os.path.basename(src))
    shutil.copy2(src, trg)


def make_folders(db, geo):
    # make all project sub-folders
    for s in db.svproj.dir.values():
        os.makedirs(os.path.join(db.get_svproj_dir(geo), s), exist_ok=True)

    # copy image
    copy_file(db, geo, db.get_img(geo), 'images')

    # copy volume mesh
    copy_file(db, geo, db.get_volume_mesh(geo), 'meshes')

    # copy surface mesh
    copy_file(db, geo, db.get_sv_surface(geo), 'meshes')
    copy_file(db, geo, db.get_sv_surface(geo), 'models')

    return True


def check_files(db, geo):
    # check if files exist
    if db.get_volume_mesh(geo) is None:
        return False, 'no volume mesh'
    if db.get_sv_surface(geo) is None:
        return False, 'no SV surface mesh'
    if db.get_img(geo) is None:
        return False, 'no medical image'
    return True, None


def main(db, geometries):
    for geo in geometries:
        print('Running geometry ' + geo)

        success, err = check_files(db, geo)
        if not success:
            print('  ' + err)
            continue

        if True:
        # try:
            make_folders(db, geo)
            write_svproj_file(db, geo)
            write_model(db, geo)

            copy_files(db, geo)
            write_inflow(db, geo, '3d')
            write_pre(db, geo)
            write_solver(db, geo)
            write_bc(os.path.join(db.get_solve_dir_3d(geo)), db, geo, False)

            sv = SimVascular()
            sv.run_pre(db.get_solve_dir_3d(geo), db.get_svpre_file(geo))
            # sv.run_solver(db.get_solve_dir_3d(geo), 'solver.inp')
        # except Exception:
        #     print('  failed')
        #     continue

        err = write_path_segmentation(db, geo)
        if err:
            print('  \nmissing paths:\n' + err)
        else:
            print('  success!')


if __name__ == '__main__':
    descr = 'Generate an svproject folder'
    d, g, _ = input_args(descr)
    main(d, g)
