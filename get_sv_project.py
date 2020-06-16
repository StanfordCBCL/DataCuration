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
from inflow import optimize_inflow


def get_sv_opt(db, geo):
    # get boundary conditions
    bc_def, params = db.get_bcs(geo)
    bc_type, err = db.get_bc_type(geo)

    # number of cycles
    n_cycle = 10

    # number of time steps
    numstep = int(float(params['sim_steps_per_cycle']))

    # time step
    dt = db.get_3d_timestep(geo)

    # time increment
    nt_out = db.get_3d_increment(geo)

    # read inflow conditions
    time, inflow = db.get_inflow_smooth(geo)

    inflow_str = ''
    for j, (t, i) in enumerate(zip(time, inflow)):
        inflow_str += str(t) + ' ' + str(i)
        if j < time.shape[0] - 1:
            inflow_str += '&#x0A;'

    # number of points
    n_point = time.shape[0] // 2

    # number of fourier modes
    n_fourier = n_point - 1

    # set svsolver options
    opt = {'density': '1.06',
           'viscosity': '0.04',
           'backflow': '0.2',
           'advection': 'Convective',
           'inflow': 'inflow.flow',
           'inflow_str': inflow_str,
           'fourier_modes': str(n_fourier),
           'fourier_period': str(time[-1]),
           'fourier_points': str(n_point),
           'max_iter_continuity': '400',
           'max_iter_momentum': '10',
           'max_iter_ns_solver': '10',
           'min_iter': '3',
           'num_krylov': '300',
           'num_solve': '1',
           'num_time': str(int(n_cycle * numstep + 100)),
           'num_restart': str(nt_out),
           'bool_surf_stress': 'True',
           'coupling': 'Implicit',
           'print_avg_sol': 'True',
           'print_err': 'False',
           'quad_boundary': '3',
           'quad_interior': '2',
           'residual_control': 'True',
           'residual_criteria': '0.01',
           'step_construction': '5',
           'time_int_rho': '0.5',
           'time_int_rule': 'Second Order',
           'time_step': str(dt),
           'tol_continuity': '0.01',
           'tol_momentum': '0.01',
           'tol_ns_solver': '0.01',
           'svls_type': 'NS',
           'mesh_initial': os.path.join('mesh-complete', 'initial.vtu'),
           'mesh_vtu': os.path.join('mesh-complete', 'mesh-complete.mesh.vtu'),
           'mesh_vtp': os.path.join('mesh-complete', 'mesh-complete.exterior.vtp'),
           'mesh_inflow': os.path.join('mesh-complete', 'mesh-surfaces', 'inflow.vtp'),
           'mesh_walls': os.path.join('mesh-complete', 'walls_combined.vtp')}
    return opt


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
                f.write(
                    t * 2 + '<image name="' + os.path.splitext(img)[0] + '" in_project="yes" path="' + img + '"/>\n')
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
                  t * 2 + '<model_element type="PolyData" num_sampling="0">']
    model_end = [t * 3 + '<blend_radii />',
                 t * 3 + '<blend_param blend_iters="2" sub_blend_iters="3" cstr_smooth_iters="2" lap_smooth_iters="50" '
                         'subdivision_iters="1" decimation="0.01" />',
                 t * 2 + '</model_element>',
                 t + '</timestep>',
                 '</model>']

    # read boundary conditions
    bc_def, _ = db.get_bcs(geo)
    bc_def['preid']['wall'] = 0

    # get cap names
    caps = db.get_surface_names(geo, 'caps')
    caps += ['wall']

    # sort caps according to face id
    ids = np.array([repr(int(float(bc_def['preid'][c] + 1))) for c in caps])
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
        f.write(t * 3 + '<faces>\n')
        for i, c in enumerate(caps):
            c_str = t * 4 + '<face id="' + ids[i] + '" name="' + c + '" type='
            if c == 'wall':
                c_str += '"wall"'
            else:
                c_str += '"cap"'
            for j in range(3):
                c_str += ' color' + repr(j + 1) + '="' + repr(colors[i, j]) + '"'
            f.write(c_str + ' visible="true" opacity="1" />\n')
        f.write(t * 3 + '</faces>\n')

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


def write_mesh(db, geo):
    t = str(db.svproj.t)
    mesh_generic = ['<?xml version="1.0" encoding="UTF-8" ?>',
                    '<format version="1.0" />',
                    '<mitk_mesh type="TetGen" model_name="' + geo + '">',
                    t + '<timestep id="0">',
                    t * 2 + '<mesh type="TetGen">',
                    t * 3 + '<command_history>',
                    t * 4 + '<command content="option surface 1" />',
                    t * 4 + '<command content="option volume 1" />',
                    t * 4 + '<command content="option UseMMG 1" />',
                    t * 4 + '<command content="setWalls" />',
                    # t * 4 + '<command content="option Optimization 3" />',
                    # t * 4 + '<command content="option QualityRatio 1.4" />',
                    t * 4 + '<command content="option NoBisect" />',
                    # t * 4 + '<command content="AllowMultipleRegions 0" />',
                    t * 4 + '<command content="generateMesh" />',
                    t * 4 + '<command content="writeMesh" />',
                    t * 3 + '</command_history>',
                    t * 2 + '</mesh>',
                    t + '</timestep>',
                    '</mitk_mesh>']

    fname = os.path.join(db.get_svproj_dir(geo), db.svproj.dir['meshes'], geo + '.msh')

    # write generic mesh file
    with open(fname, 'w+') as f:
        for s in mesh_generic:
            f.write(s + '\n')


def write_inflow(db, geo, model, n_mode=10, n_sample_real=256):
    # read inflow conditions
    time, inflow = db.get_inflow(geo)

    # smooth inflow
    time, inflow = optimize_inflow(time, inflow, n_mode=n_mode, n_sample_real=n_sample_real)

    # reverse flow for svOneDSolver
    if model == '1d':
        inflow *= -1

    # save inflow file
    fpath = db.get_sv_flow_path(geo, model)
    os.makedirs(os.path.dirname(fpath), exist_ok=True)

    np.savetxt(fpath, np.vstack((time, inflow)).T)
    # np.savetxt(fpath, np.vstack(([0, 1], [inflow[-1], inflow[-1]])).T)
    # np.savetxt(fpath, np.vstack(([0, 1], [0, 0])).T)

    return len(inflow), time[-1]


def write_inflow_const(db, geo):
    fpath = db.get_sv_flow_path(geo, '3d_constant')
    np.savetxt(fpath, np.vstack(([0, 1], [1, 1])).T)


def write_pre(db, geo, solver='svsolver'):
    """
    Create input file for svpre
    """
    # get boundary conditions
    bc_def, params = db.get_bcs(geo)

    # read inflow conditions
    time, _ = db.get_inflow_smooth(geo)

    # outlet names
    outlets = db.get_outlet_names(geo)

    # get solver options
    opt = get_sv_opt(db, geo)

    with open(db.get_svpre_file(geo, solver), 'w+') as f:
        # enter debug mode
        # f.write('verbose true\n')

        # write volume mesh
        f.write('mesh_and_adjncy_vtu ' + opt['mesh_vtu'] + '\n')

        # write surface mesh
        fpath_surf = os.path.join('mesh-complete', 'mesh-surfaces')

        # write surfaces (sort according to surface ID for readability)
        f.write('set_surface_id_vtp ' + opt['mesh_vtp'] + ' 0\n')
        f.write('set_surface_id_vtp ' + opt['mesh_inflow'] + ' 1\n')
        for k in outlets:
            v = bc_def['preid'][k] + 1
            if int(v) > 1:
                f_surf = os.path.join(fpath_surf, k + '.vtp')

                # check if mesh file exists
                f_surf_full = os.path.join(db.get_solve_dir_3d(geo), f_surf)
                assert os.path.exists(f_surf_full), 'file ' + f_surf + ' does not exist'

                f.write('set_surface_id_vtp ' + f_surf + ' ' + repr(int(v)) + '\n')
        f.write('\n')

        if solver == 'perigee':
            return

        # write inlet bc
        f.write('prescribed_velocities_vtp ' + opt['mesh_inflow'] + '\n\n')

        # generate inflow
        f.write('bct_analytical_shape ' + bc_def['bc']['inflow']['type'] + '\n')
        f.write('bct_period ' + opt['fourier_period'] + '\n')
        f.write('bct_point_number ' + opt['fourier_points'] + '\n')
        f.write('bct_fourier_mode_number ' + opt['fourier_modes'] + '\n')
        # f.write('bct_create ' + opt['mesh_inflow'] + ' ' + db.get_sv_flow_path_rel(geo, '3d_constant') + '\n')
        f.write('bct_create ' + opt['mesh_inflow'] + ' ' + opt['inflow'] + '\n')
        f.write('bct_write_dat bct.dat\n')
        f.write('bct_write_vtp bct.vtp\n\n')

        # write default parameters
        f.write('fluid_density ' + opt['density'] + '\n')
        f.write('fluid_viscosity ' + opt['viscosity'] + '\n\n')

        # no slip boundary condition
        f.write('noslip_vtp ' + opt['mesh_walls'] + '\n\n')

        # reference pressure
        for cap in outlets:
            bc = bc_def['bc'][cap]
            if cap == 'inflow' or cap == 'wall':
                continue
            if 'Po' in bc:
                f.write('pressure_vtp ' + os.path.join(fpath_surf, cap + '.vtp') + ' ' + str(bc['Po']) + '\n')
            else:
                f.write('zero_pressure_vtp ' + os.path.join(fpath_surf, cap + '.vtp') + '\n')
        f.write('\n')

        # set OSMSC results as initial condition
        f.write('read_pressure_velocity_vtu ' + opt['mesh_initial'] + '\n\n')
        # f.write('initial_pressure 0\n')
        # f.write('initial_velocity 0.0001 0.0001 0.0001\n\n')

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
    bc_def, _ = db.get_bcs(geo)
    bc_type, err = db.get_bc_type(geo)

    # ordered outlets
    outlets = db.get_outlet_names(geo)

    # get solver options
    opt = get_sv_opt(db, geo)

    with open(db.get_solver_file(geo), 'w+') as f:
        # write default parameters
        # todo: get from tcl
        f.write('Density: ' + opt['density'] + '\n')
        f.write('Viscosity: ' + opt['viscosity'] + '\n\n')

        # time step
        f.write('Number of Timesteps: ' + opt['num_time'] + '\n')
        f.write('Time Step Size: ' + opt['time_step'] + '\n\n')

        # output
        f.write('Number of Timesteps between Restarts: ' + opt['num_restart'] + '\n')
        f.write('Number of Force Surfaces: 1\n')
        f.write('Surface ID\'s for Force Calculation: 0\n')
        f.write('Force Calculation Method: Velocity Based\n')
        f.write('Print Average Solution: ' + opt['print_avg_sol'] + '\n')
        f.write('Print Error Indicators: ' + opt['print_err'] + '\n\n')

        f.write('Time Varying Boundary Conditions From File: True\n\n')

        f.write('Step Construction:')
        for i in range(int(opt['step_construction'])):
            f.write(' 0 1')
        f.write('\n\n')

        # collect faces for each boundary condition type
        bc_ids = defaultdict(list)
        for cap in outlets:
            bc_ids[bc_type[cap]] += [int(bc_def['preid'][cap]) + 1]

        # boundary conditions
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
            else:
                raise ValueError('Boundary condition ' + t + ' unknown')

        f.write('Pressure Coupling: ' + opt['coupling'] + '\n')
        f.write('Number of Coupled Surfaces: ' + str(len(bc_def['bc']) - 2) + '\n\n')

        f.write('Backflow Stabilization Coefficient: ' + opt['backflow'] + '\n')

        # nonlinear solver
        f.write('Residual Control: ' + opt['residual_control'] + '\n')
        f.write('Residual Criteria: ' + opt['residual_criteria'] + '\n')
        f.write('Minimum Required Iterations: ' + opt['min_iter'] + '\n')

        # linear solver
        f.write('svLS Type: ' + opt['svls_type'] + '\n')
        f.write('Number of Krylov Vectors per GMRES Sweep: ' + opt['num_krylov'] + '\n')
        f.write('Number of Solves per Left-hand-side Formation: ' + opt['num_solve'] + '\n')

        f.write('Tolerance on Momentum Equations: ' + opt['tol_momentum'] + '\n')
        f.write('Tolerance on Continuity Equations: ' + opt['tol_continuity'] + '\n')
        f.write('Tolerance on svLS NS Solver: ' + opt['tol_ns_solver'] + '\n')

        f.write('Maximum Number of Iterations for svLS NS Solver: ' + opt['max_iter_ns_solver'] + '\n')
        f.write('Maximum Number of Iterations for svLS Momentum Loop: ' + opt['max_iter_momentum'] + '\n')
        f.write('Maximum Number of Iterations for svLS Continuity Loop: ' + opt['max_iter_continuity'] + '\n')

        # time integration
        f.write('Time Integration Rule: ' + opt['time_int_rule'] + '\n')
        f.write('Time Integration Rho Infinity: ' + opt['time_int_rho'] + '\n')

        f.write('Flow Advection Form: ' + opt['advection'] + '\n')

        f.write('Quadrature Rule on Interior: ' + opt['quad_interior'] + '\n')
        f.write('Quadrature Rule on Boundary: ' + opt['quad_boundary'] + '\n')


def print_props(f, props, t):
    for h in props:
        f.write(t + '<prop key="' + h[0] + '" value="' + h[1] + '" />\n')


def write_simulation(db, geo):
    # get boundary conditions
    bc_def, params = db.get_bcs(geo)
    bc_type, err = db.get_bc_type(geo)

    # get outlet names
    outlets = db.get_outlet_names(geo)

    # get solver options
    opt = get_sv_opt(db, geo)

    # tab
    t = str(db.svproj.t)

    sim_header = ['<?xml version="1.0" encoding="UTF-8" ?>',
                  '<format version="1.0" />',
                  '<mitk_job model_name="' + geo + '" mesh_name="' + geo + '" status="Simulation failed">',
                  t + '<job>']

    basic_props = [['Fluid Density', opt['density']],
                   ['Fluid Viscosity', opt['viscosity']],
                   ['IC File', opt['mesh_initial']],
                   ['Initial Pressure', '0'],
                   ['Initial Velocities', '0.0001 0.0001 0.0001']]

    inflow_props = [['Analytic Shape', bc_def['bc']['inflow']['type']],
                    ['BC Type', 'Prescribed Velocities'],
                    ['Flip Normal', 'False'],
                    ['Flow Rate', opt['inflow_str']],
                    ['Fourier Modes', opt['fourier_modes']],
                    ['Original File', 'inflow.flow'],
                    ['Period', opt['fourier_period']],
                    ['Point Number', opt['fourier_points']]]

    wall_props = [['Type', 'rigid']]

    solver_props = [['Backflow Stabilization Coefficient', opt['backflow']],
                    ['Flow Advection Form', opt['advection']],
                    ['Force Calculation Method', 'Velocity Based'],
                    ['Maximum Number of Iterations for svLS Continuity Loop', opt['max_iter_continuity']],
                    ['Maximum Number of Iterations for svLS Momentum Loop', opt['max_iter_momentum']],
                    ['Maximum Number of Iterations for svLS NS Solver', opt['max_iter_ns_solver']],
                    ['Minimum Required Iterations', opt['min_iter']],
                    ['Number of Krylov Vectors per GMRES Sweep', opt['num_krylov']],
                    ['Number of Solves per Left-hand-side Formation', opt['num_solve']],
                    ['Number of Timesteps', opt['num_time']],
                    ['Number of Timesteps between Restarts', opt['num_restart']],
                    ['Output Surface Stress', opt['bool_surf_stress']],
                    ['Pressure Coupling', opt['coupling']],
                    ['Print Average Solution', opt['print_avg_sol']],
                    ['Print Error Indicators', opt['print_err']],
                    ['Quadrature Rule on Boundary', opt['quad_boundary']],
                    ['Quadrature Rule on Interior', opt['quad_interior']],
                    ['Residual Control', opt['residual_control']],
                    ['Residual Criteria', opt['residual_criteria']],
                    ['Step Construction', opt['step_construction']],
                    ['Time Integration Rho Infinity', opt['time_int_rho']],
                    ['Time Integration Rule', opt['time_int_rule']],
                    ['Time Step Size', opt['time_step']],
                    ['Tolerance on Continuity Equations', opt['tol_continuity']],
                    ['Tolerance on Momentum Equations', opt['tol_momentum']],
                    ['Tolerance on svLS NS Solver', opt['tol_ns_solver']],
                    ['svLS Type', opt['svls_type']]]

    run_props = [['Number of Processes', '8']]

    with open(db.get_svproj_sjb_file(geo), 'w+') as f:
        for h in sim_header:
            f.write(h + '\n')

        f.write(t * 2 + '<basic_props>\n')
        print_props(f, basic_props, t * 3)
        f.write(t * 2 + '</basic_props>\n')

        # bcs
        f.write(t * 2 + '<cap_props>\n')

        # outflow
        for k in outlets:
            f.write(t * 3 + '<cap name="' + k + '">\n')

            tp = bc_type[k]
            bc = bc_def['bc'][k]

            if tp == 'rcr':
                rcr = write_value(params, geo, bc, 'Rp') + ' ' + \
                      write_value(params, geo, bc, 'C') + ' ' + \
                      write_value(params, geo, bc, 'Rd')

                f.write(t * 4 + '<prop key="BC Type" value="RCR" />\n')
                f.write(t * 4 + '<prop key="C Values" value="" />\n')
                if 'Po' in bc:
                    f.write(t * 4 + '<prop key="Pressure" value="' + write_value(params, geo, bc, 'Po') + '" />\n')
                else:
                    f.write(t * 4 + '<prop key="Pressure" value="0" />\n')
                f.write(t * 4 + '<prop key="R Values" value="" />\n')
                f.write(t * 4 + '<prop key="Values" value="' + rcr + '" />\n')
            elif tp == 'resistance':
                f.write(t * 4 + '<prop key="BC Type" value="Resistance" />\n')
                if 'Po' in bc:
                    f.write(t * 4 + '<prop key="Pressure" value="' + write_value(params, geo, bc, 'Po') + '" />\n')
                else:
                    f.write(t * 4 + '<prop key="Pressure" value="0" />\n')
                f.write(t * 4 + '<prop key="Values" value="' + write_value(params, geo, bc, 'R') + '" />\n')
            elif tp == 'coronary':
                raise ValueError('Coronary BCs not implemented')
            else:
                raise ValueError('Boundary condition ' + tp + ' unknown')

            f.write(t * 3 + '</cap>\n')

        # inflow
        f.write(t * 3 + '<cap name="inflow">\n')
        print_props(f, inflow_props, t * 4)
        f.write(t * 3 + '</cap>\n')

        f.write(t * 2 + '</cap_props>\n')

        # wall
        f.write(t * 2 + '<wall_props>\n')
        print_props(f, wall_props, t * 3)
        f.write(t * 2 + '</wall_props>\n')

        # various
        f.write(t * 2 + '<var_props />\n')

        # solver
        f.write(t * 2 + '<solver_props>\n')
        print_props(f, solver_props, t * 3)
        f.write(t * 2 + '</solver_props>\n')

        # run
        f.write(t * 2 + '<run_props>\n')
        print_props(f, run_props, t * 3)
        f.write(t * 2 + '</run_props>\n')

        # close
        f.write(t + '</job>\n')
        f.write('</mitk_job>')


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

    # names expected by svsolver for different boundary conditions
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
            if 'Po' in bc and bc['Po'] != 0.0:
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


def copy_files(db, geo):
    # get solver options
    opt = get_sv_opt(db, geo)
    
    # define paths
    sim_dir = db.get_solve_dir_3d(geo)
    fpath_surf = os.path.join(sim_dir, 'mesh-complete', 'mesh-surfaces')

    # create simulation folder
    os.makedirs(fpath_surf, exist_ok=True)

    # copy inflow
    shutil.copy(db.get_sv_flow_path(geo, '3d'), os.path.join(sim_dir, 'inflow.flow'))

    # copy cap meshes
    for f in glob.glob(os.path.join(db.get_sv_meshes(geo), 'caps', '*.vtp')):
        shutil.copy(f, fpath_surf)

    # copy surface and volume mesh
    shutil.copy(db.get_sv_surface(geo), os.path.join(sim_dir, opt['mesh_vtp']))
    shutil.copy(db.get_volume_mesh(geo), os.path.join(sim_dir, opt['mesh_vtu']))

    # copy initial condition mesh
    shutil.copy(db.get_initial_conditions(geo), os.path.join(sim_dir, opt['mesh_initial']))

    # copy wall mesh
    shutil.copy(os.path.join(db.get_sv_meshes(geo), 'walls_combined.vtp'), os.path.join(sim_dir, opt['mesh_walls']))


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


def create_sv_project(db, geo):
    success, err = check_files(db, geo)
    if not success:
        return err

    try:
    # if True:
        make_folders(db, geo)

        write_svproj_file(db, geo)
        write_inflow_const(db, geo)
        write_inflow(db, geo, '3d')
        write_model(db, geo)
        write_simulation(db, geo)

        copy_files(db, geo)
        write_mesh(db, geo)
        write_pre(db, geo, 'svsolver')
        # write_pre(db, geo, 'perigee')
        write_solver(db, geo)
        write_bc(os.path.join(db.get_solve_dir_3d(geo)), db, geo, False)

        err = write_path_segmentation(db, geo)
        # if err:
        #     return '  \nmissing paths:\n' + err
        return False
    except Exception as e:
        return e


def main(db, geometries):
    for geo in geometries:
        print('Running geometry ' + geo)

        err = create_sv_project(db, geo)
        print('  ' + str(err))


if __name__ == '__main__':
    descr = 'Generate an svproject folder'
    d, g, _ = input_args(descr)
    main(d, g)
