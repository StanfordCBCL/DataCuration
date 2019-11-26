#!/usr/bin/env python

import os, subprocess, pdb
import numpy as np
from get_bcs import get_bcs
from get_sim import copy_files, write_pre, write_bc
from get_bc_integrals import integrate_bcs


def geo_integrate_bcs(fpath_sim, geo, res_fields, debug=False, debug_out=''):
    fpath_surf = os.path.join(fpath_sim, geo, 'extras', 'mesh-surfaces', 'extras', 'all_exterior.vtp')
    fpath_vol = os.path.join(fpath_sim, geo, 'results', geo + '_sim_results_in_cm.vtu')

    return integrate_bcs(fpath_surf, fpath_vol, res_fields, debug, debug_out)


if __name__ == '__main__':
    # todo: create simvascular object to handle external calls
    svpre = '/usr/local/sv/svsolver/2019-02-07/svpre'
    svsolver = '/usr/local/sv/svsolver/2019-02-07/svsolver'

    # todo: create object for data base entry to handle names/paths
    # folder for tcl files with boundary conditions
    fpath_bc = '/home/pfaller/work/osmsc/VMR_tcl_repository_scripts/repos_ready_cpm_scripts'

    # folder for simulation files
    fpath_sim = '/home/pfaller/work/osmsc/data_uploaded'

    # folder where generated data is saved
    fpath_gen = '/home/pfaller/work/osmsc/data_generated'

    # folder where simulation is run
    fpath_solve = '/home/pfaller/work/osmsc/simulation'

    # fields to extract
    res_fields = ['velocity', 'pressure']

    # all geometries in repository
    geometries = os.listdir(fpath_sim)
    geometries.sort()
    # geometries = ['0110_0001']
    geometries = ['0001_0001']

    # integrate surface
    for geo in geometries:
        # get boundary integrals
        bc_flow_path = os.path.join(fpath_gen, 'bc_flow', geo + '.npy')
        if not os.path.exists(bc_flow_path):
            continue

        # get boundary conditions
        geo_bc = geo.split('_')[0] + '_' + str(int(geo.split('_')[1]) - 1).zfill(4)
        bc_path = os.path.join(fpath_bc, geo_bc + '-bc.tcl')
        if not os.path.exists(bc_path):
            continue

        print('Processing ' + geo)

        fpath_solve_geo = os.path.join(fpath_solve, geo)
        try:
            os.mkdir(fpath_solve_geo)
        except OSError:
            pass

        bc_flow = np.load(bc_flow_path, allow_pickle=True).item()
        bc_def = get_bcs(bc_path)

        # write flow file
        inflow = bc_flow['velocity'][:, int(bc_def['preid']['inflow']) - 1]
        np.savetxt(os.path.join(fpath_solve_geo, 'inflow.flow'), np.vstack((bc_flow['time'], inflow)).T)

        # try:
        #     # get boundary integrals
        #     fpath_bc_npy = os.path.join(fpath_gen, 'bc_flow', geo)
        #     fpath_bc_debug = os.path.join(fpath_gen, 'vtp', geo + '.vtp')
        #     bc_flow = geo_integrate_bcs(fpath_sim, geo, res_fields, debug=True, debug_out=fpath_bc_debug)
        #     np.save(fpath_bc_npy, bc_flow)
        # except AttributeError:
        #     print('   failed!')

    copy_files(fpath_sim, fpath_solve_geo, fpath_gen, geo)
    fname_pre = write_pre(fpath_solve_geo, bc_def, geo)
    fname_bc = write_bc(fpath_solve_geo, bc_def)

    # run pre-processor
    pre_folder, pre_file = os.path.split(fname_pre)
    subprocess.run([svpre, pre_file], cwd=pre_folder)
    subprocess.run([svsolver, 'solver.inp'], cwd=pre_folder)
