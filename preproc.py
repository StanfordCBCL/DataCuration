#!/usr/bin/env python

import os, subprocess, pdb
import numpy as np
from get_bcs import get_bcs
from get_sim import write_pre, write_bc
from get_bc_integrals import integrate_bcs
from get_database import Database, SimVascular, Post


def geo_integrate_bcs(fpath_sim, geo, res_fields, debug=False, debug_out=''):
    fpath_surf = os.path.join(fpath_sim, geo, 'extras', 'mesh-surfaces', 'extras', 'all_exterior.vtp')
    fpath_vol = os.path.join(fpath_sim, geo, 'results', geo + '_sim_results_in_cm.vtu')

    return integrate_bcs(fpath_surf, fpath_vol, res_fields, debug, debug_out)


def main():
    # create object for data base entry to handle names/paths
    db = Database()

    post = Post()

    # loop geometries in repository
    # for geo in db.get_geometries():
    # for geo in ['0119_0001']:
    # for geo in ['0071_0001', '0110_0001', '0119_0002', '0144_0001']:
    for geo in ['0144_1001']:
        print('Processing ' + geo)

        bc_def, _ = db.get_bcs(geo)
        bc_flow = np.load(db.get_bc_flow_path(geo), allow_pickle=True).item()

        # write flow file
        inflow = bc_flow['velocity'][:, int(bc_def['preid']['inflow']) - 1]
        np.savetxt(os.path.join(db.get_solve_dir_3d(geo), 'inflow.flow'), np.vstack((bc_flow['time'], inflow)).T)

        # try:
            # get boundary integrals
        fpath_bc_debug = os.path.join(db.fpath_gen, 'vtp', geo + '.vtp')
        bc_flow = geo_integrate_bcs(db.fpath_sim, geo, ['pressure', 'velocity'], debug=True, debug_out=fpath_bc_debug)
        np.save(db.get_bc_flow_path(geo), bc_flow)

        # exit(0)
        # except AttributeError:
        #     print('   failed!')

        db.copy_files(geo)
        fname_pre = write_pre(db.get_solve_dir_3d(geo), bc_def, geo)
        write_bc(os.path.join(db.get_solve_dir_3d(geo), 'rcrt.dat'), db, geo)


        # run pre-processor
        pre_folder, pre_file = os.path.split(fname_pre)
        sv = SimVascular()
        sv.run_pre(pre_folder, pre_file)
        sv.run_solver(pre_folder, 'solver.inp')


if __name__ == '__main__':
    main()
