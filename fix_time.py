#!/usr/bin/env python

import pdb
import numpy as np
import os

from vtk.util.numpy_support import vtk_to_numpy as v2n
from vtk.util.numpy_support import numpy_to_vtk as n2v

from common import get_dict
from get_database import Database, input_args
from vtk_functions import read_geo, write_geo


def main(db, geometries):
    steps_new = {'0074_0001': 600,
                 '0075_1001': 1600,
                 '0090_0001': 1200}

    # get bcs of all models
    bc_all = get_dict(db.db_params)

    for geo in geometries:
        # 3d exported time
        time, inflow = db.get_inflow(geo)
        if time is None:
            continue

        # db time step
        dt = db.get_3d_timestep(geo)

        # exported time steps
        time_export = time / dt

        # find increment in time steps
        nt_out = np.unique(np.diff(np.rint(time_export).astype(int)))

        # overwrite steps per cycle if output data not equally spaced
        if len(nt_out) > 1:
            print('Fixing geometry ' + geo)
            bc_all[geo]['params']['sim_steps_per_cycle'] = str(steps_new[geo])
    np.save(db.db_params, bc_all)

    # check
    for geo in steps_new.keys():
        print('Checking geometry ' + geo + ' nt_out = ' + str(db.get_3d_increment(geo)))


if __name__ == '__main__':
    descr = 'Fix wrong time step / numstep'
    d, g, _ = input_args(descr)
    main(d, g)
