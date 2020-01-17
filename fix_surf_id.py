#!/usr/bin/env python

import numpy as np
import os

from vtk.util.numpy_support import vtk_to_numpy as v2n
from vtk.util.numpy_support import numpy_to_vtk as n2v

from get_database import Database
from vtk_functions import read_geo, write_geo
from common import input_args


def main(db, geometries):
    for geo in geometries:
        print('Fixing geometry ' + geo)
        folder_out = os.path.join(db.fpath_gen, 'surfaces', geo)

        f_all = os.path.join(folder_out, 'all_exterior.vtp')
        f_wall = os.path.join(folder_out, 'wall.vtp')

        read_all, node_all, cell_all = read_geo(f_all)
        read_wall, node_wall, cell_wall = read_geo(f_wall)

        # read indices
        fid_all = v2n(cell_all.GetArray('BC_FaceID'))
        eid_all = v2n(cell_all.GetArray('GlobalElementID'))
        eid_wall = v2n(cell_wall.GetArray('GlobalElementID'))

        # find wall indices in all_exterior
        index = np.argsort(eid_all)
        search = np.searchsorted(eid_all[index], eid_wall)
        ind_wall = index[search]

        wall_id = np.unique(fid_all[ind_wall])
        assert wall_id.shape[0] == 1, 'wall BC_FaceID not unique'

        if wall_id[0] == 0:
            print('  Nothing to do')
            continue

        # change id
        fid_all[ind_wall] = 0

        # export
        out_array = n2v(fid_all)
        out_array.SetName('BC_FaceID')
        cell_all.RemoveArray('BC_FaceID')
        cell_all.AddArray(out_array)
        write_geo(f_all, read_all)


if __name__ == '__main__':
    descr = 'Fix wrong surface id'
    d, g, _ = input_args(descr)
    main(d, g)
