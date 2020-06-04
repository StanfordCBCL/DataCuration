#!/usr/bin/env python

import numpy as np
import os
import meshio
import pdb

from vtk.util.numpy_support import vtk_to_numpy as v2n

from vtk_functions import read_geo, cell_connectivity
from get_database import input_args


def main(db, geometries):
    """
    Generate volume mesh for SimVascular: remove all unused arrays and reoder tet nodes
    """
    for geo in geometries:
        print('Running geometry ' + geo)

        if not os.path.exists(db.get_volume(geo)):
            continue

        # read volume mesh
        vol = read_geo(db.get_volume(geo)).GetOutput()

        # get geometry
        points = v2n(vol.GetPoints().GetData())
        cells = cell_connectivity(vol)

        # reorder nodes in tets to fix negative Jacobian
        tets = cells['tetra'][:, [0, 1, 3, 2]]
        cells['tetra'] = tets

        # check if Jacobian determinant is positive everywhere
        jac = np.zeros((tets.shape[0], 3, 3))
        for i, p in enumerate([[1, 0], [2, 0], [3, 0]]):
            jac[:, :, i] = points[tets[:, p[0]]] - points[tets[:, p[1]]]
        if np.sum(np.linalg.det(jac) <= 0) != 0:
            print('determinant is negative')

        # get arrays
        point_data = {'GlobalNodeID': np.expand_dims(v2n(vol.GetPointData().GetArray('GlobalNodeID')), axis=1)}
        cell_data = {'GlobalElementID': np.expand_dims(v2n(vol.GetCellData().GetArray('GlobalElementID')), axis=1)}

        # write to file
        mesh = meshio.Mesh(points, cells, point_data=point_data, cell_data=cell_data)
        meshio.write(db.get_volume_mesh(geo), mesh)


if __name__ == '__main__':
    descr = 'Create volume mesh for SimVascular'
    d, g, _ = input_args(descr)
    main(d, g)
