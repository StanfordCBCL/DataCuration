#!/usr/bin/env python

import pdb
import vtk
import os
import tempfile
import numpy as np

from vtk.util.numpy_support import vtk_to_numpy as v2n
from vtk.util.numpy_support import numpy_to_vtk as n2v

from common import input_args
from get_bc_integrals import read_geo, write_geo
from get_database import Database
from get_sim import write_bc


def get_indices(a, b):
    """
    Elementwise True if an entry in a is in any of b
    """
    i = np.zeros(a.shape, dtype=bool)
    for j in b:
        i |= (a == j)
    return i


def main(db, geometries):
    """
    Generate arrays in surface mesh used by SimVascular
    """
    for geo in geometries:
        print('Running geometry ' + geo)

        f_surf = db.get_surfaces(geo, 'all_exterior')
        if not f_surf:
            print('  no surface mesh')
            continue

        _, err = write_bc(tempfile.gettempdir(), db, geo)
        if err:
            print(err)
            continue

        # read volume mesh with results
        surf, surf_p, surf_c = read_geo(f_surf[0])

        # reconstruct SimVascular arrays from BC_FaceID
        face_id = v2n(surf_c.GetArray('BC_FaceID'))

        # read surface ids
        caps = get_indices(face_id, db.get_surface_ids(geo, 'caps'))
        inflow = get_indices(face_id, db.get_surface_ids(geo, 'inflow'))
        outlets = get_indices(face_id, db.get_surface_ids(geo, 'outlets'))

        # initialize new arrays
        n_names = ['GlobalBoundaryPoints']
        c_names = ['GlobalBoundaryCells', 'CapID', 'BadTriangle', 'FreeEdge', 'BooleanRegion', 'ModelFaceID',
                   'Normals', 'ActiveCells']
        arrays = {}
        for n in n_names:
            arrays[n] = {'handle': surf_p, 'array': np.zeros(surf.GetOutput().GetNumberOfPoints(), dtype=np.int64)}
        for n in c_names:
            arrays[n] = {'handle': surf_c, 'array': np.zeros(surf.GetOutput().GetNumberOfCells(), dtype=np.int64)}

        # rename
        arrays['ModelFaceID']['array'] = face_id

        # all caps
        arrays['ActiveCells']['array'][caps] = 1

        # inflow is 1, outflow is 2
        arrays['CapID']['array'][inflow] = 1
        arrays['CapID']['array'][outlets] = 2

        # remove old arrays
        surf_p.RemoveArray('GlobalNodeID')
        surf_c.RemoveArray('GlobalElementID')
        surf_c.RemoveArray('BC_FaceID')

        # add new arrays
        for n, v in arrays.items():
            out_array = n2v(v['array'])
            out_array.SetName(n)
            v['handle'].AddArray(out_array)

        # generate normals
        normals = vtk.vtkPolyDataNormals()
        normals.SetInputData(surf.GetOutput())
        normals.ComputeCellNormalsOn()
        normals.ComputePointNormalsOff()
        normals.ConsistencyOff()
        normals.Update()

        # export to generated folder
        write_geo(db.get_sv_surface(geo), normals)


if __name__ == '__main__':
    descr = 'Fix wrong GlobalElementID'
    d, g, _ = input_args(descr)
    main(d, g)
