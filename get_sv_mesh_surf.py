#!/usr/bin/env python

import pdb
import vtk
import os
import tempfile
import numpy as np

from vtk.util.numpy_support import vtk_to_numpy as v2n
from vtk.util.numpy_support import numpy_to_vtk as n2v

from get_bc_integrals import read_geo, write_geo
from get_database import Database, input_args


def get_indices(a, b):
    """
    Elementwise True if an entry in a is in any of b
    """
    i = np.zeros(a.shape, dtype=bool)
    for j in b:
        i |= (a == j)
    return i


def generate(db, geo):
    """
    Generate arrays in surface mesh used by SimVascular
    """

    f_surf = db.get_surfaces(geo, 'all_exterior')
    if not f_surf:
        return 'no surface mesh'

    # read volume mesh with results
    surf = read_geo(f_surf).GetOutput()
    surf_p = surf.GetPointData()
    surf_c = surf.GetCellData()

    # reconstruct SimVascular arrays from BC_FaceID
    face_id = v2n(surf_c.GetArray('BC_FaceID'))

    # read surface ids
    try:
        caps = get_indices(face_id, db.get_surface_ids(geo, 'caps'))
        inflow = get_indices(face_id, db.get_surface_ids(geo, 'inflow'))
        outlets = get_indices(face_id, db.get_surface_ids(geo, 'outlets'))
    except (KeyError, TypeError):
        return 'face missing in boundary conditions'

    # initialize new arrays
    n_names = ['GlobalBoundaryPoints']
    c_names = ['GlobalBoundaryCells', 'CapID', 'BadTriangle', 'FreeEdge', 'BooleanRegion', 'ModelFaceID',
               'Normals', 'ActiveCells']
    arrays = {}
    for n in n_names:
        arrays[n] = {'handle': surf_p, 'array': np.zeros(surf.GetNumberOfPoints(), dtype=np.int64)}
    for n in c_names:
        arrays[n] = {'handle': surf_c, 'array': np.zeros(surf.GetNumberOfCells(), dtype=np.int64)}

    # rename
    arrays['ModelFaceID']['array'] = face_id

    # all caps
    arrays['ActiveCells']['array'][caps] = 1

    # inflow is 1, outflow is 2
    arrays['CapID']['array'][inflow] = 1
    arrays['CapID']['array'][outlets] = 2

    # remove old array
    surf_c.RemoveArray('BC_FaceID')

    # add new arrays
    for n, v in arrays.items():
        out_array = n2v(v['array'])
        out_array.SetName(n)
        v['handle'].AddArray(out_array)

    # generate normals
    normals = vtk.vtkPolyDataNormals()
    normals.SetInputData(surf)
    normals.ComputePointNormalsOff()
    normals.ComputeCellNormalsOn()
    normals.SplittingOff()
    normals.Update()

    # export to generated folder
    write_geo(db.get_sv_surface_path(geo), normals.GetOutput())

    return None


def main(db, geometries):
    for geo in geometries:
        print('Running geometry ' + geo)

        err = generate(db, geo)
        if err:
            print('  ' + err)


if __name__ == '__main__':
    descr = 'Fix wrong GlobalElementID'
    d, g, _ = input_args(descr)
    main(d, g)
