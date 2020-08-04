#!/usr/bin/env python
import vtk
import os
import meshio
import argparse
import glob
import numpy as np
import pdb

from vtk.util.numpy_support import vtk_to_numpy as v2n
from vtk.util.numpy_support import numpy_to_vtk as n2v

from get_bc_integrals import get_res_names, read_geo, write_geo
from get_database import Database, input_args
from vtk_functions import read_geo, write_geo, cell_connectivity, collect_arrays


def jacobian_positive(points, tets):
    """
    Check if Jacobian determinant is positive everywhere
    """
    jac = np.zeros((tets.shape[0], 3, 3))
    for i, p in enumerate([[1, 0], [2, 0], [3, 0]]):
        jac[:, :, i] = points[tets[:, p[0]]] - points[tets[:, p[1]]]
    return np.sum(np.linalg.det(jac) <= 0.0) == 0


def get_vol(db, geo, ini_from_1d=True):
    """
    Generate volume mesh for SimVascular: remove all unused arrays and reoder tet nodes
    """
    f_vol = db.get_volume(geo)
    f_out = os.path.join(db.get_sv_meshes(geo), geo + '.vtu')
    f_ini = db.get_initial_conditions(geo)

    if not os.path.exists(f_vol):
        print('  no volume mesh')
        return

    print('  generating volume mesh')

    # read volume mesh
    vol = read_geo(f_vol).GetOutput()

    # get geometry
    points = v2n(vol.GetPoints().GetData())
    cells = cell_connectivity(vol)

    # reorder nodes in tets to fix negative Jacobian
    if not jacobian_positive(points, cells['tetra']):
        cells['tetra'] = cells['tetra'][:, [0, 1, 3, 2]]
        print('    tets flipped')
    else:
        print('    tets ok')
    # assert jacobian_positive(points, cells['tetra']), 'Jacobian negative after flipping tets'

    if not jacobian_positive(points, cells['tetra']):
        print('    Jacobian negative after flipping tets')

    # get arrays
    point_data = {'GlobalNodeID': np.expand_dims(v2n(vol.GetPointData().GetArray('GlobalNodeID')), axis=1)}
    cell_data = {'GlobalElementID': np.expand_dims(v2n(vol.GetCellData().GetArray('GlobalElementID')), axis=1)}

    # write raw write to file
    mesh = meshio.Mesh(points, cells, point_data=point_data, cell_data=cell_data)
    meshio.write(f_out, mesh)

    # get results
    res = collect_arrays(vol.GetPointData())

    # get last time step
    times = np.unique([float(k.split('_')[1]) for k in res.keys() if '_' in k])
    t_last = str(times[-1])

    # add initial conditions
    for k in res.keys():
        if t_last in k:
            if 'ini_from_1d' and 'pressure' in k:
                geo_1d = read_geo(db.get_initial_conditions_pressure(geo)).GetOutput()
                point_data[k] = v2n(geo_1d.GetPointData().GetArray('pressure'))
            else:
                point_data[k] = res[k]

    # write initial conditions to file
    mesh = meshio.Mesh(points, cells, point_data=point_data, cell_data=cell_data)
    meshio.write(f_ini, mesh)


def get_indices(a, b):
    """
    Elementwise True if an entry in a is in any of b
    """
    i = np.zeros(a.shape, dtype=bool)
    for j in b:
        i |= (a == j)
    return i


def get_surf(db, geo):
    """
    Generate arrays in surface mesh used by SimVascular
    """
    # get all surfaces
    surfaces = glob.glob(os.path.join(db.get_surface_dir(geo), '*.vtp'))

    for f_surf in surfaces:
        # read volume mesh with results
        surf = read_geo(f_surf).GetOutput()
        surf_p = surf.GetPointData()
        surf_c = surf.GetCellData()

        # get output name
        name_osmsc = os.path.basename(f_surf)
        if 'all_exterior' in name_osmsc:
            name = geo + '.vtp'
        elif 'wall' in name_osmsc:
            name = 'walls_combined.vtp'
        else:
            name = os.path.join('caps', name_osmsc)

        print('  generating surface mesh ' + name)

        f_out = os.path.join(db.get_sv_meshes(geo), name)

        # reconstruct SimVascular arrays from BC_FaceID
        face_id = v2n(surf_c.GetArray('BC_FaceID'))

        # read surface ids
        try:
            caps = get_indices(face_id, db.get_surface_ids(geo, 'caps'))
            inflow = get_indices(face_id, db.get_surface_ids(geo, 'inflow'))
            outlets = get_indices(face_id, db.get_surface_ids(geo, 'outlets'))
        except (KeyError, TypeError):
            print('  face missing in boundary conditions')
            return

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
        arrays['ModelFaceID']['array'] = face_id + 1

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
        write_geo(f_out, normals.GetOutput())


def get_meshes(db, geo):
    get_vol(db, geo)
    get_surf(db, geo)


def main(db, geometries):
    """
    Loop all geometries
    """
    for geo in geometries:
        print('Running geometry ' + geo)

        get_meshes(db, geo)


if __name__ == '__main__':
    descr = 'Generate all meshes for SimVascular'
    d, g, _ = input_args(descr)
    main(d, g)

