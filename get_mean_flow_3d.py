#!/usr/bin/env python
import vtk
import os
import pdb
import scipy
import meshio
import argparse
import numpy as np

from vtk.util.numpy_support import numpy_to_vtk as n2v
from vtk.util.numpy_support import vtk_to_numpy as v2n
from numpy import array2string as a2s
from scipy.spatial import KDTree

from get_bc_integrals import get_res_names
from get_database import Database, input_args
from vtk_functions import read_geo, write_geo, calculator, cut_plane, connectivity, get_points_cells, clean, Integration

import matplotlib.pyplot as plt


def slice_vessel(inp_3d, origin, normal):
    """
    Slice 3d geometry at certain plane
    Args:
        inp_1d: vtk InputConnection for 1d centerline
        inp_3d: vtk InputConnection for 3d volume model
        origin: plane origin
        normal: plane normal

    Returns:
        Integration object
    """
    # cut 3d geometry
    cut_3d = cut_plane(inp_3d, origin, normal)

    # extract region closest to centerline
    con = connectivity(cut_3d, origin)

    return con


def get_integral(inp_3d, origin, normal):
    """
    Slice simulation at certain plane and integrate
    Args:
        inp_1d: vtk InputConnection for 1d centerline
        inp_3d: vtk InputConnection for 3d volume model
        origin: plane origin
        normal: plane normal

    Returns:
        Integration object
    """
    # slice vessel at given location
    inp = slice_vessel(inp_3d, origin, normal)

    # recursively add calculators for normal velocities
    for v in get_res_names(inp_3d, 'velocity'):
        fun = '(iHat*'+repr(normal[0])+'+jHat*'+repr(normal[1])+'+kHat*'+repr(normal[2])+').' + v
        inp = calculator(inp, fun, [v], 'normal_' + v)

    return Integration(inp)


def extract_results(fpath_1d, fpath_3d):
    """
    Extract 3d results at 1d model nodes (integrate over cross-section)
    Args:
        fpath_1d: path to 1d model
        fpath_3d: path to 3d simulation results

    Returns:
        res: dictionary of results in all branches, in all segments for all result arrays
    """
    if not os.path.exists(fpath_1d) or not os.path.exists(fpath_3d):
        return None

    # read 1d and 3d model
    reader_1d = read_geo(fpath_1d)
    reader_3d = read_geo(fpath_3d)

    # get all result array names
    res_names = get_res_names(reader_3d, ['pressure', 'velocity'])

    # get point and normals from centerline
    points = v2n(reader_1d.GetOutput().GetPoints().GetData())
    normals = v2n(reader_1d.GetOutput().GetPointData().GetArray('CenterlineSectionNormal'))
    gid = v2n(reader_1d.GetOutput().GetPointData().GetArray('GlobalNodeId'))

    # initialize output
    for name in res_names + ['area']:
        array = vtk.vtkDoubleArray()
        array.SetName(name)
        array.SetNumberOfValues(reader_1d.GetOutput().GetNumberOfPoints())
        array.Fill(0)
        reader_1d.GetOutput().GetPointData().AddArray(array)

    # move points on caps slightly to ensure nice integration
    ids = vtk.vtkIdList()
    eps_norm = 1.0e-3

    # integrate results on all points of intergration cells
    for i in range(reader_1d.GetOutput().GetNumberOfPoints()):
        # check if point is cap
        reader_1d.GetOutput().GetPointCells(i, ids)
        if ids.GetNumberOfIds() == 1:
            if gid[i] == 0:
                # inlet
                points[i] += eps_norm * normals[i]
            else:
                # outlets
                points[i] -= eps_norm * normals[i]

        # create integration object (slice geometry at point/normal)
        integral = get_integral(reader_3d, points[i], normals[i])

        # integrate all output arrays
        for name in res_names:
            reader_1d.GetOutput().GetPointData().GetArray(name).SetValue(i, integral.evaluate(name))
        reader_1d.GetOutput().GetPointData().GetArray('area').SetValue(i, integral.area())

    return reader_1d.GetOutput()


def main(db, geometries):
    """
    Loop all geometries
    """
    for geo in geometries:
        print('Running geometry ' + geo)

        fpath_1d = db.get_centerline_path(geo)
        fpath_3d = db.get_volume(geo)

        if os.path.exists(db.get_3d_flow_path_oned_vtp(geo)):
            print('  Already exists. Skipping...')
            continue

        # extract 3d results integrated over cross-section
        try:
            res = extract_results(fpath_1d, fpath_3d)
        except Exception as e:
            print(e)
            continue

        if res is not None:
            write_geo(db.get_3d_flow_path_oned_vtp(geo), res)


if __name__ == '__main__':
    descr = 'Extract 3d-results at 1d-locations'
    d, g, _ = input_args(descr)
    main(d, g)

