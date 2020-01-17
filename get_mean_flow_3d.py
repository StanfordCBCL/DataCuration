#!/usr/bin/env python
import vtk
import os
import pdb
import scipy
import argparse
import numpy as np

from vtk.util.numpy_support import vtk_to_numpy as v2n

from common import input_args
from get_bc_integrals import get_res_names
from get_database import Database
from vtk_functions import read_geo, threshold, calculator, cut_plane, connectivity, Integration


def sort_faces(res_faces, area, path):
    """
    Arrange results from surface integration in matrix
    Args:
        res_faces: dictionary with key: segment id, value: result at a certain time step
        area: cross-sectional area of each segment
        path: one-dimensional path coordinate

    Returns:
        dictionary with results as keys and matrix at all surfaces/time steps as values
    """
    # get time steps
    times = np.unique([float(k.split('_')[1]) for v in res_faces.values() for k in v.keys()])

    # solution fields
    fields = np.unique([k.split('_')[0] for v in res_faces.values() for k in v.keys()])
    fields = np.append(fields, 'area')

    # initialize
    res = {'time': times, 'path': np.zeros(len(res_faces))}
    for f in fields:
        res[f] = np.zeros((times.shape[0], len(res_faces)))

    for k, v in res_faces.items():
        # constant area for all time steps
        res['area'][:, k] = area[k]

        # path coordinate
        res['path'][k] = path[k]

        # sort time steps
        for r_name, r in v.items():
            name, time = r_name.split('_')
            res[name][float(time) == times, k] = r

    return res


def get_integral(inp, origin, normal):
    """
    Slice simulation at certain plane and integrate
    Args:
        inp: vtk InputConnection
        origin: plane origin
        normal: plane normal

    Returns:
        Integration object
    """
    # cut geometry
    cut = cut_plane(inp, origin, normal)

    # extract region closest to centerline if there are several slices
    con = connectivity(cut, origin)

    # recursively add calculators for normal velocities
    calc = con
    for v in get_res_names(inp, 'velocity'):
        fun = '(iHat*'+repr(normal[0])+'+jHat*'+repr(normal[1])+'+kHat*'+repr(normal[2])+').' + v
        calc = calculator(calc, fun, [v], 'normal_' + v)

    return Integration(calc)


def norm_row(x):
    """
    Normalize each row of a matrix
    Args:
        x: vector or matrix

    Returns:
        normalized vector/matrix
    """
    # length of each row
    length = np.sum(np.abs(x)**2, axis=-1)**(1./2)

    # x is vector, length is scalar
    if not length.shape:
        return x / length

    # x is matrix, length is vector
    else:
        return x / length[:, np.newaxis]


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

    # fields to extract
    res_fields = ['pressure', 'velocity']

    # read 1d and 3d model
    reader_1d, _, _ = read_geo(fpath_1d)
    reader_3d, _, _ = read_geo(fpath_3d)

    # get all result array names
    res_names = get_res_names(reader_3d, res_fields)

    # number of points in model
    n_point = reader_1d.GetOutput().GetNumberOfPoints()
    n_cell = reader_1d.GetOutput().GetNumberOfCells()
    assert n_point == n_cell + 1, 'geometry inconsistent'

    points = v2n(reader_1d.GetOutput().GetPoints().GetData())
    normals = v2n(reader_1d.GetOutput().GetPointData().GetArray('normals'))
    path_1d = v2n(reader_1d.GetOutput().GetPointData().GetArray('path'))
    seg_id = v2n(reader_1d.GetOutput().GetCellData().GetArray('seg_id'))

    # integrate results on each point
    res = {}
    area = {}
    path = {}
    for i in range(n_cell):
        # id of vessel segment
        s = seg_id[i]

        # id of down-stream node of cell
        p = reader_1d.GetOutput().GetCell(i).GetPointId(1)

        # create integration object (slice geometry at point/normal)
        integral = get_integral(reader_3d, points[p], normals[p])

        # integrate all output arrays
        res[s] = {}
        for r in res_names:
            res[s][r] = integral.evaluate(r)

        # store cross-sectional area
        area[s] = integral.area()

        # store path coordinate
        path[s] = path_1d[p]

    return sort_faces(res, area, path)


def main(db, geometries):
    """
    Loop all geometries
    """
    for geo in geometries:
        print('Running geometry ' + geo)

        fpath_1d = db.get_1d_geo(geo)
        fpath_3d = db.get_volume(geo)

        if not os.path.exists(fpath_1d) or not os.path.exists(fpath_3d):
            continue

        if os.path.exists(db.get_3d_flow_path(geo)):
            continue

        # extract 3d results integrated over cross-section
        try:
            res = extract_results(fpath_1d, fpath_3d)

            # save to file
            if res is not None:
                np.save(db.get_3d_flow_path(geo), res)
        except:
            continue


if __name__ == '__main__':
    descr = 'Extract 3d-results at 1d-locations'
    d, g, _ = input_args(descr)
    main(d, g)

