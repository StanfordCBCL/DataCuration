#!/usr/bin/env python

import os
import meshio
import numpy as np
import pdb

from vtk_functions import read_geo, get_all_arrays, cell_connectivity
from get_database import Database, input_args

from vtk.util.numpy_support import vtk_to_numpy as v2n


def split(array):
    """
    Split array name in name and time step if possible
    """
    comp = array.split('_')
    num = comp[-1]

    # check if array name has a time step
    try:
        time = float(num)
        name = '_'.join([c for c in comp[:-1]])
    except ValueError:
        time = 0
        name = array
    return time, name


def convert_data(data):
    """
    Change array dimensions to comply with meshio
    """
    if len(data.shape) == 1:
        return np.expand_dims(data, axis=1)
    elif len(data.shape) == 2 and data.shape[1] == 4:
        return data[:, 1:]
    else:
        return data


def convert(f_in, f_out):
    """
    Convert .vtu/.vtp to xdmf
    """
    # read geometry
    geo = read_geo(f_in).GetOutput()
    point_arrays, cell_arrays = get_all_arrays(geo)

    # extract connectivity
    cells = cell_connectivity(geo)
    points = v2n(geo.GetPoints().GetData())

    with meshio.xdmf.TimeSeriesWriter(f_out) as writer:
        # write points and cells
        writer.write_points_cells(points, cells)

        # write point data
        for array, data in point_arrays.items():
            time, name = split(array)
            writer.write_data(time, point_data={name: convert_data(data)})

        # write cell data
        for array, data in cell_arrays.items():
            time, name = split(array)
            writer.write_data(time, cell_data={name: convert_data(data)})


def main(db, geometries):
    """
    Loop all geometries
    """
    for geo in geometries:
        print('Running geometry ' + geo)

        f_in = db.get_3d_flow(geo)
        f_out = os.path.splitext(f_in)[0] + '.xdmf'

        if not os.path.exists(f_in):
            continue

        convert(f_in, f_out)


if __name__ == '__main__':
    descr = 'Extract 3d-results at 1d-locations'
    d, g, _ = input_args(descr)
    main(d, g)
