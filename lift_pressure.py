#!/usr/bin/env python

import pdb
import sys
import os
import vtk

from collections import defaultdict
import numpy as np
from vtk.util.numpy_support import vtk_to_numpy as v2n
from vtk.util.numpy_support import numpy_to_vtk as n2v

from lift_laplace import StiffnessMatrix
from vtk_functions import read_geo, write_geo, ClosestPoints, cell_connectivity


def project_1d_3d(f_1d, f_vol, f_out, field):
    # read volume mesh
    vol = read_geo(f_vol).GetOutput()
    points_vol = v2n(vol.GetPoints().GetData())
    cells = cell_connectivity(vol)

    # read 1d results
    oned = read_geo(f_1d).GetOutput()
    points_1d = v2n(oned.GetPoints().GetData())

    # get volume points closest to centerline
    cp_vol = ClosestPoints(vol)
    ids_vol = np.unique(cp_vol.search(points_1d))

    # get centerline points closest to selected volume points
    cp_1d = ClosestPoints(oned)
    ids_cent = cp_1d.search(points_vol[ids_vol])

    # visualize imprint of centerline in volume mesh
    imprint = np.zeros(vol.GetNumberOfPoints())
    imprint[ids_vol] = 1
    arr = n2v(imprint)
    arr.SetName('imprint')
    vol.GetPointData().AddArray(arr)

    # get 1d pressure field
    pressure_1d = v2n(oned.GetPointData().GetArray(field))

    # create laplace FEM stiffness matrix
    laplace = StiffnessMatrix(cells['tetra'], points_vol)

    # solve laplace equation (map desired field from 1d to 3d)
    pressure = laplace.HarmonicLift(ids_vol, pressure_1d[ids_cent])

    # create output array
    arr = n2v(pressure)
    arr.SetName(field)
    vol.GetPointData().AddArray(arr)

    # write to file
    write_geo(f_out, vol)


def main():
    f_vol = '/home/pfaller/work/osmsc/initial_from_1d/projection_input/0003_0001.vtu'
    f_1d = '/home/pfaller/work/osmsc/initial_from_1d/projection_input/0003_0001_1d.vtp'
    f_out = '/home/pfaller/work/osmsc/initial_from_1d/projection_output_lift/0003_0001.vtu'
    field = 'pressure'
    project_1d_3d(f_1d, f_vol, f_out, field)


if __name__ == '__main__':
    main()
