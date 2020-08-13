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
from get_database import Database, SimVascular, Post, input_args
from vtk_functions import read_geo, write_geo, ClosestPoints, cell_connectivity
from simulation_io import map_1d_to_centerline


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

    # get 1d field field
    field_1d = v2n(oned.GetPointData().GetArray(field))

    # create laplace FEM stiffness matrix
    laplace = StiffnessMatrix(cells['tetra'], points_vol)

    # solve laplace equation (map desired field from 1d to 3d)
    field_3d = laplace.HarmonicLift(ids_vol, field_1d[ids_cent])

    # create output array
    arr = n2v(field_3d)
    arr.SetName(field)
    vol.GetPointData().AddArray(arr)

    # write to file
    write_geo(f_out, vol)


def main(db, geometries):
    field = 'pressure'
    for geo in geometries:
        f_vol = os.path.join(db.get_sv_meshes(geo), geo + '.vtu')
        f_1d = db.get_1d_flow_path_vtp(geo)
        f_out = db.get_initial_conditions_pressure(geo)

        if not os.path.exists(f_1d):
            continue

        print('Running geometry ' + geo)
        project_1d_3d(f_1d, f_vol, f_out, field)


if __name__ == '__main__':
    descr = 'Get 3D-3D statistics'
    d, g, _ = input_args(descr)
    main(d, g)
