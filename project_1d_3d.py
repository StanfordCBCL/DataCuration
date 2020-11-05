#!/usr/bin/env python

import pdb
import sys
import os
import vtk
import shutil

from collections import defaultdict
import numpy as np
from vtk.util.numpy_support import vtk_to_numpy as v2n
from vtk.util.numpy_support import numpy_to_vtk as n2v

from lift_laplace import StiffnessMatrix
from get_database import Database, SimVascular, Post, input_args
from vtk_functions import read_geo, write_geo, ClosestPoints, cell_connectivity, region_grow, collect_arrays
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


def get_1d_3d_map(f_1d, f_vol):
    # read geoemtries
    vol = read_geo(f_vol).GetOutput()
    oned = read_geo(f_1d).GetOutput()

    # get points
    points_vol = v2n(vol.GetPoints().GetData())
    points_1d = v2n(oned.GetPoints().GetData())

    # get volume points closest to centerline
    cp_vol = ClosestPoints(vol)
    seed_points = np.unique(cp_vol.search(points_1d))

    # map centerline points to selected volume points
    cp_1d = ClosestPoints(oned)
    seed_ids = np.array(cp_1d.search(points_vol[seed_points]))

    # call region growing algorithm
    ids, dist, rad = region_grow(vol, seed_points, seed_ids, n_max=999)

    # check 1d to 3d map
    assert np.max(ids) <= oned.GetNumberOfPoints() - 1, '1d-3d map non-conforming'

    return ids, dist, rad


def add_array(geo, name, array):
    arr = n2v(array)
    arr.SetName(name)
    geo.GetPointData().AddArray(arr)


def project_1d_3d_grow(f_1d, f_vol, f_wall, f_out):
    # read geometries
    vol = read_geo(f_vol).GetOutput()
    cent = read_geo(f_1d).GetOutput()
    wall = read_geo(f_wall).GetOutput()

    # get 1d -> 3d map
    map_ids, map_iter, map_rad = get_1d_3d_map(f_1d, f_vol)

    # get arrays
    arrays_cent = collect_arrays(cent.GetPointData())

    # map all centerline arrays to volume geometry
    for name, array in arrays_cent.items():
        add_array(vol, name, array[map_ids])

    # add mapping to volume mesh
    for name, array in zip(['MapIds', 'MapIters'], [map_ids, map_iter]):
        add_array(vol, name, array)

    # inverse map
    map_ids_inv = {}
    for i in np.unique(map_ids):
        map_ids_inv[i] = np.where(map_ids == i)

    # create radial coordinate [0, 1]
    rad = np.zeros(vol.GetNumberOfPoints())
    for i, ids in map_ids_inv.items():
        rad_max = np.max(map_rad[ids])
        if rad_max == 0:
            rad_max = np.max(map_rad)
        rad[ids] = map_rad[ids] / rad_max
    add_array(vol, 'rad', rad)

    # set points at wall to hard 1
    wall_ids = collect_arrays(wall.GetPointData())['GlobalNodeID'].astype(int) - 1
    rad[wall_ids] = 1

    # mean velocity
    names = ['flow', 'velocity']
    for n in names:
        for a in arrays_cent.keys():
            if n in a:
                u_mean = arrays_cent[a] / arrays_cent['CenterlineSectionArea']

                # parabolic velocity
                u_quad = 2 * u_mean[map_ids] * (1 - rad**2)

                # scale parabolic flow profile to preserve mean flow
                for i, ids in map_ids_inv.items():
                    u_mean_is = np.mean(u_quad[map_ids_inv[i]])
                    u_quad[ids] *= u_mean[i] / u_mean_is

                # parabolic velocity vector field
                velocity = np.outer(u_quad, np.ones(3)) * arrays_cent['CenterlineSectionNormal'][map_ids]

                # add to volume mesh
                if n == 'velocity':
                    aname = a
                elif n == flow:
                    aname = 'velocity'
                add_array(vol, aname, velocity)

    # write to file
    write_geo(f_out, vol)


def get_error(f_3d, f_1d, f_out):
    geo_3d = read_geo(f_3d).GetOutput()
    geo_1d = read_geo(f_1d).GetOutput()
    arrays_3d = collect_arrays(geo_3d.GetPointData())
    arrays_1d = collect_arrays(geo_1d.GetPointData())

    for m in arrays_1d.keys():
        if 'pressure' in m:
            norm = np.mean(arrays_3d[m])
            err = np.abs(arrays_3d[m] - arrays_1d[m]) / norm
            add_array(geo_1d, 'error_' + m, err)
        if 'velocity' in m:
            norm = np.mean(np.linalg.norm(arrays_3d[m], axis=1))
            err = np.linalg.norm(arrays_3d[m] - arrays_1d[m], axis=1) / norm
            add_array(geo_1d, 'error_' + m, err)
            pdb.set_trace()
    write_geo(f_out, geo_1d)


def main(db, geometries):
    for geo in geometries:
        f_vol = os.path.join(db.get_sv_meshes(geo), geo + '.vtu')
        f_0d = db.get_0d_flow_path_vtp(geo)
        f_1d = db.get_1d_flow_path_vtp(geo)
        f_wall = db.get_surfaces(geo, 'wall')
        f_out = db.get_initial_conditions_pressure(geo) #'test.vtu'#

        if os.path.exists(f_1d):
            print(geo + ' using 1d')
            f_red = f_1d
        elif os.path.exists(f_0d):
            print(geo + ' using 0d')
            f_red = f_0d
        else:
            print(geo + ' no 0d/1d solution found')
            continue

        # if os.path.exists(f_out):
        #     print('  map exists')
        #     continue

        project_1d_3d_grow(f_red, f_vol, f_wall, f_out)


def convert_time(db, geometries):
    for geo in geometries:
        f_vol = os.path.join(db.get_sv_meshes(geo), geo + '.vtu')
        f_res = db.get_volume(geo)
        f_red = db.get_3d_flow(geo)
        f_wall = db.get_surfaces(geo, 'wall')
        d_out = os.path.join('/home/pfaller/work/osmsc/extrapolation/', geo)
        f_out = os.path.join(d_out, geo + '_mapped.vtu')
        f_err = os.path.join(d_out, geo + '_error.vtu')

        os.makedirs(d_out, exist_ok=True)

        project_1d_3d_grow(f_red, f_vol, f_wall, f_out)

        shutil.copy(f_res, os.path.join(d_out, geo + '.vtu'))
        shutil.copy(f_red, d_out)

        get_error(f_res, f_out, f_err)


if __name__ == '__main__':
    descr = 'Get 3D-3D statistics'
    d, g, _ = input_args(descr)
    # main(d, g)
    convert_time(d, g)
