#!/usr/bin/env python
import vtk
import os
import numpy as np
import concurrent
from concurrent.futures import ProcessPoolExecutor as PPE 
from vtk.util.numpy_support import vtk_to_numpy as v2n
from tqdm import tqdm
from functools import partial

from get_bc_integrals import get_res_names
from get_database import input_args
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

def worker_process(i_list,fpath_1d,fpath_3d,res_names,points,normals,gid):
    # check if point is cap
    reader_1d = read_geo(fpath_1d).GetOutput()
    reader_3d = read_geo(fpath_3d).GetOutput()
    ids = vtk.vtkIdList()
    integration_data = []
    area_data = []
    for i in i_list:
        reader_1d.GetPointCells(i, ids)
        if ids.GetNumberOfIds() == 1:
            if gid[i] == 0:
                # inlet
                points_tmp = points[i] + 1.0e-3 * normals[i]
                normals_tmp = normals[i]
            else:
                # outlets
                points_tmp = points[i] - 1.0e-3 * normals[i]
                normals_tmp = normals[i]
        else:
            points_tmp = points[i]
            normals_tmp = normals[i]

        # create integration object (slice geometry at point/normal)
        try:
            integral = get_integral(reader_3d, points_tmp, normals_tmp)
        except Exception:
            return np.NaN, np.NaN

        # integrate all output arrays
        for name in res_names:
            int_tmp = []
            int_tmp.append(name)
            int_tmp.append(i)
            int_tmp.append(integral.evaluate(name))
            integration_data.append(int_tmp)
            #reader_1d.GetPointData().GetArray(name).SetValue(i, integral.evaluate(name))
        area_data.append(['area',i,integral.area()])
        #reader_1d.GetPointData().GetArray('area').SetValue(i, integral.area())
    return integration_data, area_data

def extract_results(fpath_1d, fpath_3d, fpath_out, only_caps=False,workers=4,chunksize=100):
    """
    Extract 3d results at 1d model nodes (integrate over cross-section)
    Args:
        fpath_1d: path to 1d model
        fpath_3d: path to 3d simulation results
        fpath_out: output path
        only_caps: extract solution only at caps, not in interior (much faster)

    Returns:
        res: dictionary of results in all branches, in all segments for all result arrays
    """
    # read 1d and 3d model
    reader_1d = read_geo(fpath_1d).GetOutput()
    reader_3d = read_geo(fpath_3d).GetOutput()

    # get all result array names
    res_names = get_res_names(reader_3d, ['pressure', 'velocity'])

    # get point and normals from centerline
    points = v2n(reader_1d.GetPoints().GetData())
    normals = v2n(reader_1d.GetPointData().GetArray('CenterlineSectionNormal'))
    gid = v2n(reader_1d.GetPointData().GetArray('GlobalNodeId'))

    # initialize output
    for name in res_names + ['area']:
        array = vtk.vtkDoubleArray()
        array.SetName(name)
        array.SetNumberOfValues(reader_1d.GetNumberOfPoints())
        array.Fill(0)
        reader_1d.GetPointData().AddArray(array)

    # move points on caps slightly to ensure nice integration
    ids = vtk.vtkIdList()
    eps_norm = 1.0e-3

    # integrate results on all points of intergration cells
    n = workers
    with PPE(max_workers=n) as executor:
        if not only_caps and workers > 1:
            tqdm_list = list(range(reader_1d.GetNumberOfPoints()))
            chunknum = len(tqdm_list)//chunksize
            chunks = []
            for j in range(1,chunknum+1):
                if j == chunknum:
                    chunks.append(tqdm_list[(j-1)*chunksize:])
                else:
                    chunks.append(tqdm_list[(j-1)*chunksize:j*chunksize])
            futures_to_read = {executor.submit(partial(worker_process,
                                               fpath_1d=fpath_1d,
                                               fpath_3d=fpath_3d,
                                               res_names=res_names,
                                               points=points,
                                               normals=normals,gid=gid),i): i for i in chunks}
            results = []
            for future in list(tqdm(concurrent.futures.as_completed(futures_to_read),total=len(chunks))):
                res = future.result()
                results.append(res)
                int_data = res[0]
                area_data = res[1]
                for j in int_data:
                    reader_1d.GetPointData().GetArray(j[0]).SetValue(j[1],j[2])
                for j in area_data: 
                    reader_1d.GetPointData().GetArray(j[0]).SetValue(j[1],j[2])
        else:
            for i in tqdm(range(reader_1d.GetNumberOfPoints())):
                # check if point is cap
                reader_1d.GetPointCells(i, ids)
                if ids.GetNumberOfIds() == 1:
                    if gid[i] == 0:
                        # inlet
                        points[i] += eps_norm * normals[i]
                    else:
                        # outlets
                        points[i] -= eps_norm * normals[i]
                else:
                    if only_caps:
                        continue

                # create integration object (slice geometry at point/normal)
                try:
                    integral = get_integral(reader_3d, points[i], normals[i])
                except Exception:
                    continue

                # integrate all output arrays
                for name in res_names:
                    reader_1d.GetPointData().GetArray(name).SetValue(i, integral.evaluate(name))
                reader_1d.GetPointData().GetArray('area').SetValue(i, integral.area())
    write_geo(fpath_out, reader_1d)


def main(db, geometries):
    """
    Loop all geometries
    """
    for geo in geometries:
        print('Running geometry ' + geo)

        fpath_1d = db.get_centerline_path(geo)
        fpath_3d = db.get_volume(geo)
        fpath_out = db.get_3d_flow(geo)

        # if os.path.exists(db.get_3d_flow_path_oned_vtp(geo)):
        #     print('  Already exists. Skipping...')
        #     continue

        if not os.path.exists(fpath_1d) or not os.path.exists(fpath_3d):
            continue

        # extract 3d results integrated over cross-section
        try:
            extract_results(fpath_1d, fpath_3d, fpath_out)
        except Exception as e:
            print(e)
            continue


if __name__ == '__main__':
    descr = 'Extract 3d-results at 1d-locations'
    d, g, _ = input_args(descr)
    main(d, g)

