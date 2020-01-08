#!/usr/bin/env python

import os
import vtk
import argparse
import pdb

import numpy as np

from vtk.util.numpy_support import numpy_to_vtk as n2v
from vtk.util.numpy_support import vtk_to_numpy as v2n

from get_database import Database
from vtk_functions import Integration, read_geo, write_geo, threshold, calculator, cut_plane


def transfer_solution(node_surf, node_vol, res_fields):
    """
    Transfer point data from volume mesh to surface mesh using GlobalNodeID
    Args:
        node_surf: surface mesh
        node_vol: volume mesh
        res_fields: point data names to transfer
    """
    # get global node ids in both meshes
    nd_id_surf = v2n(node_surf.GetArray('GlobalNodeID')).astype(int)
    nd_id_vol = v2n(node_vol.GetArray('GlobalNodeID')).astype(int)

    # map volume mesh to surface mesh
    mask = np.searchsorted(nd_id_vol, nd_id_surf)

    # transfer results from volume mesh to surface mesh
    for i in range(node_vol.GetNumberOfArrays()):
        res_name = node_vol.GetArrayName(i)
        if res_name.split('_')[0] in res_fields:
            # read results from volume mesh
            res = v2n(node_vol.GetArray(res_name))

            # create array to output surface mesh results
            out_array = n2v(res[mask])
            out_array.SetName(res_name)
            node_surf.AddArray(out_array)


def sort_faces(res_faces, area):
    """
    Arrange results from surface integration in matrix
    Args:
        res_faces: dictionary with key: cap id, value: result at a certain time step
        area: cross-sectional area of each cap

    Returns:
        dictionary with results as keys and matrix at all surfaces/time steps as values
    """
    # get time steps
    times = []
    for n in res_faces[list(res_faces)[0]].keys():
        times.append(float(n.split('_')[1]))
    times = np.unique(np.array(times))

    # sort data in arrays according to time steps
    res_array = {'time': times}
    dim = (times.shape[0], max(list(res_faces.keys())))

    for f, f_res in res_faces.items():
        for res_name, res in f_res.items():
            name, time = res_name.split('_')
            if name not in res_array:
                res_array[name] = np.zeros(dim)
            res_array[name][float(time) == times, f - 1] = res

    # repeat area for all time steps to match format
    res_array['area'] = np.zeros(dim)
    for f, f_res in area.items():
        res_array['area'][:, f - 1] = f_res

    return res_array


def get_res_names(inp, res_fields):
    # result name list
    res = []

    # get integral for each result
    for i in range(inp.GetOutput().GetPointData().GetNumberOfArrays()):
        res_name = inp.GetOutput().GetPointData().GetArrayName(i)
        field = res_name.split('_')[0]

        # check if field should be added to output
        if field in res_fields:
            res += [res_name]

    return res


def integrate_surfaces(reader_surf, cell_surf, res_fields):
    """
    Integrate desired fields on all caps of surface mesh (as defined by BC_FaceID)
    Args:
        reader_surf: reader for surface mesh
        cell_surf: surface mesh cell data
        res_fields: result fields to extract

    Returns:
        dictionary with result fields as keys and matrices with all faces and time steps as matrices
    """
    # generate surface normals
    normals = vtk.vtkPolyDataNormals()
    normals.SetInputData(reader_surf.GetOutput())
    normals.Update()

    # recursively add calculators for normal velocities
    calc = normals
    for v in get_res_names(reader_surf, 'velocity'):
        calc = calculator(calc, 'Normals.' + v, ['Normals', v], 'normal_' + v)

    # get all output array names
    res_names = get_res_names(reader_surf, res_fields)

    # boundary faces
    faces = np.unique(v2n(cell_surf.GetArray('BC_FaceID')).astype(int))
    res = {}
    area = {}

    # loop boundary faces
    for f in faces:
        # skip face 0 (vessel wall)
        if f:
            # threshhold face
            thresh = threshold(calc, f, 'BC_FaceID')

            # integrate over selected face (separately for pressure and velocity)
            integrator = Integration(thresh)

            # perform integration
            res[f] = {}
            for r in res_names:
                res[f][r] = integrator.evaluate(r)

            # store cross-sectional area
            area[f] = integrator.area()

    return sort_faces(res, area)


def integrate_bcs(fpath_surf, fpath_vol, res_fields, debug=False, debug_out=''):
    """
    Perform all steps necessary to get results averaged on caps
    Args:
        fpath_surf: surface geometry file
        fpath_vol: volume geometry file
        res_fields: results to extract
        debug: bool if debug geometry should be written
        debug_out: path for debug geometry

    Returns:
        dictionary with result fields as keys and matrices with all faces and time steps as matrices
    """
    if not os.path.exists(fpath_surf) or not os.path.exists(fpath_vol):
        return None

    # read surface and volume meshes
    reader_surf, node_surf, cell_surf = read_geo(fpath_surf)
    _, node_vol, _ = read_geo(fpath_vol)

    # transfer solution from volume mesh to surface mesh
    transfer_solution(node_surf, node_vol, res_fields)

    # integrate data on boundary surfaces
    res_faces = integrate_surfaces(reader_surf, cell_surf, res_fields)

    # write results for debugging in paraview
    if debug:
        write_geo(debug_out, reader_surf)
    return res_faces


def main(param):
    """
    Loop all geometries in database
    """
    # get model database
    db = Database(param.study)

    # choose geometries to evaluate
    if param.geo:
        geometries = [param.geo]
    elif param.geo == 'select':
        geometries = db.get_geometries_select()
    else:
        geometries = db.get_geometries()

    for geo in geometries:
        print('Processing ' + geo)

        # file paths
        fpath_surf = db.get_surfaces(geo, 'all_exterior')
        fpath_vol = db.get_volume(geo)

        bc_flow = integrate_bcs(fpath_surf, fpath_vol, ['pressure', 'velocity'])

        if bc_flow is not None:
            np.save(db.get_bc_flow_path(geo), bc_flow)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract 3d-results at 1d-locations')
    parser.add_argument('-g', '--geo', help='geometry')
    parser.add_argument('-s', '--study', help='study name')
    main(parser.parse_args())
