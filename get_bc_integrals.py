#!/usr/bin/env python

import numpy as np
import os, vtk, pdb
from vtk.util import numpy_support


def read_geo(fname):
    _, ext = os.path.splitext(fname)
    if ext == '.vtp':
        reader = vtk.vtkXMLPolyDataReader()
    elif ext == '.vtu':
        reader = vtk.vtkXMLUnstructuredGridReader()
    else:
        raise ValueError('File extension ' + ext + ' unknown.')
    reader.SetFileName(fname)
    reader.Update()
    return reader, reader.GetOutput().GetPointData(), reader.GetOutput().GetCellData()


def write_geo(fname, reader):
    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetFileName(fname)
    writer.SetInputConnection(reader.GetOutputPort())
    writer.Update()
    writer.Write()
    return


def transfer_solution(node_surf, node_vol, res_fields):
    # get unique node ids in both meshes
    nd_id_surf = numpy_support.vtk_to_numpy(node_surf.GetArray('GlobalNodeID')).astype(int)
    nd_id_vol = numpy_support.vtk_to_numpy(node_vol.GetArray('GlobalNodeID')).astype(int)

    # map volume mesh to surface mesh
    mask = np.searchsorted(nd_id_vol, nd_id_surf)

    # transfer results from volume mesh to surface mesh
    for i in range(node_vol.GetNumberOfArrays()):
        res_name = node_vol.GetArrayName(i)
        if res_name.split('_')[0] in res_fields:
            # read results from volume mesh
            res = numpy_support.vtk_to_numpy(node_vol.GetArray(res_name))

            # create array to output surface mesh results
            out_array = numpy_support.numpy_to_vtk(res[mask])
            out_array.SetName(res_name)
            node_surf.AddArray(out_array)


def sort_faces(res_faces):
    # get time steps
    times = []
    for n in res_faces[1].keys():
        times.append(float(n.split('_')[1]))
    times = np.unique(np.array(times))

    # sort data in arrays according to time steps
    res_array = {'time': times}
    for f, f_res in res_faces.items():
        res_array[f] = {}
        for res_name, res in f_res.items():
            name, time = res_name.split('_')
            if name not in res_array[f]:
                res_array[f][name] = np.zeros(times.shape[0])
            res_array[f][name][np.where(float(time) == times)] = res

    return res_array


def integrate_surfaces(reader_surf, cell_surf, res_fields):
    # boundary faces
    faces = np.unique(numpy_support.vtk_to_numpy(cell_surf.GetArray('BC_FaceID')).astype(int))
    res_faces = {}

    # loop boundary faces
    for f in faces:
        # skip face 0
        if f:
            # initialize result dic for face
            res_faces[f] = {}

            # threshhold face
            thresh = vtk.vtkThreshold()
            thresh.SetInputConnection(reader_surf.GetOutputPort())
            thresh.SetInputArrayToProcess(0, 0, 0, 1, 'BC_FaceID')
            thresh.ThresholdBetween(f, f)
            thresh.Update()

            # integrate over selected face
            integrate = vtk.vtkIntegrateAttributes()
            integrate.SetInputData(thresh.GetOutput())
            integrate.Update()
            integrate_node = integrate.GetOutput().GetPointData()

            # get integral for each result
            for i in range(integrate_node.GetNumberOfArrays()):
                res_name = integrate_node.GetArrayName(i)
                if res_name.split('_')[0] in res_fields:
                    out = numpy_support.vtk_to_numpy(integrate_node.GetArray(res_name))

                    # export scalar
                    if len(out.shape) == 1 and out.shape[0] == 1:
                        res_faces[f][res_name] = out[0]

                    # export vector norm
                    elif len(out.shape) == 2 and out.shape[1] == 3:
                        res_faces[f][res_name] = np.linalg.norm(out)
                    else:
                        raise ValueError('Unknown shape of array ' + repr(out.shape))

    return sort_faces(res_faces)


def integrate_bcs(fpath_surf, fpath_vol, res_fields):
    # read surface and volume meshes
    reader_surf, node_surf, cell_surf = read_geo(fpath_surf)
    reader_vol, node_vol, _ = read_geo(fpath_vol)

    # transfer solution from volume mesh to surface mesh
    transfer_solution(node_surf, node_vol, res_fields)

    # integrate data on boundary surfaces
    res_faces = integrate_surfaces(reader_surf, cell_surf, res_fields)

    # write results for debugging in paraview
    # write_geo('test.vtp', reader_surf)
    return res_faces
