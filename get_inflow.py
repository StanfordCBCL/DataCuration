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


def read_faces(fpath_solve):
    # read geometry
    fname = os.path.join(fpath_solve, 'mesh-complete', 'all_exterior.vtp')
    reader = read_geo(fname).GetOutput()

    # get cell connectivity (can this be done without loops??)
    cells = []
    for i in range(reader.GetNumberOfCells()):
        c = reader.GetCell(i)
        points = []
        for j in range(c.GetNumberOfPoints()):
            points.append(c.GetPointIds().GetId(j))
        cells.append(points)
    cells = np.array(cells)

    # extract face and global element/node IDs
    faces = numpy_support.vtk_to_numpy(reader.GetCellData().GetArray('BC_FaceID'))
    glob_elem_id = numpy_support.vtk_to_numpy(reader.GetCellData().GetArray('GlobalElementID'))
    glob_node_id = numpy_support.vtk_to_numpy(reader.GetPointData().GetArray('GlobalNodeID')).astype(int)

    # store
    face_dic = {'cells': {}, 'points': {}}
    for f in np.unique(faces):
        # select all elements of face
        id_elem = faces == f

        # select all nodes of face
        id_node = np.unique(cells[id_elem])

        # store for later use
        face_dic['cells'][f] = glob_elem_id[id_elem]
        face_dic['points'][f] = glob_node_id[id_node]

    return face_dic


def transfer_solution(fpath_surf, fpath_vol):
    # read surface and volume mesh
    reader_surf, node_surf, cell_surf = read_geo(fpath_surf)
    reader_vol, node_vol, _ = read_geo(fpath_vol)

    # get unique node ids in both meshes
    nd_id_surf = numpy_support.vtk_to_numpy(node_surf.GetArray('GlobalNodeID')).astype(int)
    nd_id_vol = numpy_support.vtk_to_numpy(node_vol.GetArray('GlobalNodeID')).astype(int)

    # map volume mesh to surface mesh
    mask = np.searchsorted(nd_id_vol, nd_id_surf)

    # fields to extract
    res_fields = ['velocity', 'pressure']

    # transfer results from volume mesh to surface mesh
    for i in range(node_vol.GetNumberOfArrays()):
        res_name = node_vol.GetArrayName(i)
        if res_name.split('_')[0] in res_fields:
            # read results from volume mesh
            res = numpy_support.vtk_to_numpy(node_vol.GetArray(res_name))

            # create array to output surface results
            out_array = numpy_support.numpy_to_vtk(res[mask])
            out_array.SetName(res_name)
            node_surf.AddArray(out_array)

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

    # todo: sort according to time
    keys = list(res_faces[1].keys())
    names = []
    times = []
    for n in keys:
        name, time = n.split('_')
        names.append(name)
        times.append(float(time))
    names = np.unique(np.array(names))
    times = np.unique(np.array(times))

    res_faces_array = {'time': times}
    for face, face_res in res_faces.items():
        res_faces_array[face] = {}
        for res_name, res in face_res.items():
            name, time = res_name.split('_')
            if name not in res_faces_array[face]:
                res_faces_array[face][name] = np.zeros(times.shape[0])
            res_faces_array[face][name][np.where(float(time) == times)] = res

    pdb.set_trace()

    # write results
    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetFileName('test.vtp')
    writer.SetInputConnection(reader_surf.GetOutputPort())
    writer.Update()
    writer.Write()

    pdb.set_trace()

    return


def int_surface(fpath_solve, fpath_res):
    fpath_surf = os.path.join(fpath_solve, 'mesh-complete', 'all_exterior.vtp')
    transfer_solution(fpath_surf, fpath_res)

    return
