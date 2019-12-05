#!/usr/bin/env python

import os
import vtk
import pdb

import numpy as np
from vtk.util.numpy_support import numpy_to_vtk as n2v
from vtk.util.numpy_support import vtk_to_numpy as v2n


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


def sort_faces(res_faces):
    # get time steps
    times = []
    for n in res_faces[list(res_faces)[0]].keys():
        times.append(float(n.split('_')[1]))
    times = np.unique(np.array(times))

    # sort data in arrays according to time steps
    res_array = {'time': times}

    for f, f_res in res_faces.items():
        for res_name, res in f_res.items():
            name, time = res_name.split('_')
            if name not in res_array:
                res_array[name] = np.zeros((times.shape[0], max(list(res_faces.keys()))))
            res_array[name][float(time) == times, f - 1] = res

    return res_array


class Integration:
    def __init__(self, int_input):
        # integrate over selected face
        self.integrator = vtk.vtkIntegrateAttributes()
        self.integrator.SetInputData(int_input)
        self.integrator.Update()

    def evaluate(self, field, res_name):
        int_out = self.integrator.GetOutput()

        if field == 'velocity':
            int_name = 'normal_' + res_name
        else:
            int_name = res_name

        # evaluate integral
        integral = v2n(int_out.GetPointData().GetArray(int_name))

        # evaluate surface area
        area = v2n(int_out.GetCellData().GetArray('Area'))[0]

        # choose if integral should be divided by area
        if field == 'velocity':
            out = integral
        else:
            out = integral / area

        return out


def threshold(inp, t):
    thresh = vtk.vtkThreshold()
    thresh.SetInputConnection(inp.GetOutputPort())
    thresh.SetInputArrayToProcess(0, 0, 0, 1, 'BC_FaceID')
    thresh.ThresholdBetween(t, t)
    thresh.Update()
    thresh_node = thresh.GetOutput().GetPointData()
    return thresh, thresh_node


def add_calculator(inp, velocities):
    res_name = velocities[-1]
    calc = vtk.vtkArrayCalculator()
    calc.AddVectorArrayName('Normals')
    calc.AddVectorArrayName(res_name)
    calc.SetInputData(inp.GetOutput())
    calc.SetAttributeModeToUsePointData()
    calc.SetFunction('Normals.' + res_name)
    calc.SetResultArrayName('normal_' + res_name)
    calc.Update()

    velocities.pop()
    if not velocities:
        return calc
    else:
        return add_calculator(calc, velocities)


def integrate_surfaces(reader_surf, cell_surf, res_fields):
    # generate surface normals
    normals = vtk.vtkPolyDataNormals()
    normals.SetInputData(reader_surf.GetOutput())
    normals.Update()

    # get names of all velocity results
    velocities = []
    for i in range(normals.GetOutput().GetPointData().GetNumberOfArrays()):
        res_name = normals.GetOutput().GetPointData().GetArrayName(i)
        if res_name.split('_')[0] == 'velocity':
            velocities.append(res_name)

    # recursively add calculators for normal velocities
    calc = add_calculator(normals, velocities)

    # boundary faces
    faces = np.unique(v2n(cell_surf.GetArray('BC_FaceID')).astype(int))
    res_faces = {}

    # loop boundary faces
    for f in faces:
        # skip face 0
        if f:
            # initialize result dic for face
            res_faces[f] = {}

            # threshhold face
            thresh, thresh_node = threshold(calc, f)

            # integrate over selected face (separately for pressure and velocity)
            integrator = Integration(thresh.GetOutput())

            # get integral for each result
            for i in range(thresh_node.GetNumberOfArrays()):
                res_name = thresh_node.GetArrayName(i)
                field = res_name.split('_')[0]

                # check if field should be added to output
                if field in res_fields:
                    # perform integration
                    out = integrator.evaluate(field, res_name)
                    res_faces[f][res_name] = out

    return sort_faces(res_faces)


def integrate_bcs(fpath_surf, fpath_vol, res_fields, debug=False, debug_out=''):
    # read surface and volume meshes
    reader_surf, node_surf, cell_surf = read_geo(fpath_surf)
    reader_vol, node_vol, _ = read_geo(fpath_vol)

    # transfer solution from volume mesh to surface mesh
    transfer_solution(node_surf, node_vol, res_fields)

    # integrate data on boundary surfaces
    res_faces = integrate_surfaces(reader_surf, cell_surf, res_fields)

    # write results for debugging in paraview
    if debug:
        write_geo(debug_out, reader_surf)
    return res_faces
