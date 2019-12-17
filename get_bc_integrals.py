#!/usr/bin/env python

import os
import vtk
import pdb

import numpy as np
from vtk.util.numpy_support import numpy_to_vtk as n2v
from vtk.util.numpy_support import vtk_to_numpy as v2n

from get_database import Database


def read_geo(fname):
    """
    Read geometry from file, chose corresponding vtk reader
    Args:
        fname: vtp surface or vtu volume mesh

    Returns:
        vtk reader, point data, cell data
    """
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
    """
    Write geometry to file
    Args:
        fname: file name
        reader: vtkXMLPolyData
    """
    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetFileName(fname)
    writer.SetInputConnection(reader.GetOutputPort())
    writer.Update()
    writer.Write()


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


def sort_faces(res_faces):
    """
    Arrange results from surface integration in matrix
    Args:
        res_faces: dictionary where each key is a result at a certain time step

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

    for f, f_res in res_faces.items():
        for res_name, res in f_res.items():
            name, time = res_name.split('_')
            if name not in res_array:
                res_array[name] = np.zeros((times.shape[0], max(list(res_faces.keys()))))
            res_array[name][float(time) == times, f - 1] = res

    return res_array


class Integration:
    """
    Class to perform integration on surface caps
    """

    def __init__(self, int_input):
        # integrate over selected face
        self.integrator = vtk.vtkIntegrateAttributes()
        self.integrator.SetInputData(int_input)
        self.integrator.Update()

    def evaluate(self, field, res_name):
        """
        Evaluate integral.
        Distinguishes between scalar integration (e.g. pressure) and normal projection (velocity)
        Optionally divides integral by integrated area
        Args:
            field: pressure, velocity, ...
            res_name: name of array

        Returns:
            Scalar integral
        """
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
    """
    Threshold according to BC_FaceID to extract caps
    Args:
        inp: InputConnection
        t: BC_FaceID

    Returns:
        reader, point data
    """
    thresh = vtk.vtkThreshold()
    thresh.SetInputConnection(inp.GetOutputPort())
    thresh.SetInputArrayToProcess(0, 0, 0, 1, 'BC_FaceID')
    thresh.ThresholdBetween(t, t)
    thresh.Update()
    thresh_node = thresh.GetOutput().GetPointData()
    return thresh, thresh_node


def add_calculator(inp, velocities):
    """
    Recursive function to add projection of time step velocities onto cap normals
    Args:
        inp: InputConnection
        velocities: list of velocity array names at time steps, e.g. velocity_0.7500

    Returns:
        Final calculator object
    """
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

    # get names of all velocity results
    velocities = []
    for i in range(normals.GetOutput().GetPointData().GetNumberOfArrays()):
        res_name = normals.GetOutput().GetPointData().GetArrayName(i)
        if 'velocity' in res_name:
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


def integrate_bcs(db, geo, res_fields, debug=False, debug_out=''):
    """
    Perform all steps necessary to get results averaged on caps
    Args:
        db: Database object
        geo: geometry name
        res_fields: results to extract
        debug: bool if debug geometry should be written
        debug_out: path for debug geometry

    Returns:
        dictionary with result fields as keys and matrices with all faces and time steps as matrices
    """
    fpath_surf = db.get_surfaces(geo, 'all_exterior')
    fpath_vol = db.get_volume(geo)

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


def main():
    """
    Loop all geometries in database
    """
    # create object for data base entry to handle names/paths
    db = Database()

    for geo in db.get_geometries():
        print('Processing ' + geo)

        bc_flow = integrate_bcs(db, geo, ['pressure', 'velocity'])

        if bc_flow is not None:
            np.save(db.get_bc_flow_path(geo), bc_flow)


if __name__ == '__main__':
    main()
