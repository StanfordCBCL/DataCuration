#!/usr/bin/env python
import vtk
import os
import pdb
import numpy as np

from vtk.util.numpy_support import vtk_to_numpy as v2n


from get_bc_integrals import get_res_names
from get_database import Database
from vtk_functions import read_geo, threshold, calculator, cut_plane, connectivity, Integration


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
    # pdb.set_trace()

    # sort data in arrays according to time steps
    res_array = {'time': times}

    for f, f_res in res_faces.items():
        for res_name, res in f_res.items():
            name, time = res_name.split('_')
            if name not in res_array:
                res_array[name] = {}
            if f not in res_array[name]:
                res_array[name][f] = np.zeros((times.shape[0], len(res)))
            res_array[name][f][float(time) == times] = res

    return res_array


def get_integral(inp, origin, normal):
    """
    Slice simulation at certain plane and integrate
    Args:
        inp: vtk InputConnection
        origin: plane origin
        normal: plane normal

    Returns:
        Integration object
    """
    # cut geometry
    cut = cut_plane(inp, origin, normal)

    # extract region closest to centerline if there are several slices
    con = connectivity(cut, origin)

    # recursively add calculators for normal velocities
    calc = con
    for v in get_res_names(inp, 'velocity'):
        fun = '(iHat*'+repr(normal[0])+'+jHat*'+repr(normal[1])+'+kHat*'+repr(normal[2])+').' + v
        calc = calculator(calc, fun, [v], 'normal_' + v)

    return Integration(calc)


def extract_results(fpath_1d, fpath_3d):
    """
    Extract 3d results at 1d model nodes (integrate over cross-section)
    Args:
        fpath_1d: path to 1d model
        fpath_3d: path to 3d simulation results

    Returns:
        res: dictionary of results in all branches, in all segments for all result arrays
    """
    if not os.path.exists(fpath_1d) or not os.path.exists(fpath_3d):
        return None

    # fields to extract
    res_fields = ['pressure', 'velocity']

    # read 1d and 3d model
    reader_1d, _, cell_1d = read_geo(fpath_1d)
    reader_3d, _, _ = read_geo(fpath_3d)

    # get all result array names
    res_names = get_res_names(reader_3d, res_fields)

    # group ids in model
    groups = np.unique(v2n(cell_1d.GetArray('group')))

    # initialize output array
    res = {}
    for g in groups:
        res[g] = {}
        for r in res_names:
            res[g][r] = []

    # create integrals
    for g in groups:
        print('  group ' + repr(g))

        # threshold vessel branch
        thresh = threshold(reader_1d, g, 'group')

        # get point and normal on each 1d node
        points = v2n(thresh.GetOutput().GetPoints().GetData())
        normals = v2n(thresh.GetOutput().GetPointData().GetArray('normals'))

        for i, (p, n) in enumerate(zip(points[1:], normals[1:])):
            # create integration object
            integral = get_integral(reader_3d, p, n)

            # integrate all output arrays
            for r in res_names:
                res[g][r] += [integral.evaluate(r)]

    return res


def main():
    """
    Loop all geometries
    """
    db = Database()

    for geo in ['0071_0001']:
    # for geo in db.get_geometries():
        print('Running geometry ' + geo)

        fpath_1d = db.get_1d_geo(geo)
        fpath_3d = db.get_volume(geo)

        # extract 3d results integrated over cross-section
        res = extract_results(fpath_1d, fpath_3d)

        # save to file
        np.save(db.get_3d_flow_path(geo), sort_faces(res))


if __name__ == '__main__':
    main()
