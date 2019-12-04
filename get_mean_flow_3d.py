#!/usr/bin/env python

from vtk.util import numpy_support
import numpy as np
import os, glob, csv
import scipy.spatial.distance

import paraview.simple as pv 

from vtk.util.numpy_support import vtk_to_numpy as v2n

import pdb, readline

mode = 'volume'

fpath_res_3d = '/home/pfaller/work/simvascular_demo/SVProject/Simulations/demojob/demojob-converted-results'
fname_res_3d = 'all_results_*.vtu'
fname_cent = '/home/pfaller/work/simvascular_demo/SVProject/1d/centerlines.vtp'

fpath_res_1d = '/home/pfaller/work/simvascular_demo/SVProject/1d'
fname_res_1d = 'demoGroup*_flow.dat'

eps = 1e-3

# todo: read from file
nodes = [[-1.9641913175582886, -1.634276032447815, 12.921036720275879],
         [0.6680281162261963, 3.1779656410217285, -7.453646183013916],
         [-5.755790710449219, 5.635354518890381, -20.655569076538086],
         [8.165690422058105, 5.082368850708008, -19.949831008911133]]
nodes = np.array(nodes)
caps = [0, 2, 3]
normals =[[-0.449216, -0.382102, 0.807591],
          [-0.303607, 0.336595, -0.891362],
          [0.312343, 0.397111, -0.862986]]
normals = np.array(normals)
orientation = [-1, 1, 1]


# return list of result file paths
def get_res(fpath, fname):
    res_list = glob.glob(os.path.join(fpath, fname))
    res_list.sort()
    return res_list


# normalize each row of a matrix
def norm_row(x):
    # length of each row
    length = np.sum(np.abs(x)**2,axis=-1)**(1./2)

    # x is vector, length is scalar
    if not length.shape:
        return x / length

    # x is matrix, length is vector
    else:
        return x / length[:, np.newaxis]


# slice simulation at certain location
def get_slice(input, origin, normal):
    slice = pv.Slice(Input=input)
    slice.SliceType = 'Plane'
    slice.SliceType.Origin = origin.tolist()
    slice.SliceType.Normal = normal.tolist()
    return slice


def main():
    result_list_1d = get_res(fpath_res_1d, fname_res_1d)

    # read 1D simulation results
    results_1d = []

    # loop segments
    for f_res in result_list_1d:
        with open(f_res, 'r') as f:
            reader = csv.reader(f, delimiter=' ')

            # loop nodes
            results_1d_f = []
            for line in reader:
                results_1d_f.append([float(l) for l in line if l][1:])
            results_1d.append(np.array(results_1d_f))

    # get all time steps
    result_list_3d = get_res(fpath_res_3d, fname_res_3d)

    # read 3D simulation results
    results_3d = pv.XMLUnstructuredGridReader(FileName=result_list_3d)

    # number of segments
    n_s = len(caps)

    # create integrals
    integrals = []
    for i in range(n_s):
        n = normals[i]
        n /= np.linalg.norm(n)
        p = nodes[caps[i]] - eps * n
        n *= orientation[i]

        # slice geometry
        calc_input = get_slice(results_3d, p, n)

        # project velocity on segment direction
        calc = 'velocity.(iHat*'+repr(n[0])+'+jHat*'+repr(n[1])+'+kHat*'+repr(n[2])+')'
        calculator = pv.Calculator(Input=calc_input)
        calculator.Function = calc
        calculator.ResultArrayName = 'velocity_normal'

        # only select closest slice in case it cuts the geometry multiple times
        connectivity = pv.Connectivity(Input=calculator)
        connectivity.ExtractionMode = 'Extract Closest Point Region'
        connectivity.ClosestPoint = p.tolist()

        # integrate slice
        integral = pv.IntegrateVariables(Input=connectivity)

        integrals.append(integral)

    # loop time steps
    times = results_3d.TimestepValues
    n_t = len(times)

    # initialize results array
    pressure = np.zeros((n_t, len(integrals)))
    flow = np.zeros((n_t, len(integrals)))

    for t in range(n_t):
        for i in range(len(integrals)):
            # get current time step
            integrals[i].UpdatePipeline(times[t])

            # evaluate integral
            integral = pv.paraview.servermanager.Fetch(integrals[i])

            norm = v2n(integral.GetCellData().GetArray('Area'))[0]
            pres = v2n(integral.GetPointData().GetArray('pressure'))[0]
            velo = v2n(integral.GetPointData().GetArray('velocity_normal'))[0]

            # normalize by volume
            pressure[t, i] = pres / norm
            flow[t, i] = velo

    results = {'flow': flow, 'pressure': pressure}
    np.save(os.path.join(fpath_res_3d, 'results_avg'), results)


if __name__ == '__main__':
    main()
