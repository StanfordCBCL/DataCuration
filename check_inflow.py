#!/usr/bin/env python

import numpy as np
import os
import pdb

import vtk
from vtk.util.numpy_support import vtk_to_numpy as v2n
from vtk.util.numpy_support import numpy_to_vtk as n2v

from get_database import input_args, Database, Post, SimVascular
from vtk_functions import read_geo, write_geo
from get_bc_integrals import integrate_surfaces, integrate_bcs

import matplotlib.pyplot as plt


def read_velocity(f_dat):
    """
    Read velocity, time steps, and node ids from bct.dat
    """
    # read text file
    with open(f_dat) as f:
        lines = f.readlines()

    # get number of points and time steps
    n_p, n_t = (int(l) for l in lines[0].strip().split())

    # extract information
    vel = []
    time = []
    points = []
    for i in range(n_p):
        # index of point header
        k_p = 1 + i * (n_t + 1)

        # point id
        points += [int(lines[k_p].strip().split()[-1])]

        vel_p = []
        for j in range(n_t):
            # index of time step
            k_t = 2 + i + i * n_t + j

            split = lines[k_t].split()

            # velocity vector
            vel_p += [[float(split[i]) for i in range(3)]]

            # time
            if i == 0:
                time += [float(split[-1])]

        vel += [vel_p]
    return np.array(vel), np.array(time), np.array(points)


def add_velocity(inlet, vel, time, points):
    """
    Add velocity vectors to inlet geometry bct.vtp
    """
    # get unique point ids
    ids = v2n(inlet.GetPointData().GetArray('GlobalNodeID'))

    # remove all point arrays except GLobalNodeId
    names = [inlet.GetPointData().GetArrayName(i) for i in range(inlet.GetPointData().GetNumberOfArrays())]
    for n in names:
        if n != 'GlobalNodeID':
            inlet.GetPointData().RemoveArray(n)

    # add velocity vectors to nodes
    for i, t in enumerate(time):
        # create new array for time step
        array = vtk.vtkDoubleArray()
        array.SetNumberOfComponents(3)
        array.SetNumberOfTuples(vel.shape[0] * 3)
        array.SetName('velocity_' + str(t))
        inlet.GetPointData().AddArray(array)

        # fill array
        for j, p in enumerate(points):
            k = np.where(ids == p)[0][0]
            v = vel[j, i]
            array.SetTuple3(k, v[0], v[1], v[2])


def integrate_inlet(f_in, f_out):
    """
    Get inlet flow from bct.dat and bct.vtp
    """
    # read inlet geometry from bct.vtp
    inlet = read_geo(f_in + '.vtp').GetOutput()

    # read information from bct.dat
    vel, time, points = read_velocity(f_in + '.dat')

    # add velocity vectors to bct.vtp
    add_velocity(inlet, vel, time, points)

    # export geometry
    write_geo(f_out + '.vtp', inlet)

    # integrate over inlet
    return integrate_surfaces(inlet, inlet.GetCellData(), 'velocity')


def main(db, geometries):
    post = Post()

    for geo in geometries:
        print('Checking geometry ' + geo)

        # define project paths
        f_in = os.path.join(db.get_solve_dir_3d(geo), 'bct')
        f_out = os.path.join(db.get_solve_dir_3d(geo), 'inflow')

        # read inflow from file
        time, inflow = db.get_inflow(geo)

        # get model inlet from bct.dat and bct.vtp
        surf_int = integrate_inlet(f_in, f_out)

        # postproc initial conditions
        sv = SimVascular()
        sv.run_post(db.get_solve_dir_3d(geo), ['-start', '0', '-stop', '0', '-incr', '1', '-vtkcombo', '-vtp', 'initial.vtp'])

        # get initial conditions
        fpath_surf = db.get_surfaces(geo, 'all_exterior')
        f_initial = os.path.join(db.get_solve_dir_3d(geo), 'initial.vtp')
        ini = integrate_bcs(fpath_surf, f_initial, ['pressure', 'velocity'])

        # plot comparison
        fig, ax = plt.subplots(dpi=300, figsize=(12, 6))
        plt.plot(time, inflow * post.convert['flow'])
        plt.plot(surf_int['time'], surf_int['velocity'] * post.convert['flow'])
        plt.plot(0, ini['velocity'][0][0] * post.convert['flow'], 'ro', fillstyle='none')
        plt.xlabel('Time [s]')
        plt.ylabel('Flow [l/h]')
        plt.grid()
        ax.legend(['OSMSC', 'Rerun', 'Initial condition'])
        fig.savefig(f_out, bbox_inches='tight')
        plt.cla()


if __name__ == '__main__':
    descr = 'Check inlet flow of 3d simulation'
    d, g, _ = input_args(descr)
    main(d, g)
