#!/usr/bin/env python

import numpy as np
import os
import pdb

from scipy.interpolate import CubicSpline, interp1d
from scipy.optimize import minimize

import vtk
from vtk.util.numpy_support import vtk_to_numpy as v2n
from vtk.util.numpy_support import numpy_to_vtk as n2v

from get_database import input_args, Database, Post, SimVascular
from vtk_functions import read_geo, write_geo
from get_bc_integrals import integrate_surfaces, integrate_bcs

import matplotlib.pyplot as plt


def fourier(x, n_sample_freq=128):
    """
    Inverse fourier transformation from frequencies in x (real, imaginary)
    """
    assert x.shape[0] % 2 == 0, 'odd number of parameters'
    n_mode = x.shape[0] // 2

    x_complex = x[:n_mode] + 1j * x[n_mode:]
    inflow_fft = np.zeros(n_sample_freq + 1, dtype=complex)
    inflow_fft[:n_mode] = x_complex

    return np.fft.irfft(inflow_fft)


def error(time, inflow, time_smooth, inflow_smooth):
    """
    Get error between input inflow and smooth inflow
    """
    # repeat last value at the start
    time_smooth = np.insert(time_smooth, 0, 0)
    inflow_smooth = np.insert(inflow_smooth, 0, inflow_smooth[-1])

    # interpolate to coarse time
    inflow_interp = interp1d(time_smooth, inflow_smooth)(time)

    return np.sqrt(np.sum((inflow - inflow_interp) ** 2))


def optimize_inflow(time, inflow, n_sample_real=256):
    """
    Optimize fourier-smoothed inflow to interpolate input inflow
    """
    assert n_sample_real % 2 == 0, 'odd number of samples'

    # define fourier smoothing
    n_sample_freq = n_sample_real // 2
    n_mode = 10
    debug = False

    # insert last 3d time step as 1d initial condition (periodic solution)
    time = np.insert(time, 0, 0)
    inflow = np.insert(inflow, 0, inflow[-1])

    # linearly interpolate at fine time points
    time_smooth = np.linspace(0, time[-1], n_sample_real + 1)[1:]
    inflow_interp_lin = interp1d(time, inflow)(time_smooth)

    # get starting value from fft
    inflow_fft = np.fft.rfft(inflow_interp_lin)
    x0 = inflow_fft[:n_mode]
    x0_split = np.array(np.hstack((np.real(x0), np.imag(x0))))

    # setup otimization problem
    run = lambda x: error(time, inflow, time_smooth, fourier(x, n_sample_freq))

    # optimize frequencies to match inflow profile
    res = minimize(run, x0_split, tol=1.0e-8, options={'disp': debug})
    inflow_smooth = fourier(res.x, n_sample_freq)

    # add time step zero
    time_smooth = np.insert(time_smooth, 0, 0)
    inflow_smooth = np.insert(inflow_smooth, 0, inflow_smooth[-1])

    # re-sample to n_sample_real
    time_out = np.linspace(0, time_smooth[-1], n_sample_real)
    inflow_out = interp1d(time_smooth, inflow_smooth)(time_out)

    if debug:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(dpi=300, figsize=(12, 6))
        ax.plot(time_smooth[1:], fourier(x0_split), 'b-')
        ax.plot(time_smooth, inflow_smooth, 'r-')
        ax.plot(time, inflow, 'kx')
        plt.grid()
        plt.show()

    return time_out, inflow_out


def read_velocity(f_dat):
    """
    Read velocity, time steps, and node ids from bct.dat
    """
    # read text file
    with open(f_dat) as f:
        lines = f.readlines()

    # get number of points and time steps
    n_p, n_t = (int(l) for l in lines[0].strip().split())

    # read points
    vel = []
    time = []
    points = []
    coords = []
    for i in range(n_p):
        # line of point header
        split = lines[1 + i * (n_t + 1)].strip().split()

        # point coordinates
        coords += [[float(split[i]) for i in range(3)]]

        # point id
        points += [int(split[-1])]

        # read time steps
        vel_p = []
        for j in range(n_t):
            # line of time step
            split = lines[2 + i + i * n_t + j].split()

            # velocity vector
            vel_p += [[float(split[i]) for i in range(3)]]

            # time
            if i == 0:
                time += [float(split[-1])]

        vel += [vel_p]
    return np.array(vel), np.array(time), np.array(points), np.array(coords)


def write_velocity(f_dat, vel, time, points, coords):
    """
    Write bct.dat file
    """
    # get dimensions
    n_p, n_t, dim = vel.shape

    assert n_p == points.shape[0], 'number of points mismatch'
    assert n_p == coords.shape[0], 'number of coordinates mismatch'
    assert n_t == time.shape[0], 'number of time steps mismatch'
    assert dim == 3, 'number of dimensions mismatch'
    assert coords.shape[1] == 3, 'number of dimensions mismatch'

    with open(f_dat, 'w+') as f:
        # write header
        f.write(str(n_p) + ' ' + str(n_t) + '\n')

        # write points
        for i in range(n_p):
            # write point
            for j in range(3):
                f.write("{:.6e}".format(coords[i, j]) + ' ')
            f.write(str(n_t) + ' ' + str(points[i]) + '\n')

            # write time steps
            np.savetxt(f, np.vstack((vel[i].T, time)).T, fmt='%1.6e')


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


def integrate_inlet(f_in):
    """
    Get inlet flow from bct.dat and bct.vtp
    """
    # read inlet geometry from bct.vtp
    inlet = read_geo(f_in + '.vtp').GetOutput()

    # integrate over inlet
    return integrate_surfaces(inlet, inlet.GetCellData(), 'velocity')


def overwrite_inflow(db, geo, n_sample_real=256):
    """
    Overwrite bct.dat and bct.vtp from svpre with own high-fidelity inflow
    """
    # define project paths
    f_in = os.path.join(db.get_solve_dir_3d(geo), 'bct')

    # read inflow from file
    time, inflow = db.get_inflow(geo)

    # fit inflow using fourier smoothing
    time, inflow = optimize_inflow(time, inflow, n_sample_real)

    # read constant inflow
    vel_dat, time_dat, points, coords = read_velocity(f_in + '.dat')

    # integrate inflow from from bct.dat and bct.vtp
    surf_int = integrate_inlet(f_in)

    # scale velocity
    vel_scaled = vel_dat / surf_int['velocity'] * np.expand_dims(inflow, axis=1)

    # overwrite bct.dat
    write_velocity(f_in + '.dat', vel_scaled, time, points, coords)

    # overwrite bct.vtp
    inlet = read_geo(f_in + '.vtp').GetOutput()
    add_velocity(inlet, vel_scaled, time, points)
    write_geo(f_in + '.vtp', inlet)


def check_inflow(db, geo):
    post = Post()

    # define project paths
    f_in = os.path.join(db.get_solve_dir_3d(geo), 'bct')
    f_out = os.path.join(db.get_solve_dir_3d(geo), geo + '_inflow')

    # read inflow from file
    time, inflow = db.get_inflow(geo)

    # get model inlet from bct.dat and bct.vtp
    surf_int = integrate_inlet(f_in)

    # postproc initial conditions
    sv = SimVascular()
    sv.run_post(db.get_solve_dir_3d(geo), ['-start', '0', '-stop', '0', '-incr', '1', '-vtkcombo', '-vtp', 'initial.vtp'])

    # get initial conditions
    fpath_surf = db.get_surfaces(geo, 'all_exterior')
    f_initial = os.path.join(db.get_solve_dir_3d(geo), 'initial.vtp')
    ini = integrate_bcs(fpath_surf, f_initial, ['pressure', 'velocity'])

    # plot comparison
    fig, ax = plt.subplots(dpi=300, figsize=(12, 6))
    plt.plot(time, inflow * post.convert['flow'], 'kx')
    plt.plot(surf_int['time'], surf_int['velocity'] * post.convert['flow'], 'r-')
    plt.plot(0, ini['velocity'][0][0] * post.convert['flow'], 'bo', fillstyle='none')
    plt.xlabel('Time [s]')
    plt.ylabel('Flow [l/h]')
    plt.grid()
    ax.legend(['OSMSC', 'Optimized for rerun', 'Initial condition for rerun'])
    fig.savefig(f_out, bbox_inches='tight')
    plt.cla()


def main(db, geometries):
    for geo in geometries:
        print('Checking geometry ' + geo)
        check_inflow(db, geo)


if __name__ == '__main__':
    descr = 'Check inlet flow of 3d simulation'
    d, g, _ = input_args(descr)
    main(d, g)