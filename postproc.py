#!/usr/bin/env python

import os
import vtk
import argparse
import pdb

import numpy as np

from vtk.util.numpy_support import numpy_to_vtk as n2v
from vtk.util.numpy_support import vtk_to_numpy as v2n

from get_database import input_args, Database, Post
from get_bc_integrals import integrate_surfaces, transfer_solution
from get_mean_flow_3d import extract_results
from vtk_functions import Integration, read_geo, write_geo, threshold, calculator, cut_plane

import matplotlib.pyplot as plt


def map_meshes(nd_id_src, nd_id_trg):
    """
    Map source mesh to target mesh
    """
    index = np.argsort(nd_id_src)
    search = np.searchsorted(nd_id_src[index], nd_id_trg)
    return index[search]


def transfer_array(node_trg, node_src, name):
    """
    Transfer point data from volume mesh to surface mesh using GlobalNodeID
    Args:
        node_trg: target mesh
        node_src: source mesh
        name: array name to be transfered
    """
    # get global node ids in both meshes
    nd_id_trg = v2n(node_trg.GetPointData().GetArray('GlobalNodeID')).astype(int)
    nd_id_src = v2n(node_src.GetPointData().GetArray('GlobalNodeID')).astype(int)

    # map source mesh to target mesh
    mask = map_meshes(nd_id_src, nd_id_trg)

    # transfer array from volume mesh to surface mesh
    assert node_src.GetPointData().HasArray(name), 'Source mesh has no array ' + name
    res = v2n(node_src.GetPointData().GetArray(name))

    # create array in target mesh
    out_array = n2v(res[mask])
    out_array.SetName(name)
    node_trg.GetPointData().AddArray(out_array)


def main(db, geometries):
    """
    Loop all geometries in database
    """
    for geo in geometries:
        print('Processing ' + geo)

        # paths
        fpath_vol = db.get_res_3d_vol_rerun(geo)
        fpath_surf = db.get_res_3d_surf_rerun(geo)
        fpath_1d = db.get_centerline_path(geo)
        fpath_out = db.get_3d_flow_rerun(geo)
        fpath_surf_ref = db.get_surfaces(geo, 'all_exterior')

        # file paths
        res_fields = ['pressure', 'velocity']

        surf = read_geo(fpath_surf).GetOutput()
        surf_ref = read_geo(fpath_surf_ref).GetOutput()

        # transfer surface ids
        cell_to_point = vtk.vtkCellDataToPointData()
        cell_to_point.SetInputData(surf_ref)
        cell_to_point.Update()

        transfer_array(surf, cell_to_point.GetOutput(), 'BC_FaceID')

        surf_cell = vtk.vtkPointDataToCellData()
        surf_cell.SetInputData(surf)
        surf_cell.PassPointDataOn()
        surf_cell.Update()

        # integrate data on boundary surfaces
        res = integrate_surfaces(surf_cell.GetOutput(), surf_cell.GetOutput().GetCellData(), res_fields)
        res['flow'] = res['velocity']
        del res['velocity']

        # read ordered outlet names from file
        caps = ['inlet']
        with open(db.get_centerline_outlet_path(geo)) as file:
            for line in file:
                caps += line.splitlines()

        post = Post()

        fields = post.fields
        fields.remove('area')

        fig, ax = plt.subplots(dpi=300, figsize=(12, 6))

        n_max = -1

        for f in fields:
            legend = caps
            plt.plot(res['time'][:n_max], res[f][:n_max, :] * post.convert[f])
            if f == 'flow':
                plt.plot(res['time'][:n_max], np.sum(res[f][:n_max, :], axis=1) * post.convert[f])
                legend += ['sum']
            plt.xlabel('Time step [0.4 ms]')
            plt.ylabel(f.capitalize() + ' [' + post.units[f] + ']')
            plt.grid()
            plt.subplots_adjust(right=0.7)
            ax.legend(legend, loc=(1.04, 0.5))
            fig.savefig(os.path.splitext(fpath_surf)[0] + '_' + f + '.png', bbox_inches='tight')
            plt.cla()

        # get 3d results
        extract_results(fpath_1d, fpath_vol, fpath_out)


if __name__ == '__main__':
    descr = 'Plot comparison of xd-results'
    d, g, _ = input_args(descr)
    main(d, g)
