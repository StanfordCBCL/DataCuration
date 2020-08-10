#!/usr/bin/env python

import os
import vtk
import argparse
import pdb

import numpy as np

from get_database import input_args, Database, Post
from get_mean_flow_3d import extract_results


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
        fpath_1d = db.get_centerline_path(geo)
        fpath_vol = db.get_res_3d_vol_rerun(geo)
        fpath_out = db.get_3d_flow_rerun(geo)
        if not os.path.exists(fpath_out):
            continue

        print('Processing ' + geo)
        extract_results(fpath_1d, fpath_vol, fpath_out, only_caps=True)


if __name__ == '__main__':
    descr = 'Plot comparison of xd-results'
    d, g, _ = input_args(descr)
    main(d, g)
