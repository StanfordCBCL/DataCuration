#!/usr/bin/env python

import pdb
import vtk

import numpy as np

from vtk.util.numpy_support import numpy_to_vtk as n2v
from vtk.util.numpy_support import vtk_to_numpy as v2n

from get_database import Database, SimVascular, input_args
from vtk_functions import read_geo, write_geo, collect_arrays, threshold, geo
from vmtk import vtkvmtk, vmtkscripts


def split_geo(fpath_surf, fpath_cent, fpath_sect, fpath_vol):
    surf = read_geo(fpath_surf).GetOutput()
    cent = read_geo(fpath_cent).GetOutput()
    sect = read_geo(fpath_sect).GetOutput()
    vol = read_geo(fpath_vol).GetOutput()

    arr_surf = collect_arrays(surf.GetPointData())
    arr_cent = collect_arrays(cent.GetPointData())

    bifurcation_ids = np.unique(arr_cent['BifurcationId']).tolist()
    bifurcation_ids.remove(-1)

    pids = vtk.vtkIdList()
    cids = vtk.vtkIdList()

    # collect points surrounding each bifurcation
    bifurcations = {}
    for bf in bifurcation_ids:
        bifurcations[bf] = {}
        bifurcations[bf]['branches'] = []
        bifurcations[bf]['points_local'] = []
        bifurcations[bf]['points_global'] = []
        for i in range(cent.GetNumberOfPoints()):
            if arr_cent['BifurcationId'][i] == bf:
                cent.GetPointCells(i, cids)
                for j in range(cids.GetNumberOfIds()):
                    cent.GetCellPoints(cids.GetId(j), pids)
                    for k in range(pids.GetNumberOfIds()):
                        br_id = arr_cent['BranchId'][pids.GetId(k)]
                        if br_id != -1:
                            bifurcations[bf]['branches'] += [br_id]
                            bifurcations[bf]['points_local'] += [pids.GetId(k)]
                            bifurcations[bf]['points_global'] += [arr_cent['GlobalNodeId'][pids.GetId(k)]]

    print(bifurcations)

    branch_ids = arr_surf['BranchId']
    bifurcation_ids = arr_surf['BifurcationId']

    # distance array used for clipping
    distance = -1 * np.ones(surf.GetNumberOfPoints())

    # loop bifurcations
    for b, bf in bifurcations.items():
        # loop attached branches
        for i, (j, p, br) in enumerate(zip(bf['points_local'], bf['points_global'], bf['branches'])):
            print(i, j, p)
            # pick slice separating bifurcation and branch
            sliced = threshold(sect, p, 'GlobalNodeId').GetOutput()

            # signed distance from slice
            dist = vtk.vtkDistancePolyDataFilter()
            dist.SetInputData(0, surf)
            dist.SetInputData(1, geo(sliced))
            dist.Update()
            dist = v2n(dist.GetOutput().GetPointData().GetArray('Distance'))

            # reverse inlet
            if i > 0:
                dist *= -1

            # only overwrite where indicator is one: overlap of attached branch and bifurcation
            indicator = (branch_ids == br) * (bifurcation_ids == b)

            distance[indicator] = dist[indicator]

    cut_name = 'distance_bifurcation'

    out = vtk.vtkDoubleArray()
    out.SetNumberOfValues(surf.GetNumberOfPoints())
    out.SetName(cut_name)
    surf.GetPointData().AddArray(out)
    for i in range(surf.GetNumberOfPoints()):
        out.SetValue(i, distance[i])

    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetFileName('/home/pfaller/test.vtp')
    # writer.SetInputData(clip.GetOutput(0))
    writer.SetInputData(surf)
    writer.Update()
    writer.Write()

    import sys
    sys.exit(0)


def main(db, geometries):
    for geo in geometries:
        fpath_cent = db.get_centerline_path(geo)
        fpath_surf = db.get_surfaces_grouped_path(geo)
        fpath_sect = db.get_section_path(geo)
        fpath_vol = db.get_volume(geo)

        split_geo(fpath_surf, fpath_cent, fpath_sect, fpath_vol)


if __name__ == '__main__':
    descr = 'Split geometry in branches and bifurcation_ids'
    d, g, _ = input_args(descr)
    main(d, g)
