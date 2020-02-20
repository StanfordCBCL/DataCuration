#!/usr/bin/env python
# coding=utf-8

import pdb
import vtk
import numpy as np

from vtk.util.numpy_support import numpy_to_vtk as n2v
from vtk.util.numpy_support import vtk_to_numpy as v2n

from get_database import Database, SimVascular, Post, input_args
from vtk_functions import read_geo, write_geo, connectivity_all, clean, add_scalars, rename, replace


def preproc_centerline(centerline):
    # cleanup: merge duplicate points
    cleaned = clean(centerline)

    # clipping function using bifurcation array
    centerline.GetOutput().GetPointData().SetActiveScalars('CenterlineSectionBifurcation')
    fun = vtk.vtkImplicitDataSet()
    fun.SetDataSet(centerline.GetOutput())

    # clip centerline in branches and bifurcations
    clip = vtk.vtkClipPolyData()
    clip.SetInputData(cleaned.GetOutput())
    clip.SetClipFunction(fun)
    clip.InsideOutOn()
    clip.SetValue(0)
    clip.GenerateClippedOutputOn()
    clip.Update()

    # enumerate all branches
    branches = connectivity_all(clip.GetOutput(0))
    rename(branches, 'RegionId', 'BranchId')
    add_scalars(branches, 'BifurcationId', -1)

    # enumerate all bifurcations
    bifurcations = connectivity_all(clip.GetOutput(1))
    rename(bifurcations, 'RegionId', 'BifurcationId')
    add_scalars(bifurcations, 'BranchId', -1)

    # bring bifurcations and branches back together
    append = vtk.vtkAppendFilter()
    append.AddInputData(branches.GetOutput())
    append.AddInputData(bifurcations.GetOutput())
    append.MergePointsOn()
    append.Update()

    # convert vtkUnstructerdGrid to vtkPolyData
    geo = vtk.vtkGeometryFilter()
    geo.SetInputData(append.GetOutput())
    geo.Update()

    return geo


def get_connectivity(cent):
    # read arrays from centerline
    bifurcation = v2n(cent.GetOutput().GetPointData().GetArray('BifurcationId'))
    branch = v2n(cent.GetOutput().GetPointData().GetArray('BranchId'))

    # get centerline connectivity: which branches are attached to which bifurcation?
    connectivity = {}
    for c in range(cent.GetOutput().GetNumberOfCells()):
        ele = cent.GetOutput().GetCell(c)
        point_ids = np.array([ele.GetPointIds().GetId(i) for i in range(ele.GetPointIds().GetNumberOfIds())])
        br = branch[point_ids].tolist()

        # find cells that are at borders of bifurcations (more two unique RegionIds)
        if np.unique(br).shape[0] == 2:
            # remove branch RegionId
            assert -1 in br,  'Centerline inconsistent'

            # element ids of branch and bifurcation
            i_bf_ele = br.index(-1)
            i_br_ele = int(not i_bf_ele)

            # global ids of branch and bifurcation
            i_bf = bifurcation[point_ids[i_bf_ele]]
            i_br = branch[point_ids[i_br_ele]]

            assert i_bf != -1, 'Centerline incosistent'
            assert i_br != -1, 'Centerline incosistent'

            # store unique branch id
            if i_bf in connectivity:
                if i_br not in connectivity[i_bf]:
                    connectivity[i_bf] += [i_br]
            else:
                connectivity[i_bf] = [i_br]

    return clean_connectivity(cent, connectivity)


def clean_connectivity(cent, connectivity):
    # get arrays from centerline
    bifurcation = v2n(cent.GetOutput().GetCellData().GetArray('BifurcationId'))
    branch = v2n(cent.GetOutput().GetCellData().GetArray('BranchId'))
    blanking = v2n(cent.GetOutput().GetCellData().GetArray('Blanking'))
    group = v2n(cent.GetOutput().GetCellData().GetArray('GroupIds'))

    # all bifurcations
    bifurcations = np.unique(bifurcation).tolist()
    bifurcations.remove(-1)

    # store blanking groups for each bifurcation
    bifurcation_groups = {}
    for bf in bifurcations:
        bifurcation_groups[bf] = []
    for bf in bifurcations:
        for g in np.unique(group[bifurcation == bf]):
            if g in np.unique(group[blanking == 1]):
                bifurcation_groups[bf] += [g]

    # do the actual cleanup
    for bf, g in bifurcation_groups.items():
        # find fake bifurcations (within branch): contains no bifurcation group
        if not g:
            assert len(connectivity[bf]) == 2, 'Centerline inconsistent'

            # joint branch number for fake inlet, fake outlet, and fake bifurcation
            branch[branch == connectivity[bf][1]] = connectivity[bf][0]
            branch[bifurcation == bf] = connectivity[bf][0]

            # remove bifurcation
            bifurcation[bifurcation == bf] = -1
            del connectivity[bf]
            continue

        # find joint bifurcations: same bifurcation group present in muliple bifurcations
        for bfbf, gg in bifurcation_groups.items():
            if bfbf != bf and gg == g and bf in connectivity:
                # joint bifurcation number
                bifurcation[bifurcation == bfbf] = bf
                connectivity[bf] += connectivity[bfbf]
                del connectivity[bfbf]

    # store old (remaining) numbering of bifurcations and branches
    branches_old = []
    bifurcations_old = []
    for bf, branches in connectivity.items():
        bifurcations_old += [bf]
        branches_old += branches
    bifurcations_old.sort()
    branches_old = np.unique(branches_old).tolist()

    # number bifurcations and branches consecutively
    connectivity_clean = {}
    for bf_new, bf_old in enumerate(bifurcations_old):
        assert len(connectivity[bf_old]) >= 3, 'Centerline inconsistent: Fake bifurcation exists'
        bifurcation[bifurcation == bf_old] = bf_new
        connectivity_clean[bf_new] = []
        for br_old in connectivity[bf_old]:
            br_new = branches_old.index(br_old)
            branch[branch == br_old] = br_new
            connectivity_clean[bf_new] += [br_new]

    # replace arrays
    assert not np.any(branch[bifurcation == -1] == -1), 'branch array inconsistent'
    assert not np.any(bifurcation[branch == -1] == -1), 'bifurcation array inconsistent'
    replace(cent, 'BranchId', branch)
    replace(cent, 'BifurcationId', bifurcation)

    # remove outdated arrays
    cent.GetOutput().GetPointData().RemoveArray('BranchId')
    cent.GetOutput().GetPointData().RemoveArray('BifurcationId')

    write_geo('test.vtp', cent)
    return connectivity_clean


def get_model(centerline):
    # read centerline from file
    cent_raw, _, _ = read_geo(centerline)

    # centerline preprocessing: enumerate bifurcations and branches
    cent1 = preproc_centerline(cent_raw)

    # extract centerline conncetivity
    connectivity = get_connectivity(cent1)


def main(db, geometries):
    for geo in geometries:
        get_model(db.get_centerline_path(geo))


if __name__ == '__main__':
    descr = 'Automatically create, run, and post-process 1d-simulations'
    d, g, _ = input_args(descr)
    main(d, g)
