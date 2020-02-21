#!/usr/bin/env python
# coding=utf-8

import os
import pdb
import vtk
import scipy
import numpy as np

from vtk.util.numpy_support import numpy_to_vtk as n2v
from vtk.util.numpy_support import vtk_to_numpy as v2n

from get_database import Database, SimVascular, Post, input_args
from vtk_functions import read_geo, write_geo, connectivity_all, clean, add_scalars, rename, replace, threshold


def create_centerline(centerline):
    # cleanup: merge duplicate points
    cent = clean(centerline).GetOutput()

    # check if geometry is one piece
    test = connectivity_all(cent)
    n_geo = test.GetNumberOfExtractedRegions()
    assert n_geo == 1, 'centerline consists of more than one piece (' + repr(n_geo) + ')'

    n_points = cent.GetNumberOfPoints()
    n_cells = cent.GetNumberOfCells()

    # build connected centerline geometry
    points = vtk.vtkPoints()
    lines = vtk.vtkCellArray()

    # add inlet point
    points.InsertNextPoint(cent.GetPoint(0))
    point_ids = [0]

    # loop all centerlines
    for c in range(n_cells):
        # individual centerline
        cell = cent.GetCell(c)

        # loop through all points in centerline (assumes they are ordered consecutively)
        for p in range(cell.GetPointIds().GetNumberOfIds()):
            i = cell.GetPointId(p)

            # add point and line only if point hasn't been added yet
            if i not in point_ids and i > 0:
                points.InsertNextPoint(cent.GetPoint(i))
                point_ids += [i]

                # add line connecting this point and previous point
                line = vtk.vtkLine()
                line.GetPointIds().SetId(0, point_ids.index(cell.GetPointId(p - 1)))
                line.GetPointIds().SetId(1, i)
                lines.InsertNextCell(line)

    # create polydata
    polydata = vtk.vtkPolyData()
    polydata.SetPoints(points)
    polydata.SetLines(lines)
    polydata.Modified()

    assert polydata.GetNumberOfPoints() == n_points, 'number of points mismatch'
    assert polydata.GetNumberOfPoints() == polydata.GetNumberOfCells() + 1, 'number of cells mismatch'

    # add unique identifiers
    point_id = vtk.vtkIntArray()
    cell_id = vtk.vtkIntArray()
    point_id.SetName('GlobalNodeID')
    cell_id.SetName('GlobalElementID')
    point_id.SetNumberOfValues(n_points)
    cell_id.SetNumberOfValues(n_points - 1)
    for i in range(n_points):
        point_id.SetValue(i, i)
    for i in range(n_points - 1):
        cell_id.SetValue(i, i)
    polydata.GetPointData().AddArray(point_id)
    polydata.GetCellData().AddArray(cell_id)

    # collect all centerlines that run through each point
    mesh = [[]] * n_points
    for c in range(n_cells):
        for p in range(cent.GetCell(c).GetPointIds().GetNumberOfIds()):
            i = cent.GetCell(c).GetPointId(p)
            if not len(mesh[i]):
                mesh[i] = [c]
            else:
                mesh[i] += [c]

    # find unique segments by the combintation of centerlines in each point
    sections = sorted(np.unique(mesh), key=len, reverse=True)

    assert len(sections[0]) == n_cells, 'inlet segment not detected'
    assert len(sections[0]) > len(sections[1]), 'multiple inlet segments detected'
    for i in range(n_cells):
        assert len(sections[-1 - i]) == 1, 'outlet segment not detected'
    for s in sections:
        assert len(s) > 0, 'point does not belong to any segment'

    # export array with unique branch identifier (BranchId)
    branch = vtk.vtkIntArray()
    branch.SetName('BranchId')
    branch.SetNumberOfValues(n_points)
    for i, m in enumerate(sections):
        for j, p in enumerate(mesh):
            if p == m:
                branch.SetValue(j, i)
    polydata.GetPointData().AddArray(branch)

    # export array with unique bifurcation identifier (BifurcationId)
    bifurcation = vtk.vtkIntArray()
    bifurcation.SetName('BifurcationId')
    bifurcation.SetNumberOfValues(n_points)
    bifurcation.Fill(-1)
    ids = vtk.vtkIdList()

    # export dictionary with list of attached branches for each bifurcation
    for p in range(n_points):
        polydata.GetPointCells(p, ids)
        n_ids = ids.GetNumberOfIds()
        if n_ids > 2:
            # store bifurcation id in array
            bifurcation.SetValue(p, 1)

    polydata.GetPointData().AddArray(bifurcation)

    return polydata


def clean_bifurcation(centerline):
    # make sure actual bifurcation points are included
    bf = centerline.GetPointData().GetArray('CenterlineSectionBifurcation')
    bf_id = centerline.GetPointData().GetArray('BifurcationId')
    for i in range(centerline.GetNumberOfPoints()):
        if bf_id.GetValue(i) > -1:
            bf.SetValue(i, 1)

    # extract all bifurcations
    clip = split_centerline(centerline)
    bifurcations = connectivity_all(clip.GetOutput(0))
    for i in range(bifurcations.GetNumberOfExtractedRegions()):
        # extract
        t = threshold(bifurcations.GetOutput(), i, 'RegionId').GetOutput()

        # not a real bifurcation if it contains only one BranchId
        if np.unique(v2n(t.GetPointData().GetArray('BranchId'))).shape[0] == 1:
            # remove bifurcation
            t_p_id = t.GetPointData().GetArray('GlobalNodeID')
            for j in range(t.GetNumberOfPoints()):
                bf.SetValue(t_p_id.GetValue(j), 0)

    # todo: what happens if inlet or outlets are bifurcations?


def split_centerline(centerline):
    assert centerline.GetPointData().HasArray(
        'CenterlineSectionBifurcation'), 'no CenterlineSectionBifurcation'

    # clipping function using bifurcation array
    centerline.GetPointData().SetActiveScalars('CenterlineSectionBifurcation')
    fun = vtk.vtkImplicitDataSet()
    fun.SetDataSet(centerline)

    # clip centerline in branches and bifurcations
    clip = vtk.vtkClipPolyData()
    clip.SetInputData(centerline)
    clip.SetClipFunction(fun)
    # clip.InsideOutOn()
    clip.SetValue(0)
    clip.GenerateClippedOutputOn()
    clip.Update()
    return clip


def group_centerline(centerline):
    # clip centerline at bifurcations
    clip = split_centerline(centerline)

    # enumerate all branches
    branches = connectivity_all(clip.GetOutput(1))
    rename(branches, 'RegionId', 'BranchId')
    add_scalars(branches, 'BifurcationId', -1)

    # enumerate all bifurcations
    bifurcations = connectivity_all(clip.GetOutput(0))
    rename(bifurcations, 'RegionId', 'BifurcationId')
    add_scalars(bifurcations, 'BranchId', -1)

    # bring bifurcations and branches back together
    append = vtk.vtkAppendFilter()
    append.AddInputData(branches.GetOutput())
    append.AddInputData(bifurcations.GetOutput())
    append.MergePointsOn()
    append.Update()

    # check if geometry is one piece
    test = connectivity_all(append.GetOutput())
    n_geo = test.GetNumberOfExtractedRegions()
    assert n_geo == 1, 'centerline consists of more than one piece (' + repr(n_geo) + ')'

    # convert vtkUnstructerdGrid to vtkPolyData
    geo = vtk.vtkGeometryFilter()
    geo.SetInputData(append.GetOutput())
    geo.Update()

    return geo


def get_connectivity(cent):
    # read arrays from centerline
    bifurcation = v2n(cent.GetPointData().GetArray('BifurcationId'))
    branch = v2n(cent.GetPointData().GetArray('BranchId'))

    # get centerline connectivity: which branches are attached to which bifurcation?
    connectivity = {}
    for c in range(cent.GetNumberOfCells()):
        ele = cent.GetCell(c)
        point_ids = np.array([ele.GetPointIds().GetId(i) for i in range(ele.GetPointIds().GetNumberOfIds())])
        br = branch[point_ids].tolist()

        # find cells that are at borders of bifurcations (more two unique RegionIds)
        if np.unique(br).shape[0] == 2:
            # should be one branch and one bifurcation
            assert -1 in br, 'No bifurcation in cell'

            # local node ids of branch and bifurcation (0 or 1)
            i_br_ele = br.index(-1)
            i_bf_ele = int(not i_br_ele)

            # branch and bifurcation id
            i_bf = bifurcation[point_ids[i_br_ele]]
            i_br = branch[point_ids[i_bf_ele]]

            assert i_bf != -1, 'Multiple bifurcations in cell'
            assert i_br != -1, 'Multiple branches in cell'

            # store unique branch id
            if i_bf in connectivity:
                if i_br not in connectivity[i_bf]:
                    connectivity[i_bf] += [i_br]
            else:
                connectivity[i_bf] = [i_br]

    for c in connectivity.values():
        assert len(c) >= 3, 'bifurcation with less than 3 branches detected (' + repr(len(c)) + ')'

    return connectivity


def transfer_arrays(src, trg):
    tree = scipy.spatial.KDTree(v2n(src.GetPoints().GetData()))
    _, ids = tree.query(trg.GetPoints().GetData())

    for a in range(src.GetPointData().GetNumberOfArrays()):
        array_src = src.GetPointData().GetArray(a)

        array_trg = vtk.vtkDoubleArray()
        array_trg.SetName(src.GetPointData().GetArrayName(a))
        array_trg.SetNumberOfValues(trg.GetNumberOfPoints())
        for i, j in enumerate(ids):
            array_trg.SetValue(i, array_src.GetValue(j))
        trg.GetPointData().AddArray(array_trg)


def get_model(fpath_cent_raw, fpath_cent_sections):
    # read raw centerline from file
    cent_raw, _, _ = read_geo(fpath_cent_raw)

    # create watertight centerline
    cent_water = create_centerline(cent_raw)

    # read centerline with CenterlineSectionBifurcation point array
    cent_sections, _, _ = read_geo(fpath_cent_sections)

    # move point arrays to split centerline
    transfer_arrays(cent_sections.GetOutput(), cent_water)

    # make sure only real bifurcations are included
    clean_bifurcation(cent_water)

    # split centerline
    cent_split = group_centerline(cent_water)

    # get centerline branch and bifurcation connectivity
    split = get_connectivity(cent_split.GetOutput())

    return cent_split, split


def main(db, geometries):
    for geo in geometries:
        print('Running geometry ' + geo)

        if not os.path.exists(db.get_centerline_path(geo)):
            print('  no centerline')
            continue

        try:
            mesh, split = get_model(db.get_centerline_path_raw(geo), db.get_centerline_path(geo))
        except (AssertionError, AttributeError, KeyError) as e:
            print('  ' + str(e))
            continue

        print(split)
        write_geo(db.get_centerline_path_oned(geo), mesh)


if __name__ == '__main__':
    descr = 'Automatically create, run, and post-process 1d-simulations'
    d, g, _ = input_args(descr)
    main(d, g)
