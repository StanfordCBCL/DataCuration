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
    assert n_geo == 1, 'input centerline consists of more than one piece (' + repr(n_geo) + ')'

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

        # loop through all points in centerline (assumes points are ordered consecutively)
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
    assert np.unique(mesh).shape[0] > 1, 'geometry is a single branch'
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
    # modify this array to ensure bifurcations are only where they should be
    bf = centerline.GetPointData().GetArray('CenterlineSectionBifurcation')

    # make sure actual bifurcation points are included
    bf_id = centerline.GetPointData().GetArray('BifurcationId')
    for i in range(centerline.GetNumberOfPoints()):
        if bf_id.GetValue(i) > -1:
            bf.SetValue(i, 1)

    # exclude bifurcations that are somewhere in the middle of a branch
    clip = split_centerline(centerline)
    bifurcations = connectivity_all(clip.GetOutput(0))
    for i in range(bifurcations.GetNumberOfExtractedRegions()):
        # extract bifurcation
        t = threshold(bifurcations.GetOutput(), i, 'RegionId').GetOutput()

        # not a real bifurcation if it contains only one BranchId
        if np.unique(v2n(t.GetPointData().GetArray('BranchId'))).shape[0] == 1:
            # remove bifurcation (convert to branch)
            t_p_id = t.GetPointData().GetArray('GlobalNodeID')
            for j in range(t.GetNumberOfPoints()):
                bf.SetValue(t_p_id.GetValue(j), 0)

    # eclude branches that consist of a single point
    clip = split_centerline(centerline)
    branches = connectivity_all(clip.GetOutput(1))
    for i in range(branches.GetNumberOfExtractedRegions()):
        # extract branch
        t = threshold(branches.GetOutput(), i, 'RegionId').GetOutput()

        # remove branch (convert to bifurcation)
        if t.GetNumberOfPoints() == 1:
            t_p_id = t.GetPointData().GetArray('GlobalNodeID')
            bf.SetValue(t_p_id.GetValue(0), 1)


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

    # add path array to geometry
    points = v2n(geo.GetOutput().GetPoints().GetData())
    branch_id = v2n(geo.GetOutput().GetPointData().GetArray('BranchId'))
    branches = np.unique(branch_id).tolist()
    branches.remove(-1)
    dist = np.zeros(points.shape[0])
    for br in branches:
        ids = branch_id == br
        dist[ids] = np.cumsum(np.insert(np.linalg.norm(np.diff(points[ids], axis=0), axis=1), 0, 0))
    array = n2v(dist)
    array.SetName('Path')
    geo.GetOutput().GetPointData().AddArray(array)

    return geo


def transfer_arrays(src, trg):
    # get mapping between src and trg points
    tree = scipy.spatial.KDTree(v2n(src.GetPoints().GetData()))
    _, ids = tree.query(trg.GetPoints().GetData())

    # loop all arrays
    for a in range(src.GetPointData().GetNumberOfArrays()):
        # get source array
        array_src = src.GetPointData().GetArray(a)

        n_val = array_src.GetNumberOfComponents()

        # setup target array
        array_trg = vtk.vtkDoubleArray()
        array_trg.SetName(array_src.GetName())
        array_trg.SetNumberOfComponents(n_val)
        array_trg.SetNumberOfValues(trg.GetNumberOfPoints() * n_val)

        # copy array values using point map
        for i, j in enumerate(ids):
            val = array_src.GetTuple(j)
            if n_val == 1:
                array_trg.SetValue(i, val[0])
            elif n_val == 3:
                array_trg.SetTuple3(i, val[0], val[1], val[2])
            else:
                raise ValueError('Not implemented')

        trg.GetPointData().AddArray(array_trg)


def group_surface(cent_clean, fpath_surf):
    surf = read_geo(fpath_surf)

    normals = vtk.vtkPolyDataNormals()
    normals.SetInputData(surf.GetOutput())
    normals.SplittingOff()
    normals.AutoOrientNormalsOn()
    normals.ConsistencyOn()
    normals.FlipNormalsOn()
    normals.Update()

    normal = v2n(normals.GetOutput().GetPointData().GetNormals())
    radius = v2n(cent_clean.GetPointData().GetArray('MaximumInscribedSphereRadius'))
    branch = v2n(cent_clean.GetPointData().GetArray('BranchId'))
    points_cent = v2n(cent_clean.GetPoints().GetData())
    points_surf = v2n(surf.GetOutput().GetPoints().GetData())

    r_max = np.max(radius)

    tree_cent = scipy.spatial.KDTree(points_cent)

    array = vtk.vtkIntArray()
    array.SetName('BranchId')
    array.SetNumberOfValues(surf.GetOutput().GetNumberOfPoints())

    # loop all points in surface (this is very expensive)
    for i, (p, n) in enumerate(zip(points_surf, normal)):
        # get centerline points within sphere (this could possibly be made smaller)
        # todo: speedup be using query_ball_tree ??
        ids = tree_cent.query_ball_point(p, 2 * r_max)

        # vector from surface point to centerline points
        vec = points_cent[ids] - p

        # signed distance in inward normal direction
        dist_n = np.dot(vec, n)
        vec_n = np.outer(dist_n, n)
        vec_p = vec - vec_n

        # distance perpendicular to normal direction (i.e. parallel to centerline)
        dist_p = np.linalg.norm(vec_p, axis=1)

        dist_m = abs(dist_p) + abs(dist_n)

        # dist = np.sqrt(dist_p**2 + dist_n**2)

        # only choose from centerline points inside surface
        i_inside = np.where((dist_n > 0) * (dist_n < 2 * radius[ids]))[0]
        # if not np.any(i_inside):
        #     pdb.set_trace()

        # choose closest point
        # j = np.argmin(dist[i_inside])

        # choose point with biggest radius
        # j = np.argmax(radius[i_inside])

        n_min = 3
        close = np.argpartition(dist_p[i_inside], n_min)[:n_min]

        # j = np.argmax(radius[i_inside[close]])
        # j = np.argmin(dist_n[i_inside[close]])
        # j = np.argmin(dist_m[i_inside])

        dist_p_min = 1e16
        dist_n_min = 1e16
        for k in range(i_inside.shape[0]):
            if dist_p[i_inside[k]] < dist_p_min + 0.1 and dist_n[k] < dist_n_min:
                j = k
                dist_p_min = dist_p[i_inside[k]]
                dist_n_min = dist_n[i_inside[k]]
                # print(dist_p_min)
        # pdb.set_trace()

        # if np.unique(branch[np.array(ids)[i_inside[close]]]).shape[0] > 1:
        if i == 4946:
            print(i)
            import matplotlib.pyplot as plt
            # plt.scatter(dist_p, dist_n, c=(dist - dist.min()) / (dist.max() - dist.min()))
            plt.scatter(dist_p[i_inside], dist_n[i_inside], c=branch[np.array(ids)[i_inside]])
            # plt.plot(dist_p[i_inside[close[j]]], dist_n[i_inside[close[j]]], 'rx')
            plt.plot(dist_p[i_inside[j]], dist_n[i_inside[j]], 'rx')
            plt.show()
            pdb.set_trace()

        try:
            # array.SetValue(i, int(branch[ids[i_inside[close[j]]]]))
            array.SetValue(i, int(branch[ids[i_inside[j]]]))
        except Exception:
            pdb.set_trace()
    #
    # tree_surf = scipy.spatial.KDTree(points_surf)
    #
    # pdb.set_trace()
    #
    # for i, (p, r) in enumerate(zip(points_cent, radius)):
    #     ids = tree_surf.query_ball_point(p, 2 * r)
    #
    #     for j in ids:
    #         array.SetValue(j, cent_clean.GetPointData().GetArray('BranchId').GetValue(i))
    surf.GetOutput().GetPointData().AddArray(array)
    return surf


def get_model(fpath_cent_raw, fpath_cent_sections, fpath_surf):
    # read raw centerline from file
    cent_raw = read_geo(fpath_cent_raw)

    # create "clean" watertight centerline
    cent_clean = create_centerline(cent_raw)

    # read "dirty" centerline with CenterlineSectionBifurcation point array (cells might be disconnected or distorted)
    cent_dirty = read_geo(fpath_cent_sections)

    # move point arrays to split centerline
    # todo: merge all steps in one by creating a clean centerline from the beginning
    transfer_arrays(cent_dirty.GetOutput(), cent_clean)

    # make sure only real bifurcations are included
    # todo: possibly make bifurcation detection more robust so this step is not neccessary
    clean_bifurcation(cent_clean)

    # split centerline
    cent_split = group_centerline(cent_clean)

    # WIP
    surf_group = group_surface(cent_clean, fpath_surf)

    return cent_split, surf_group


def main(db, geometries):
    for geo in geometries:
        print('Running geometry ' + geo)

        fpath_cent_raw = db.get_centerline_path_raw(geo)
        fpath_cent_sections = db.get_centerline_path(geo)
        fpath_surf = db.get_surfaces(geo, 'all_exterior')

        if not os.path.exists(fpath_cent_raw) or not os.path.exists(fpath_cent_sections):
            print('  no centerline')
            continue

        # try:
        mesh_cent, mesh_surf = get_model(fpath_cent_raw, fpath_cent_sections, fpath_surf)
        # except (AssertionError, AttributeError, KeyError, TypeError) as e:
        #     print('  ' + str(e))
        #     continue

        write_geo(db.get_centerline_path_oned(geo), mesh_cent)
        write_geo(db.get_surfaces_grouped_path_oned(geo), mesh_surf)


if __name__ == '__main__':
    descr = 'Automatically create, run, and post-process 1d-simulations'
    d, g, _ = input_args(descr)
    main(d, g)
