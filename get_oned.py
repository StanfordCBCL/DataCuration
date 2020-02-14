#!/usr/bin/env python
# coding=utf-8

import pdb
import vtk
import numpy as np

from vtk.util.numpy_support import numpy_to_vtk as n2v
from vtk.util.numpy_support import vtk_to_numpy as v2n

from get_database import Database, SimVascular, Post, input_args
from vtk_functions import read_geo, write_geo


def get_model(centerline):
    r, p, c = read_geo(centerline)
    r.GetOutput().GetPointData().SetActiveScalars('CenterlineSectionBifurcation')

    fun = vtk.vtkImplicitDataSet()
    fun.SetDataSet(r.GetOutput())

    clean = vtk.vtkCleanPolyData()
    clean.SetInputData(r.GetOutput())
    clean.PointMergingOn()
    clean.Update()

    clip = vtk.vtkClipDataSet()
    clip.SetInputData(clean.GetOutput())
    clip.SetClipFunction(fun)
    clip.SetValue(0.5)
    clip.GenerateClippedOutputOn()
    clip.Update()

    con = vtk.vtkConnectivityFilter()
    con.SetInputData(clip.GetOutput())
    con.SetExtractionModeToAllRegions()
    con.Update()

    array = n2v(-1 * np.ones(clip.GetOutput(1).GetNumberOfPoints()))
    array.SetName('RegionId')
    clip.GetOutput(1).GetPointData().AddArray(array)

    append = vtk.vtkAppendFilter()
    append.AddInputData(con.GetOutput())
    # append.AddInputData(clip.GetOutput(1))
    append.MergePointsOn()
    append.Update()

    write_geo('test.vtp', clean)
    write_geo('test.vtu', con)


def main(db, geometries):
    for geo in geometries:
        get_model(db.get_centerline_path(geo))


if __name__ == '__main__':
    descr = 'Automatically create, run, and post-process 1d-simulations'
    d, g, _ = input_args(descr)
    main(d, g)
