#!/usr/bin/env python

import paraview.simple as pv
import numpy as np
import os, glob, pdb

from vtk.util.numpy_support import vtk_to_numpy as v2n
from vtk.util.numpy_support import numpy_to_vtk as n2v

from get_bc_integrals import read_geo, write_geo


def write_geo_pv(fname_in, fname_out, ele_id, aname):
    out_array = n2v(ele_id)
    out_array.SetName(aname)
    reader, _, reader_cell = read_geo(fname_in)
    reader_cell.RemoveArray(aname)
    reader_cell.AddArray(out_array)
    write_geo(fname_out, reader)


def read_geo_pv(fname):
    _, ext = os.path.splitext(fname)
    if ext == '.vtp':
        reader = pv.XMLPolyDataReader(FileName=[fname])
    elif ext == '.vtu':
        reader_3d = pv.XMLUnstructuredGridReader(FileName=[fname])
        reader = pv.ExtractSurface(Input=reader_3d)
    else:
        raise ValueError('File extension ' + ext + ' unknown.')
    reader_fetch = pv.servermanager.Fetch(reader)
    return reader, reader_fetch, reader_fetch.GetPointData(), reader_fetch.GetCellData()


def is_unique(ids):
    return np.unique(ids).shape[0] == ids.shape[0]


def read_array(data, name):
    return v2n(data.GetArray(name)).astype(int)


def get_connectivity(reader):
    cells = []
    # loop all cells
    for i in range(reader.GetNumberOfCells()):
        c = reader.GetCell(i)
        points = []
        # loop all points of current cell
        for j in range(c.GetNumberOfPoints()):
            points.append(c.GetPointIds().GetId(j))
        cells.append(points)
    return np.array(cells)


def get_ids(fpath):
    reader, reader_fetch, point_data, cell_data = read_geo_pv(fpath)

    node_id = read_array(point_data, 'GlobalNodeID')
    assert is_unique(node_id), 'GlobalNodeID is not unique'

    cell_id = read_array(cell_data, 'GlobalElementID')
    connectivity = get_connectivity(reader_fetch)

    return reader, cell_id, np.sort(node_id[connectivity], axis=1)


# folder for simulation files
fpath_sim = '/home/pfaller/work/osmsc/data_uploaded'

# folder where generated data is saved
fpath_gen = '/home/pfaller/work/osmsc/data_generated'

# all geometries in repository
geometries = os.listdir(fpath_sim)
geometries.sort()
# geometries = ['0110_0001']

# loop all geometries in repository
for geo in geometries:
    print('Fixing geometry ' + geo)
    folder_out = os.path.join(fpath_gen, 'surfaces', geo)
    try:
        os.mkdir(folder_out)
    except OSError:
        pass

    # get volume mesh
    fpath_vol = os.path.join(fpath_sim, geo, 'results', geo + '_sim_results_in_cm.vtu')

    if not os.path.exists(fpath_vol):
        print('  skipping')
        continue

    _, vol_cell, vol_conn = get_ids(fpath_vol)

    # get all surface files of current geometry
    fpath_surf = glob.glob(os.path.join(fpath_sim, geo, 'extras', 'mesh-surfaces', '*.vtp'))
    fpath_surf.append(os.path.join(fpath_sim, geo, 'extras', 'mesh-surfaces', 'extras', 'all_exterior.vtp'))
    for f in fpath_surf:
        surf_fname = os.path.basename(f)
        print('  mesh ' + surf_fname)

        # get surface mesh
        surf_reader, surf_cell, surf_conn = get_ids(f)

        # match GlobalElementID in surface mesh with volume mesh
        surf_cell_new = np.zeros(surf_cell.shape, dtype=int)
        for i, cell in enumerate(surf_conn):
            found = vol_cell[(vol_conn == cell).all(axis=1)]
            assert found.shape[0] == 1, 'non-matching 2d/3d meshes'
            surf_cell_new[i] = found[0]

        assert np.max(np.abs(surf_cell_new - surf_cell)) <= 5, 'round-off error bigger than expected'

        # export surface
        fpath_out = os.path.join(folder_out, surf_fname)
        write_geo_pv(f, fpath_out, surf_cell_new, 'GlobalElementID')
