#!/usr/bin/env python

import os
import sys
import pdb
import shutil
import subprocess
import tempfile
import vtk

from collections import OrderedDict
import numpy as np

from get_database import Database, SimVascular, input_args
from vtk_functions import ClosestPoints, read_geo, write_geo
from vmtk import vtkvmtk, vmtkscripts

sys.path.append('/home/pfaller/work/repos/SimVascular/Python/site-packages/')

from sv_1d_simulation import Centerlines


class Params(object):
    """
    Minimal parameter set for get_inlet_outlet_centers function in Centerlines class
    """
    def __init__(self, surf_dir, inflow, surf):
        self.boundary_surfaces_dir = surf_dir
        self.inlet_face_input_file = inflow
        self.surface_model = surf
        self.output_directory = '/home/pfaller'
        self.CENTERLINES_OUTLET_FILE_NAME = 'bla'


def main(db, geometries):
    # SimVascular instance
    sv = SimVascular()

    # Centerline instance
    cl = Centerlines()

    for geo in geometries:
        print('Running geometry ' + geo)

        if not db.get_surfaces(geo, 'all_exterior'):
            continue

        # get model paths
        p = OrderedDict()
        p['surf_in'] = db.get_surfaces(geo, 'all_exterior')
        p['surf_out'] = db.get_surfaces_grouped_path(geo)
        p['sections'] = db.get_section_path(geo)
        p['cent'] = db.get_centerline_path(geo)

        # copy cap surfaces to temp folder
        fpath_surf = tempfile.mkdtemp()
        for f in db.get_surfaces(geo, 'caps'):
            shutil.copy2(f, fpath_surf)

        # get inlet and outlet centers
        params = Params(fpath_surf, os.path.basename(db.get_surfaces(geo, 'inflow')), p['surf_in'])
        cl.get_inlet_outlet_centers(params)

        # find corresponding point id
        cp = ClosestPoints(p['surf_in'])
        caps = np.vstack((cl.inlet_center, np.array(cl.outlet_centers).reshape(-1, 3)))
        id_caps = cp.search(caps)

        p['caps'] = repr(id_caps).replace(' ', '')

        # write outlet names to file
        with open(db.get_centerline_outlet_path(geo), 'w+') as f:
            for s in cl.outlet_face_names:
                f.write(s + '\n')

        # assemble call string
        sv_string = [os.path.join(os.getcwd(), 'get_centerline_sv.py')]
        for v in p.values():
            sv_string += [v]

        # execute SimVascular-Python
        sv.run_python(sv_string)

        # remove temp dir
        shutil.rmtree(fpath_surf)


if __name__ == '__main__':
    descr = 'Generate a new surface mesh'
    d, g, _ = input_args(descr)
    main(d, g)
