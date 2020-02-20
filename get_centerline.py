#!/usr/bin/env python

import os
import sys
import pdb
import shutil
import subprocess
import tempfile

from collections import OrderedDict
import numpy as np

from get_database import Database, SimVascular, input_args
from vtk_functions import ClosestPoints

sys.path.append('/home/pfaller/work/repos/SimVascular/Python/site-packages/')

from sv_1d_simulation import Centerlines


class Params(object):
    """
    Minimal parameter set for get_inlet_outlet_centers function in Centerlines class
    """
    def __init__(self, surf_dir, inflow):
        self.boundary_surfaces_dir = surf_dir
        self.inlet_face_input_file = inflow


def main(db, geometries):
    # SimVascular instance
    sv = SimVascular()

    # Centerline instance
    cl = Centerlines()

    for geo in geometries:
        print('Running geometry ' + geo)

        if not db.get_surfaces(geo, 'all_exterior'):
            continue

        # if os.path.exists(db.get_section_path(geo)):
        #     continue

        # get model paths
        p = OrderedDict()
        p['surf'] = db.get_surfaces(geo, 'all_exterior')
        p['lines'] = db.get_centerline_path(geo)
        p['sections'] = db.get_section_path(geo)
        p['surf_grouped'] = db.get_surfaces_grouped_path(geo)

        # copy cap surfaces to temp folder
        fpath_surf = tempfile.mkdtemp()
        for f in db.get_surfaces(geo, 'caps'):
            shutil.copy2(f, fpath_surf)

        # get inlet and outlet centers
        params = Params(fpath_surf, os.path.basename(db.get_surfaces(geo, 'inflow')))
        cl.get_inlet_outlet_centers(params)

        # remove temp dir
        shutil.rmtree(fpath_surf)

        # find corresponding point id
        cp = ClosestPoints(p['surf'])
        caps = np.vstack((cl.inlet_center, np.array(cl.outlet_centers).reshape(-1, 3)))
        id_caps = cp.search(caps)

        # convert for export
        p['caps'] = repr(id_caps).replace(' ', '')

        # assemble call string
        sv_string = [os.path.join(os.getcwd(), 'get_centerline_sv.py')]
        for v in p.values():
            sv_string += [v]

        # write outlet names to file
        with open(db.get_centerline_outlet_path(geo), 'w+') as f:
            for s in cl.outlet_face_names:
                f.write(s + '\n')

        # execute SimVascular-Python
        sv.run_python(sv_string)
        # sv.run_python_debug(sv_string)


if __name__ == '__main__':
    descr = 'Generate a new surface mesh'
    d, g, _ = input_args(descr)
    main(d, g)
