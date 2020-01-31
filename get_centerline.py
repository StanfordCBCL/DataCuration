#!/usr/bin/env python

import os
import sys
import pdb
import subprocess
from collections import OrderedDict

from get_database import Database, SimVascular
from common import input_args

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

        if not os.path.exists(db.get_centerline_path(geo)) or not os.path.exists(db.get_surfaces(geo, 'all_exterior')):
            continue

        if os.path.exists(db.get_section_path(geo)):
            continue

        # get model paths
        p = OrderedDict()
        p['cent'] = db.get_centerline_path(geo)
        p['surf'] = db.get_surfaces(geo, 'all_exterior')
        p['lines'] = db.get_centerline_section_path(geo)
        p['sections'] = db.get_section_path(geo)

        # get inlet and outlet centers
        params = Params(db.get_surface_dir(geo), os.path.basename(db.get_surfaces(geo, 'inflow')))
        cl.get_inlet_outlet_centers(params)
        p['inlet'] = repr(cl.inlet_center).replace(' ', '')
        p['outlets'] = repr(cl.outlet_centers).replace(' ', '')

        # assemble call string
        sv_string = [os.path.join(os.getcwd(), 'get_centerline_sv.py')]
        for v in p.values():
            sv_string += [v]

        print(sv_string)

        # execute SimVascular-Python
        sv.run_python(sv_string)


if __name__ == '__main__':
    descr = 'Generate a new surface mesh'
    d, g, _ = input_args(descr)
    main(d, g)
