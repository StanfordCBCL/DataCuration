#!/usr/bin/env python

import os
import sys
import pdb
import tempfile
import shutil

from get_database import Database, input_args

sys.path.append('/home/pfaller/work/repos/SimVascular/Python/site-packages/')

from sv_1d_simulation import centerlines


class Params(object):
    """
    Minimal parameter set for get_inlet_outlet_centers function in Centerlines class
    """
    def __init__(self, p):
        self.boundary_surfaces_dir = p['f_surf_caps']
        self.inlet_face_input_file = p['f_inflow']
        self.surface_model = p['f_surf_in']
        self.output_directory = os.path.dirname(p['f_outlet'])
        self.CENTERLINES_OUTLET_FILE_NAME = os.path.basename(p['f_outlet'])
        self.centerlines_output_file = p['f_cent_out_vmtk']
        self.cent_out = p['f_cent_out']
        self.surf_out = p['f_surf_out']
        self.sections_out = p['f_sections_out']


def main(db, geometries):
    for geo in geometries:
        print('Running geometry ' + geo)

        if not db.get_surfaces(geo, 'all_exterior'):
            continue

        # get model paths
        params = {'f_surf_in': db.get_surfaces(geo, 'all_exterior'),
                  'f_surf_caps': tempfile.mkdtemp(),
                  'f_inflow': os.path.basename(db.get_surfaces(geo, 'inflow')),
                  'f_outlet': db.get_centerline_outlet_path(geo),
                  'f_cent_out_vmtk': db.get_centerline_vmtk_path(geo),
                  'f_cent_out': db.get_centerline_path(geo),
                  'f_surf_out': db.get_surfaces_grouped_path(geo),
                  'f_sections_out': db.get_section_path(geo)}

        # copy cap surfaces to temp folder
        for f in db.get_surfaces(geo, 'caps'):
            shutil.copy2(f, params['f_surf_caps'])

        params = Params(params)

        # call SimVascular centerline extraction
        cl = centerlines.Centerlines()
        cl.extract_center_lines(params)
        cl.extract_branches(params)
        cl.write_outlet_face_names(params)


if __name__ == '__main__':
    descr = 'Generate a new surface mesh'
    d, g, _ = input_args(descr)
    main(d, g)
