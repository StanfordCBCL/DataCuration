#!/usr/bin/env python

import numpy as np
import os
import pdb

from vtk.util.numpy_support import vtk_to_numpy as v2n
from vtk.util.numpy_support import numpy_to_vtk as n2v

from common import input_args
from get_bc_integrals import read_geo, write_geo
from get_database import Database


def main(db, geometries):
    """
    Remove all results from volumetric mesh
    """
    for geo in geometries:
        print('Running geometry ' + geo)

        if not os.path.exists(db.get_volume(geo)):
            continue

        # read volume mesh with results
        vol, vol_n, vol_c = read_geo(db.get_volume(geo))

        # arrays to keep
        remove = []
        remove += [{'handle': vol_n, 'keep': ['GlobalNodeID'], 'remove': []}]
        remove += [{'handle': vol_c, 'keep': ['GlobalElementID', 'BC_FaceID'], 'remove': []}]

        # collect array names to remove
        for r in remove:
            for i in range(r['handle'].GetNumberOfArrays()):
                name = r['handle'].GetArrayName(i)
                if name not in r['keep']:
                    r['remove'] += [name]

        # remove arrays
        for r in remove:
            for a in r['remove']:
                r['handle'].RemoveArray(a)

        # export to generated folder
        write_geo(db.get_volume_mesh(geo), vol)


if __name__ == '__main__':
    descr = 'Fix wrong GlobalElementID'
    d, g, _ = input_args(descr)
    main(d, g)
