#!/usr/bin/env python

import os
import vtk
import argparse
import pdb

import numpy as np

from get_database import input_args, Database, Post
from get_mean_flow_3d import extract_results


def main(db, geometries):
    """
    Loop all geometries in database
    """
    for geo in geometries:
        print('Processing ' + geo)

        fpath_1d = db.get_centerline_path(geo)
        fpath_vol = db.get_res_3d_vol_rerun(geo)
        fpath_out = db.get_3d_flow_rerun(geo)
        extract_results(fpath_1d, fpath_vol, fpath_out, only_caps=True)


if __name__ == '__main__':
    descr = 'Plot comparison of xd-results'
    d, g, _ = input_args(descr)
    main(d, g)
