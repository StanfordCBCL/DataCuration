#!/usr/bin/env python

import contextlib
import csv
import glob
import io
import os
import pdb
import re
import shutil
import sys
import argparse
import subprocess

import numpy as np

from common import input_args
from get_database import Database, SimVascular, Post

import matplotlib.pyplot as plt


def main(db, params):
    res = db.get_dict(db.get_log_file_1d())

    # sort errors
    for k, v in res.items():
        if '3d geometry has multiple inlets' in v:
            res[k] = '3d geometry has multiple inlets'
        if 'Inlet group id is not 0 or number of centerlines is not equal to the number of outlets' in v:
            res[k] = 'centerline creation failed'
        if 'float division by zero' in v:
            res[k] = '3d geometry is corrupted'

    res['0001_0001'] = '3d geometry contains a loop'

    errors = np.array([k for k in res.values()])

    # count errors
    num_errors = {}
    for err in np.unique(errors):
        num_errors[err] = {}
        num_errors[err]['n'] = np.sum(errors == err)
        num_errors[err]['geos'] = [k for k, v in res.items() if v == err]

    for err in num_errors.keys():
        g_string = err + '\n'
        for g in num_errors[err]['geos']:
            g_string += g + '\n'
        num_errors[err]['g_string'] = g_string[:-1]

    # make a montage for every error with the geometries
    for e in num_errors.keys():
        err = e.replace(' ', '_')
        err = err.replace('(', '')
        err = err.replace(')', '')

        g_string = ['/usr/bin/montage']
        for g in num_errors[e]['geos']:
            src = os.path.join(db.fpath_png, 'OSMSC' + g + '-sim.png')
            if not os.path.exists(src):
                src = os.path.join(db.fpath_png, 'OSMSC' + g + '-vol.png')
            g_string += [src]
        g_string += [os.path.join(db.get_statistics_dir(), err) + '.png']
        subprocess.Popen(g_string)

    # Pie chart, where the slices will be ordered and plotted counter-clockwise:
    labels_error = list(num_errors.keys())
    labels = [num_errors[v]['g_string'] for v in labels_error]
    sizes = [num_errors[v]['n'] for v in labels_error]

    num_geo = np.sum(np.array(sizes))
    assert num_geo == len(res), 'wrong number of errors'
    print('number of geometries: ' + repr(num_geo))

    explode = [0] * len(labels_error)
    explode[labels_error.index('success')] = 0.1
    explode = tuple(explode)

    fig1, ax1 = plt.subplots()
    # autopct = '%1.1f%%'
    # autopct = lambda p: '{:.0f}'.format(p * total / 100)
    ax1.pie(sizes, explode=explode, labels=labels, autopct=lambda p: '{:.0f}'.format(p * num_geo / 100), startangle=90)
    ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

    plt.show()




if __name__ == '__main__':
    descr = 'Automatically create, run, and post-process 1d-simulations'
    d, _, p = input_args(descr)
    main(d, p)
