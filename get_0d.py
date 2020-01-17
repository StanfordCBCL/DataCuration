#!/usr/bin/env python

import numpy as np
import sys
import os
import shutil
import glob
import csv
import re
import pdb
import contextlib, io

from get_database import Database, SimVascular
from get_sim import write_bc


def read_results_0d(fpath, geo):
    # requested output fields
    fields = ['flow', 'pressure']
    conversion = {'flow': 1 / 0.06, 'pressure': 1/7.50062e-4}

    # read 1D simulation results
    res = {}
    for field in fields:
        # list all output files for field
        result_list_1d = glob.glob(os.path.join(fpath, field + '_*.txt'))

        # loop segments
        res[field] = {}
        for f_res in result_list_1d:
            with open(f_res) as f:
                reader = csv.reader(f, delimiter=' ')

                # loop nodes
                results_f = []
                time_f = []
                for line in reader:
                    time_f.append(float(line[0]))
                    results_f.append(float(line[1]))

            # store results and GroupId
            segment = int(re.findall(r'\d+', f_res)[-1])
            res[field][segment] = np.array(results_f) * conversion[field]
        res['time'] = np.array(time_f)

    return res


def generate_0d(db, geo):

    return True, None


def main(db, geometries):
    for geo in geometries:
        print('Running geometry ' + geo)

        # output path for 0d results
        fpath_out = os.path.join(db.fpath_gen, '0d_flow', geo)

        # extract results
        results_0d = read_results_0d(db.get_solve_dir_0d(geo), geo)
        np.save(fpath_out, results_0d)


if __name__ == '__main__':
    descr = 'Automatically create, run, and post-process 1d-simulations'
    d, g, _ = input_args(descr)
    main(d, g)
