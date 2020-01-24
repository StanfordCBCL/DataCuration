#!/usr/bin/env python
# coding=utf-8

import argparse
import glob
import os
import csv
import re

import numpy as np


def get_dict(fpath):
    """
    Read .npy dictionary saved with numpy.save if path is defined and exists
    Args:
        fpath: path to .npy file

    Returns:
        dictionary
    """
    if fpath is not None and os.path.exists(fpath):
        return np.load(fpath, allow_pickle=True).item()
    else:
        return {}


def read_results_1d(res_dir, params_file=None):
    """
    Read results from oneDSolver and store in dictionary
    Args:
        res_dir: directory containing 1D results
        params_file: optional, path to dictionary of oneDSolver input parameters

    Returns:
    Dictionary sorted as [result field][group id][time step]
    """
    # requested output fields
    fields_res_1d = ['flow', 'pressure', 'area', 'wss', 'Re']

    # read 1D simulation results
    results_1d = {}
    for field in fields_res_1d:
        # list all output files for field
        result_list_1d = glob.glob(os.path.join(res_dir, '*Group*Seg*_' + field + '.dat'))

        # loop segments
        results_1d[field] = {}
        for f_res in result_list_1d:
            with open(f_res) as f:
                reader = csv.reader(f, delimiter=' ')

                # loop nodes
                results_1d_f = []
                for line in reader:
                    results_1d_f.append([float(l) for l in line if l][1:])

            # store results and GroupId
            group = int(re.findall(r'\d+', f_res)[-2])
            results_1d[field][group] = np.array(results_1d_f)

    # read simulation parameters and add to result dict
    results_1d['params'] = get_dict(params_file)

    return results_1d


if __name__ == '__main__':
    # parse input arguments
    parser = argparse.ArgumentParser(description='Read oneDSolver results into dictionary')
    parser.add_argument('dir', help='directory containing oneDSolver results')
    parser.add_argument('out', help='path of output dictionary')
    parser.add_argument('-p', '--param', help='path to .npy dictionary containing oneDSolver options')
    p = parser.parse_args()

    # get model database
    res = read_results_1d(p.dir, p.param)

    # save to file
    np.save(p.out, res)
