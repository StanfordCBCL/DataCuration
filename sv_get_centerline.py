#!/usr/bin/env python

try:
    import sv
except ImportError:
    raise ImportError('Run with sv --python -- this_script.py')

import pdb
import os
import json
import argparse
import subprocess

from vtk_functions import read_geo, write_geo, clean


def centerlines(p, params):
    sv.Repository.ReadXMLPolyData('surf_in', p['surf_in'])
    sv.VMTKUtils.Centerlines('surf_in', [params.caps[0]], params.caps[1:], 'cent', 'voronoi')
    sv.Repository.WriteXMLPolyData('cent', p['cent'])
    sv.Repository.Delete('surf_in')
    sv.Repository.Delete('cent')
    sv.Repository.Delete('voronoi')


def sections(p):
    sv.Repository.ReadXMLPolyData('surf_in', p['surf_in'])
    sv.Repository.ReadXMLPolyData('cent_in', p['cent'])
    sv.VMTKUtils.CenterlineSections('cent_in', 'surf_in', 'cent_out', 'surf_out', 'sections')
    sv.Repository.WriteXMLPolyData('cent_out', p['cent'])
    sv.Repository.WriteXMLPolyData('surf_out', p['surf_out'])
    sv.Repository.WriteXMLPolyData('sections', p['sections'])
    sv.Repository.Delete('cent_in')
    sv.Repository.Delete('surf_in')
    sv.Repository.Delete('cent_out')
    sv.Repository.Delete('surf_out')
    sv.Repository.Delete('sections')


def main(params):
    # get model parameters
    p = {'surf_in': params.surf_in, 'surf_out': params.surf_out, 'cent': params.cent, 'sections': params.sections}

    centerlines(p, params)
    sections(p)


if __name__ == '__main__':
    # parse input arguments
    parser = argparse.ArgumentParser(description='generate centerline in SimVascular')
    parser.add_argument('surf_in', help='input surface')
    parser.add_argument('surf_out', help='output surface')
    parser.add_argument('sections', help='output sections')
    parser.add_argument('cent', help='output centerline')
    parser.add_argument('caps', type=json.loads, help='cap point ids')

    # run script
    main(parser.parse_args())
