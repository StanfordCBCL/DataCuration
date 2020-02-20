#!/usr/bin/env python

try:
    import sv
except ImportError:
    raise ImportError('Run with sv --python -- this_script.py')

import os
import json
import argparse


def centerlines(p, params):
    sv.Repository.ReadXMLPolyData('surf', p['surf'])
    sv.VMTKUtils.Centerlines('surf', [params.caps[0]], params.caps[1:], 'tmp1', 'voronoi')
    sv.Repository.WriteXMLPolyData('tmp1', p['lines'])
    sv.Repository.Delete('surf')
    sv.Repository.Delete('tmp1')
    sv.Repository.Delete('voronoi')


def separate(p):
    sv.Repository.ReadXMLPolyData('tmp1', p['lines'])
    sv.VMTKUtils.Separatecenterlines('tmp1', 'tmp2')
    sv.Repository.WriteXMLPolyData('tmp2', p['lines'])
    sv.Repository.Delete('tmp1')
    sv.Repository.Delete('tmp2')


def group(p):
    sv.Repository.ReadXMLPolyData('surf', p['surf'])
    sv.Repository.ReadXMLPolyData('tmp2', p['lines'])
    sv.VMTKUtils.Grouppolydata('surf', 'tmp2', 'surf_grouped')
    sv.Repository.WriteXMLPolyData('surf_grouped', p['surf_grouped'])
    sv.Repository.Delete('surf')
    sv.Repository.Delete('tmp2')
    sv.Repository.Delete('surf_grouped')


def sections(p):
    sv.Repository.ReadXMLPolyData('surf', p['surf'])
    sv.Repository.ReadXMLPolyData('tmp2', p['lines'])
    sv.Repository.ReadXMLPolyData('surf_grouped', p['surf_grouped'])
    sv.VMTKUtils.CenterlineSections('tmp2', 'surf', 'surf_grouped', 'lines', 'sections')
    sv.Repository.WriteXMLPolyData('lines', p['lines'])
    sv.Repository.WriteXMLPolyData('sections', p['sections'])
    sv.Repository.Delete('tmp2')
    sv.Repository.Delete('surf_grouped')
    sv.Repository.Delete('lines')
    sv.Repository.Delete('sections')


def main(params):
    # get model parameters
    p = {'surf': params.surf, 'lines': params.lines, 'surf_grouped': params.surf_grouped, 'sections': params.sections}

    # execute VMTK centerline functions
    if not os.path.exists(p['surf']):
        print('Skipping (no surface mesh)')
        return

    if not os.path.exists(p['lines']):
        centerlines(p, params)
        separate(p)
    else:
        print('Found centerline')

    if not os.path.exists(p['surf_grouped']):
        group(p)
    else:
        print('Found surface grouped')

    if not os.path.exists(p['sections']):
        sections(p)
    else:
        print('Found sections')


if __name__ == '__main__':
    # parse input arguments
    parser = argparse.ArgumentParser(description='generate centerline in SimVascular')
    parser.add_argument('surf', help='input surface')
    parser.add_argument('lines', help='output centerline')
    parser.add_argument('sections', help='output sections')
    parser.add_argument('surf_grouped', help='output surface grouped')
    parser.add_argument('caps', type=json.loads, help='cap point ids')

    # run script
    main(parser.parse_args())
