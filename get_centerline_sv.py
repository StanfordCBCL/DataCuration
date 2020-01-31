#!/usr/bin/env python

try:
    import sv
except ImportError:
    raise ImportError('Run with sv --python -- this_script.py')

import json
import argparse
import pdb

def main(p):
    # set SimVascular internal names (arbitrary)
    names = {'cent': 'centerlines_in', 'surf': 'surface', 'lines': 'centerline_out', 'sections': 'sections_out'}

    # get model paths
    paths = {'cent': p.cent, 'surf': p.surf, 'lines': p.lines, 'sections': p.sections}

    # add files to repository
    for k in ['cent', 'surf']:
        sv.Repository.ReadXMLPolyData(names[k], paths[k])

    # todo: execute complete VMTK centerline toolchain

    # execute VMTK function
    sv.VMTKUtils.CenterlineSections(names['cent'], names['surf'], names['lines'], names['sections'])

    # write from repository to file
    for k in ['lines', 'sections']:
        sv.Repository.WriteXMLPolyData(names[k], paths[k])

    # remove from repository
    for v in names.values():
        sv.Repository.Delete(v)


if __name__ == '__main__':
    # parse input arguments
    parser = argparse.ArgumentParser(description='generate centerline in SimVascular')
    parser.add_argument('cent', help='input centerline')
    parser.add_argument('surf', help='input surface')
    parser.add_argument('lines', help='output centerline')
    parser.add_argument('sections', help='output sections')
    parser.add_argument('inlet', type=json.loads, help='inlet center')
    parser.add_argument('outlets', type=json.loads, help='outlet centers')

    # run script
    main(parser.parse_args())
