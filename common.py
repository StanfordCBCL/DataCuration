#!/usr/bin/env python

import argparse
from get_database import Database


def input_args(description):
    """
    Handles input arguments to scripts
    Args:
        description: script description (hgelp string)

    Returns:
        database: Database object for study
        geometries: list of geometries to evaluate
    """
    # parse input arguments
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('study', help='study name')
    parser.add_argument('-g', '--geo', help='individual geometry or subset name')
    param = parser.parse_args()

    # get model database
    database = Database(param.study)

    # choose geometries to evaluate
    if param.geo in database.get_geometries():
        geometries = [param.geo]
    elif param.geo is None:
        geometries = database.get_geometries()
    elif param.geo[-1] == ':':
        geo_all = database.get_geometries()
        geo_first = geo_all.index(param.geo[:-1])
        geometries = geo_all[geo_first:]
    else:
        geometries = database.get_geometries_select(param.geo)

    return database, geometries, param
