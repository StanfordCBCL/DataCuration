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
    parser.add_argument('-g', '--geo', help='geometry')
    parser.add_argument('-s', '--study', help='study name')
    param = parser.parse_args()

    # get model database
    database = Database(param.study)

    # choose geometries to evaluate
    if param.geo in database.get_geometries():
        geometries = [param.geo]
    elif param.geo == 'select':
        geometries = database.get_geometries_select()
    else:
        geometries = database.get_geometries()

    return database, geometries
