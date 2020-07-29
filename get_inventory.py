#!/usr/bin/env python

import pdb
import csv
from collections import defaultdict, OrderedDict

from get_database import Database


def main():
    db = Database()
    geometries = db.get_geometries()
    geometries_paper = db.get_geometries_select('paper')

    properties = defaultdict(OrderedDict)
    params_read = ['age', 'gender', 'deliverable_category', 'vascular_state', 'treatment', 'other_disease',
                   'sim_physio_state', 'image_data_modality', 'paper_reference',
                   'model_source', 'simulation_source', 'sim_steps_per_cycle']

    # extract simulation parameters
    for geo in geometries:
        print('Running geometry ' + geo)
        params = db.get_params(geo)

        for pm in params_read:
            if params is None or pm not in params:
                val = 'unknown'
            else:
                val = params[pm]
            properties[geo][pm] = val

    # check if simulation is to be published
    for geo in geometries:
        if geo in geometries_paper:
            publish = 'yes'
        else:
            publish = 'no'
        properties[geo]['publish'] = publish

    # check if boundary conditions exist
    for geo in geometries:
        _, err = db.get_bc_type(geo)
        if not err:
            bc = 'yes'
        else:
            bc = 'no'
        properties[geo]['has_bcs'] = bc

    # get inflow type
    for geo in geometries:
        bc_def, params = db.get_bcs(geo)
        if bc_def is None:
            inflow = 'none'
        else:
            inflow = bc_def['bc']['inflow']['type']
        properties[geo]['inflow_type'] = inflow

    with open('osmsc2.csv', 'w', newline='') as csvfile:
        reader = csv.writer(csvfile, delimiter=',')

        # write header
        reader.writerow(['model_id'] + list(properties[geometries[0]].keys()))

        # write rows
        for geo in geometries:
            reader.writerow([geo] + [v for v in properties[geo].values()])


if __name__ == '__main__':
    main()
