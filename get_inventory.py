#!/usr/bin/env python

import csv
from collections import defaultdict, OrderedDict

from get_database import input_args


def main(db, geometries):
    properties = defaultdict(OrderedDict)
    params_read = ['age', 'gender', 'deliverable_category', 'vascular_state', 'treatment', 'other_disease',
                   'model_source', 'simulation_source']

    for geo in geometries:
        print('Running geometry ' + geo)
        params = db.get_params(geo)

        for pm in params_read:
            if params is None or pm not in params:
                val = 'unknown'
            else:
                val = params[pm]
            properties[geo][pm] = val

    with open('osmsc.csv', 'w', newline='') as csvfile:
        reader = csv.writer(csvfile, delimiter=',')

        # write header
        reader.writerow(['model_id'] + params_read)

        for geo in geometries:
            reader.writerow([geo] + [v for v in properties[geo].values()])


if __name__ == '__main__':
    descr = 'Generate a new surface mesh'
    d, g, _ = input_args(descr)
    main(d, g)
