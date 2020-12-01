#!/usr/bin/env python

import pdb
import os
import csv
import numpy as np
from collections import defaultdict, OrderedDict

from get_database import Database
from get_sv_project import coronary_sv_to_oned

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick


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
        bc_def = db.get_bcs(geo)

        for pm in params_read:
            if bc_def['params'] is None or pm not in bc_def['params']:
                val = 'unknown'
            else:
                val = bc_def['params'][pm]
            properties[geo][pm] = val

    # check if simulation is to be published
    for geo in geometries:
        if geo in geometries_paper:
            publish = 'yes'
        else:
            publish = 'no'
        properties[geo]['publish'] = publish

    # get inflow type
    for geo in geometries:
        bc_def = db.get_bcs(geo)
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


def time_constant():
    db = Database()
    # geometries = db.get_geometries()
    geometries = db.get_geometries_select('paper')

    # color by category
    colors = {'Cerebrovascular': 'k',
              'Coronary': 'r',
              'Aortofemoral': 'm',
              'Pulmonary': 'c',
              'Congenital Heart Disease': 'y',
              'Aorta': 'b',
              'Animal and Misc': '0.75'}

    fig1, ax1 = plt.subplots(dpi=400, figsize=(15, 6))

    geos = []
    i = 0
    for geo in geometries:
        params = db.get_bcs(geo)
        time, _ = db.get_inflow(geo)

        # skip geometries without BCs
        if params is None:
            continue

        col = colors[params['params']['deliverable_category']]

        # collect all time constants
        tau_bc = []
        for cp, bc in params['bc'].items():
            if 'Rd' in bc:
                tau = bc['Rd'] * bc['C']
                m = '_'
            elif 'Pim' in bc:
                p = {}
                cor = coronary_sv_to_oned(bc)
                p['R1'], p['R2'], p['R3'], p['C1'], p['C2'] = (cor['Ra1'], cor['Ra2'], cor['Rv1'], cor['Ca'], cor['Cc'])
                tau1 = p['C1'] / (1 / (p['R2'] + p['R1']) + 1 / p['R3'])
                tau2 = p['C2'] / (1 / (p['R2'] + p['R3']) + 1 / p['R1'])
                tau = tau1 + tau2
                if 'l' in cp.lower():
                    m = 'x'
                else:
                    m = 'o'
            else:
                continue

            #  tau in cardiac cycles
            tau /= time[-1]

            ax1.plot(i, tau, marker=m, color=col)
            tau_bc += [tau]

        # skip geometries without RCR
        if len(tau_bc) == 0:
            continue

        # ax1.boxplot(tau_bc, positions=[i])
        ax1.plot([i, i], [np.min(tau_bc), np.max(tau_bc)], '-', color=col)

        geos += [geo]
        i += 1

    # plt.yscale('log')
    ax1.xaxis.grid('minor')
    ax1.yaxis.grid(True)
    ax1.set_ylim(0, 13)
    plt.xticks(np.arange(len(geos)), geos, rotation='vertical')
    plt.ylabel('Time constant [cycles]')
    fname = os.path.join(db.fpath_gen, 'time_constants.png')
    fig1.savefig(fname, bbox_inches='tight')


if __name__ == '__main__':
    # main()
    time_constant()
