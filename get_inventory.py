#!/usr/bin/env python

import pdb
import os
import csv
import numpy as np
from collections import defaultdict, OrderedDict

from get_database import Database, Post
from get_sv_project import coronary_sv_to_oned
from common import get_dict

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

# color by category


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
    db = Database('1spb_length')
    post = Post()
    geometries = db.get_geometries()
    # geometries = db.get_geometries_select('paper')

    # get numerical time constants
    res_num = get_dict(db.get_convergence_path())

    fig1, ax1 = plt.subplots(dpi=400, figsize=(15, 6))

    geos = []
    i = 0
    for geo in geometries:
        params = db.get_bcs(geo)
        time, _ = db.get_inflow(geo)

        # collect all time constants
        tau_bc = db.get_time_constants(geo)

        # skip geometries without RCR
        if len(tau_bc) == 0:
            continue

        col = post.colors[params['params']['deliverable_category']]
        if geo in res_num:
            tau_num = res_num[geo]['tau']
            for j in range(len(tau_num['flow'])):
                # ax1.plot(i, tau_num[geo]['flow'][j], marker='x', color=col)
                ax1.plot(i, tau_num['pressure'][j], marker='x', color=col)

            print(geo + ' tau_num = ' + '{:2.1f}'.format(np.mean(tau_num['pressure'])))

        # ax1.boxplot(tau_bc, positions=[i])
        ax1.plot([i, i], [np.min(tau_bc), np.max(tau_bc)], '-', color=col)
        ax1.plot(i, np.min(tau_bc), '_', color=col)
        ax1.plot(i, np.max(tau_bc), '_', color=col)

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
