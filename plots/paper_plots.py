#!/usr/bin/env python

import os
import re
import vtk
import argparse
import pdb

import numpy as np
import matplotlib.pyplot as plt

from collections import defaultdict

from vtk.util.numpy_support import numpy_to_vtk as n2v
from vtk.util.numpy_support import vtk_to_numpy as v2n

from get_database import Database
from vtk_functions import read_geo, write_geo, get_all_arrays


fsize = 20
plt.rcParams.update({'font.size': fsize})


def plot_area():
    db = Database()
    f_in = db.get_centerline_path('0003_0001')
    f_out = os.path.join(db.img_path, 'radius_vs_area', 'OSMSC_0003_0001_branch0')

    # get centerline
    geo = read_geo(f_in).GetOutput()
    arrays, _ = get_all_arrays(geo)

    # extract branch
    br = 0
    mask = arrays['BranchId'] == br

    # get plot quantities
    path = arrays['Path'][mask]
    area_slice = arrays['CenterlineSectionArea'][mask]
    area_vmtk = arrays['MaximumInscribedSphereRadius'][mask] ** 2 * np.pi

    print('factor', area_slice[0] / area_vmtk[0])

    # make plot
    fig, ax = plt.subplots(dpi=300, figsize=(6, 6))
    ax.plot(path, area_slice, 'r-')
    ax.plot(path, area_vmtk, 'b-')
    ax.legend(['Area from slicing', 'Area from MISR'])
    ax.set_xlim(left=0)
    ax.set_xticks([0, 2, 4])
    ax.set_xticklabels(['Inlet', '2', '4'])
    plt.xlabel('Branch path [cm]')
    plt.ylabel('Area [cm^2]')
    plt.grid()
    fig.savefig(f_out, bbox_inches='tight')


def plot_model_statistics():
    db = Database()
    geometries_paper = db.get_geometries_select('paper')

    pie = defaultdict(lambda: defaultdict(int))
    cats = ['deliverable_category', 'vascular_state', 'treatment', 'image_data_modality', 'paper_reference', 'gender']

    names = {'deliverable_category': 'Vascular anatmoy',
             'vascular_state': 'Vascular state',
             'treatment': 'Treatment',
             'image_data_modality': 'Imaging',
             'paper_reference': 'Literature reference',
             'gender': 'Gender'}

    # count all categories
    for geo in geometries_paper:
        _, err = db.get_bc_type(geo)
        if not err:
            pie['has_bc']['yes'] += 1
            params = db.get_params(geo)
            for cat in cats:
                name = params[cat].capitalize()
                if name == '' or 'Unpublished' in name:
                    name = 'None'
                pie[cat][name] += 1
        else:
            pie['has_bc']['no'] += 1

    # make plots
    fig, axs = plt.subplots(2, 2, dpi=300, figsize=(30, 20))

    selection = ['deliverable_category', 'vascular_state', 'treatment', 'paper_reference']
    for cat, ax in zip(selection, axs.ravel()):
        labels = np.array([re.sub(r'\([^)]*\)', '', c) for c in pie[cat].keys()])
        sizes = np.array(list(pie[cat].values()))
        order = np.argsort(sizes)

        print('num', np.sum(sizes))
        abs_size = lambda p: '{:.0f}'.format(p * np.sum(sizes) / 100)
        ax.pie(sizes[order], labels=labels[order], autopct=abs_size)
        ax.axis('equal')
        ax.set_title(names[cat], fontsize=40, pad=20)

    f_out = os.path.join(db.img_path, 'repository', 'repo_statistics')#.pgf
    fig.savefig(f_out, bbox_inches='tight')
    plt.close(fig)


def main():
    plot_model_statistics()
    plot_area()


if __name__ == '__main__':
    descr = 'Make plots for 3D-1D-0D paper'
    main()
