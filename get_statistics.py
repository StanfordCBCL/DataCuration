#!/usr/bin/env python

import contextlib
import csv
import glob
import io
import os
import pdb
import re
import shutil
import sys
import argparse
import subprocess

import numpy as np

from get_database import Database, SimVascular, Post, input_args
from simulation_io import get_dict

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from matplotlib.offsetbox import OffsetImage, AnnotationBbox


def print_error(db, geometries):
    # get post-processing constants
    post = Post()

    folder = db.get_statistics_dir()

    # get simulation errors
    res_all = get_dict(db.get_1d_3d_comparison())

    # use only geometries in selection
    res = {}
    for k, v in res_all.items():
        if k in geometries:
            res[k] = v

    # make plots
    plot_err_bar(res, folder)
    plot_err_scatter(res, folder)
    plot_img_scatter(db, res, folder)


def plot_err_bar(res, folder):
    # get post-processing constants
    post = Post()

    domain = ['cap', 'int']
    metric0 = ['avg', 'max']#, 'sys', 'dia'

    fig1, ax1 = plt.subplots(dpi=300, figsize=(12, 6))
    for f in post.fields:
        for d in domain:
            for m0 in metric0:
                labels = []
                values = []
                for k in res:
                    labels += [k]
                    values += [res[k][f][d][m0]['all']]

                labels = np.array(labels)
                values = np.array(values)
                xtick = np.arange(len(values))

                order = np.argsort(values)
                plot_bar(fig1, ax1, xtick, values, labels, order, m0, f, d, folder, 'sorted')

                order = np.argsort(labels)
                plot_bar(fig1, ax1, xtick, values, labels, order, m0, f, d, folder, 'aplhabetical')
    plt.close(fig1)


def plot_bar(fig1, ax1, xtick, values, labels, order, m0, f, d, folder, name):
    plt.cla()
    plt.yscale('log')
    ax1.bar(xtick, values[order])
    ax1.yaxis.grid(True)
    ax1.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1, decimals=1))
    ax1.set_ylim(0.001, 1)
    plt.xticks(xtick, labels[order], rotation='vertical')
    plt.ylabel(m0 + ' ' + f + ' error at ' + d + ' [1]')
    fname = os.path.join(folder, 'error_' + name + '_' + f + '_' + d + '_' + m0 + '.png')
    fig1.savefig(fname, bbox_inches='tight')


def plot_err_scatter(res, folder):
    # plot different correlations of errors
    combinations = [['flow', 'pressure'], ['area', 'flow'], ['area', 'pressure']]

    domain = ['cap', 'int']
    metric0 = ['avg', 'max']#, 'sys', 'dia'

    fig1, ax1 = plt.subplots(dpi=300, figsize=(12, 6))
    for c in combinations:
        fx = c[0]
        fy = c[1]

        for d in domain:
            for m0 in metric0:
                ux = '1'
                uy = '1'
                scale = 100

                plt.cla()
                for geo, err in res.items():
                    x = err[fx][d][m0]['all'] * scale
                    y = err[fy][d][m0]['all'] * scale

                    ax1.plot(x, y, 'o')
                    ax1.annotate(geo, (x, y))

                plt.xlabel(m0 + ' ' + fx + ' error at ' + d + ' [' + ux + ']')
                plt.ylabel(m0 + ' ' + fy + ' error at ' + d + ' [' + uy + ']')
                plt.xscale('log')
                plt.yscale('log')
                plt.grid()
                ax1.xaxis.set_major_formatter(mtick.PercentFormatter(decimals=1))
                ax1.yaxis.set_major_formatter(mtick.PercentFormatter(decimals=1))
                # if m0 == 'avg':
                #     ax1.set_xlim(1, 200)
                #     ax1.set_ylim(1, 200)
                # elif m0 == 'max':
                #     ax1.set_xlim(1, 1000)
                #     ax1.set_ylim(1, 1000)
                ax1.set_xlim(0.1, 100)
                ax1.set_ylim(0.1, 100)

                fname = 'error_correlation_' + fx + '_' + fy + '_' + d + '_' + m0 + '.png'
                fpath = os.path.join(folder, fname)
                fig1.savefig(fpath, bbox_inches='tight')
    plt.close(fig1)


def plot_img_scatter(db, res, folder):
    fsize = 50
    fig1, ax1 = plt.subplots(dpi=100, figsize=(60, 30))
    plt.rcParams.update({'font.size': fsize})

    domain = ['cap', 'int']
    metric0 = ['avg', 'max']#, 'sys', 'dia'

    for d in domain:
        for m0 in metric0:
            fx = 'flow'
            fy = 'pressure'

            ux = '1'
            uy = '1'
            scale = 100

            plt.cla()
            for geo, err in res.items():
                x = err[fx][d][m0]['all'] * scale
                y = err[fy][d][m0]['all'] * scale
                ab = AnnotationBbox(OffsetImage(plt.imread(db.get_png(geo))), (x, y), frameon=False)
                ax1.scatter(x, y, c='k')
                ax1.add_artist(ab)

            plt.xlabel(m0 + ' ' + fx + ' error at ' + d + ' [' + ux + ']', fontsize=fsize)
            plt.ylabel(m0 + ' ' + fy + ' error at ' + d + ' [' + uy + ']', fontsize=fsize)
            plt.xscale('log')
            plt.yscale('log')
            plt.grid()
            ax1.xaxis.set_major_formatter(mtick.PercentFormatter(decimals=1))
            ax1.yaxis.set_major_formatter(mtick.PercentFormatter(decimals=1))
            # if m0 == 'avg':
            #     ax1.set_xlim(1, 200)
            #     ax1.set_ylim(1, 200)
            # elif m0 == 'max':
            #     ax1.set_xlim(1, 1000)
            #     ax1.set_ylim(1, 1000)
            ax1.set_xlim(0.1, 100)
            ax1.set_ylim(0.1, 100)

            fname = 'error_correlation_' + fx + '_' + fy + '_' + d + '_' + m0 + '_img.png'
            fpath = os.path.join(folder, fname)
            fig1.savefig(fpath, bbox_inches='tight')
    plt.close(fig1)


def print_statistics(db, geometries):
    res_all = get_dict(db.get_log_file_1d())

    # use only geometries in selection
    res = {}
    for k, v in res_all.items():
        if k in geometries:
            res[k] = v

    # sort errors
    success = '1D simulation\nsuccessful'
    for k, v in res.items():
        if '3d geometry has multiple inlets' in v:
            res[k] = 'Multiple inlets'
        if 'Inlet group id is not 0 or number of centerlines is not equal to the number of outlets' in v:
            res[k] = 'Centerline extraction\nfailed'
        if 'float division by zero' in v:
            res[k] = '3D geometry is corrupted'
        if 'boundary conditions not implemented (coronary)' in v:
            res[k] = 'Coronary\nboundary conditions'
        if 'The number of BC values' in v:
            res[k] = 'Missing boundary conditions'
        if 'unconverged' in v:
            res[k] = '1D simulation\nunconverged'
        if 'bifurcation with less than 2 outflows detected' in v:
            res[k] = 'Bifurcation is at outlet'
        if 'KeyError(None,)' in v:
            res[k] = 'Bifurcation is at inlet'
        if 'Centerline consist of more than one region' in v:
            res[k] = 'Centerline consists of >1 piece'
        if 'success' in v:
            res[k] = success
        if k == '0001_0001' or k == '0106_0001':
            res[k] = '3D geometry contains a loop'

    errors = np.array([k for k in res.values()])

    # count errors
    num_errors = {}
    for err in np.unique(errors):
        num_errors[err] = {}
        num_errors[err]['n'] = np.sum(errors == err)
        num_errors[err]['geos'] = [k for k, v in res.items() if v == err]

    for err, geos in num_errors.items():
        print(err)
        print(geos['geos'])

    for err in num_errors.keys():
        g_string = err + '\n'
        for g in num_errors[err]['geos']:
            g_string += g + '\n'
        num_errors[err]['g_string'] = g_string[:-1]

    # make a montage for every error with the geometries
    montage = num_errors.copy()
    montage['all'] = {'geos': geometries}
    for e in montage.keys():
        err = e.replace(' ', '_')
        err = err.replace('\n', '_')

        g_string = ['/usr/bin/montage']
        for g in montage[e]['geos']:
            extensions = ['_sim.png', '_sim.jpg', '_model.jpg', '_vol.png']
            for ext in extensions:
                src = os.path.join(db.fpath_png, 'OSMSC' + g + ext)
                if os.path.exists(src):
                    break
            g_string += [src]
        g_string += [os.path.join(db.get_statistics_dir(), 'models_' + err.lower()) + '.png']
        subprocess.Popen(g_string)

    # print statistics
    num_sim = len(res)
    print('number of simulations: ' + repr(num_sim))

    geo_id = [geo[:6] for geo in geometries]
    print('number of unique geometries: ' + repr(np.unique(geo_id).shape[0]))

    geo_pat = [geo[:4] for geo in geometries]
    print('number of unique patients: ' + repr(np.unique(geo_pat).shape[0]))

    # Pie chart, where the slices will be ordered and plotted counter-clockwise:
    labels_error = list(num_errors.keys())
    labels = [num_errors[v]['g_string'] for v in labels_error]
    sizes = [num_errors[v]['n'] for v in labels_error]
    assert np.sum(np.array(sizes)) == num_sim, 'wrong number of errors'

    explode = [0] * len(labels_error)
    explode[labels_error.index(success)] = 0.1
    explode = tuple(explode)

    fig1, ax1 = plt.subplots(dpi=300, figsize=(4, 4))
    ax1.axis('equal')
    # autopct = '%1.1f%%'
    # autopct = lambda p: '{:.0f}'.format(p * total / 100)

    labels_pie = ax1.pie(sizes, explode=explode, labels=labels_error,
                         autopct=lambda p: '{:.0f}'.format(p * num_sim / 100), startangle=90)
    #textprops={'weight': 'bold', labeldistance=1)

    # for label in labels_pie[1]:
    #     label.set_horizontalalignment('center')

    # plt.tight_layout()

    fig1.savefig(os.path.join(db.get_statistics_dir(), 'statistics.png'), bbox_inches='tight')

    # plt.cla()
    # ax1.pie(sizes, explode=explode, labels=labels, autopct=lambda p: '{:.0f}'.format(p * num_sim / 100), startangle=90)
    # fig1.savefig(os.path.join(db.get_statistics_dir(), 'statistics_geometries.png'))

    plt.close()


def main(db, geometries, params):
    print_statistics(db, geometries)
    print_error(db, geometries)

if __name__ == '__main__':
    descr = 'Automatically create, run, and post-process 1d-simulations'
    d, g, p = input_args(descr)
    main(d, g, p)
