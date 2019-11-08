#!/usr/bin/env python

import numpy as np
import sys, os, re, tkinter, pdb

def get_dic(st):
    assert len(st) % 2 == 0, 'array order is not right'
    dc = {}
    for i in range(len(st) // 2):
        a = st[i * 2]
        b = st[i * 2 + 1]

        # convert dict value to float if possible
        try:
            val = float(b)
        except:
            val = b
        dc[a] = val

    return dc

def read_array(r, name):
    exe = r.tk.eval('array get ' + name)

    # extract dict components
    st = list(filter(None, re.split(' |({|})', exe)))

    # check for sub-dicts
    limits = [i for i, s in enumerate(st) if s == '{' or s == '}']

    # assemble string into dict
    if len(limits) == 0:
        dc = get_dic(st)
    else:
        dc = {}
        for a, b in zip(limits[::2], limits[1::2]):
            dc[st[a - 1]] = get_dic(st[a + 1:b])

    return dc

def get_bcs():
    # name of geometry
    geo = '0110_0000'

    # folder for tcl files with boundary conditions
    fpath_bc = '/home/pfaller/work/osmsc/VMR_tcl_repository_scripts/repos_ready_cpm_scripts'

    # folder for simulation files
    fpath_sim = '/home/pfaller/work/osmsc/data_uploaded'

    # evaluate tcl-script to extract variables
    r = tkinter.Tk()
    r.tk.eval('source ' + os.path.join(fpath_bc, geo + '-bc.tcl'))

    # generate dictionaries from tcl output
    sim_bc = read_array(r, 'sim_bc')
    sim_spid = read_array(r, 'sim_spid')
    #sim_preid = read_array(r, 'sim_preid')
    #sim_spname = read_array(r, 'sim_spname')

    return sim_bc, sim_spid
