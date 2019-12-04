#!/usr/bin/env python

import re
import tkinter


def get_dic(st):
    """
    assemble string into dict
    :param st: string
    :return: dictionary
    """
    assert len(st) % 2 == 0, 'array order is not right'
    dc = {}
    for i in range(len(st) // 2):
        a = st[i * 2]
        b = st[i * 2 + 1]

        # convert dict value to float if possible
        try:
            val = float(b)
        except ValueError:
            val = b
        dc[a] = val

    return dc


def read_array(r, name, subdic=True):
    """
    read tcl array
    :param r: tkinter object
    :param name: array name
    :param subdic: group array in sub-dicts if possible?
    :return: dictionary
    """
    exe = r.tk.eval('array get ' + name)

    # extract dict components
    st = list(filter(None, re.split(' |({|})', exe)))

    # replace empty dicts by -1
    st_clean = []
    i = 0
    while i < len(st) - 1:
        if st[i] == '{' and st[i + 1] == '}':
            st_clean.append('-1')
            i += 1
        else:
            st_clean.append(st[i])
        i += 1
    if i == len(st) - 1:
        st_clean.append(st[-1])

    # check for sub-dicts
    limits = [i for i, s in enumerate(st_clean) if s == '{' or s == '}']
    assert len(limits) % 2 == 0, 'array order is not right'

    # one dictionary
    if len(limits) == 0:
        dc = get_dic(st_clean)

    # several sub-dictionaries
    else:
        dc = {}
        for a, b in zip(limits[::2], limits[1::2]):
            if not subdic:
                dc[st_clean[a - 1]] = st_clean[a + 1:b]
            else:
                dc[st_clean[a - 1]] = get_dic(st_clean[a + 1:b])
    return dc


def get_bcs(tcl, tcl_bc):
    # evaluate tcl-script to extract variables for boundary conditions
    r_bc = tkinter.Tk()
    r_bc.tk.eval('source ' + tcl_bc)

    # generate dictionaries from tcl output
    sim_bc = read_array(r_bc, 'sim_bc')
    sim_spid = read_array(r_bc, 'sim_spid')
    sim_preid = read_array(r_bc, 'sim_preid')
    sim_spname = read_array(r_bc, 'sim_spname', False)

    bcs = {'bc': sim_bc, 'spid': sim_spid, 'preid': sim_preid, 'spname': sim_spname}

    # evaluate tcl-script to extract variables for general simulation parameters
    r = tkinter.Tk()
    # todo: handle non-existing function math_addVectors
    r.tk.eval('source ' + tcl)
    units = r.tk.getvar('sim_units')

    return bcs, units

