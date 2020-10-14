#!/usr/bin/env python

import os
import sys
import re
import vtk
import argparse
import pdb

import numpy as np
import matplotlib.pyplot as plt

from collections import defaultdict

from vtk.util.numpy_support import numpy_to_vtk as n2v
from vtk.util.numpy_support import vtk_to_numpy as v2n

sys.path.append('..')
from get_database import Database
from vtk_functions import read_geo, write_geo, get_all_arrays


fsize = 20
plt.rcParams.update({'font.size': fsize})


def plot_pressure():
    db = Database()
    geo = '0107_0001'

def main():
    plot_model_statistics()


if __name__ == '__main__':
    descr = 'Make plots for 3D-1D-0D paper'
    main()
