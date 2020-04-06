#!/usr/bin/env python

import pdb
import os
import shutil
import glob
import matplotlib.cm as cm
from collections import OrderedDict

import numpy as np

from get_database import input_args, Database, SVProject, SimVascular
from get_sim import write_bc


def write_svproj(db, geo):
    t = str(db.svproj.t)
    proj_head = ['<?xml version="1.0" encoding="UTF-8"?>',
                 '<projectDescription version="1.0">']
    proj_end = ['</projectDescription>']

    with open(db.get_svproj_file(geo), 'w+') as f:
        # write header
        for s in proj_head:
            f.write(s + '\n')

        # write images/segmentations
        for k, s in db.svproj.dir.items():
            f.write(t + '<' + k + ' folder_name="' + s + '"')
            if k == 'images':
                img = os.path.basename(db.get_img(geo))
                f.write('>\n')
                f.write(t*2 + '<image name="' + os.path.splitext(img)[0] + '" in_project="yes" path="' + img + '"/>\n')
                f.write(t + '</' + k + '>\n')
            # elif k == 'segmentations':
            else:
                f.write('/>\n')

        # write end
        for s in proj_end:
            f.write(s + '\n')


def write_model(db, geo):
    t = str(db.svproj.t)
    model_head = ['<?xml version="1.0" encoding="UTF-8" ?>',
                  '<format version="1.0" />',
                  '<model type="PolyData">',
                  t + '<timestep id="0">',
                  t*2 + '<model_element type="PolyData" num_sampling="0">']
    model_end = [t*3 + '<blend_radii />',
                 t*3 + '<blend_param blend_iters="2" sub_blend_iters="3" cstr_smooth_iters="2" lap_smooth_iters="50" '
                       'subdivision_iters="1" decimation="0.01" />',
                 t*2 + '</model_element>',
                 t + '</timestep>',
                 '</model>']

    # read boundary conditions
    bc_def, _ = db.get_bcs(geo)
    bc_def['spid']['wall'] = 0

    # get cap names
    caps = db.get_surface_names(geo, 'caps')
    caps += ['wall']

    # sort caps according to face id
    ids = np.array([repr(int(float(bc_def['spid'][c]))) for c in caps])
    order = np.argsort(ids)
    caps = np.array(caps)[order]
    ids = ids[order]

    # display colors for caps
    colors = cm.jet(np.linspace(0, 1, len(caps)))

    # write model file
    with open(db.get_svproj_mdl_file(geo), 'w+') as f:
        # write header
        for s in model_head:
            f.write(s + '\n')

        # write faces
        f.write(t*3 + '<faces>\n')
        for i, c in enumerate(caps):
            c_str = t*4 + '<face id="' + ids[i] + '" name="' + c + '" type='
            if c == 'wall':
                c_str += '"wall"'
            else:
                c_str += '"cap"'
            for j in range(3):
                c_str += ' color' + repr(j + 1) + '="' + repr(colors[i, j]) + '"'
            f.write(c_str + ' visible="true" opacity="1" />\n')
        f.write(t*3 + '</faces>\n')

        # write end
        for s in model_end:
            f.write(s + '\n')


def write_path_segmentation(db, geo):
    # SimVascular instance
    sv = SimVascular()

    # get paths
    p = OrderedDict()
    p['f_path_in'] = db.get_path_file(geo)
    p['f_path_out'] = os.path.join(db.get_svproj_dir(geo), db.svproj.dir['paths'])
    p['f_seg_in'] = db.get_seg_dir(geo)
    p['f_seg_out'] = os.path.join(db.get_svproj_dir(geo), db.svproj.dir['segmentations'])

    # assemble call string
    sv_string = [os.path.join(os.getcwd(), 'sv_get_path_segmentation.py')]
    for v in p.values():
        sv_string += [v]

    # execute SimVascular-Python
    out, err = sv.run_python_legacyio(sv_string)
    return err

    # check if all paths exist
    f_seg_in = db.get_seg_dir(geo)

    if not os.path.exists(f_seg_in[:-1]):
        return 'folder does not exist: ' + f_seg_in[:-1]
    if not glob.glob(f_seg_in):
        return 'folder is empty: ' + f_seg_in

    err = ''
    path_list = ''
    seg_list = ''
    for name in glob.glob(f_seg_in):
        if '.tcl' in name:
            continue

        # modify name
        # name = name.replace('_new', '')
        # name = name.replace('_final', '')
        # name = name.replace('_FINAL', '')

        path_file_name = os.path.join(p['f_path_out'], os.path.basename(name) + '.pth')
        if not os.path.exists(path_file_name):
            err += os.path.basename(path_file_name) + '\n'
        else:
            seg_list += name + ' '
            path_list += path_file_name + ' '

    if err:
        err = '\nmissing paths:\n' + err
        err += '\nfound paths:\n'
        for name in glob.glob(os.path.join(p['f_path_out'], '*')):
            err += os.path.basename(name) + '\n'

        return str(err)

    # get paths
    p = OrderedDict()
    p['path_list'] = '-path_list_in ' + path_list
    p['seg_list'] = '-seg_list_in ' + seg_list
    p['f_seg_out'] = '-seg_out ' + os.path.join(db.get_svproj_dir(geo), db.svproj.dir['segmentations'])

    # assemble call string
    sv_string = [os.path.join(os.getcwd(), 'sv_get_segmentation.py')]
    for v in p.values():
        sv_string += [v]

    # execute SimVascular-Python
    out, err = sv.run_python_legacyio(sv_string)
    return err


def copy_file(db, geo, src, trg_dir):
    trg = os.path.join(db.get_svproj_dir(geo), db.svproj.dir[trg_dir], os.path.basename(src))
    shutil.copy2(src, trg)


def make_folders(db, geo):
    # make all project sub-folders
    for s in db.svproj.dir.values():
        os.makedirs(os.path.join(db.get_svproj_dir(geo), s), exist_ok=True)

    # copy image
    copy_file(db, geo, db.get_img(geo), 'images')

    # copy volume mesh
    copy_file(db, geo, db.get_volume_mesh(geo), 'meshes')

    # copy surface mesh
    copy_file(db, geo, db.get_sv_surface(geo), 'meshes')
    copy_file(db, geo, db.get_sv_surface(geo), 'models')

    return True


def check_files(db, geo):
    # check if files exist
    if db.get_volume_mesh(geo) is None:
        return False, 'no volume mesh'
    if db.get_sv_surface(geo) is None:
        return False, 'no SV surface mesh'
    if db.get_img(geo) is None:
        return False, 'no medical image'
    return True, None


def main(db, geometries):
    for geo in geometries:
        print('Running geometry ' + geo)

        success, err = check_files(db, geo)
        if not success:
            print('  ' + err)
            continue

        make_folders(db, geo)
        write_svproj(db, geo)
        write_model(db, geo)
        err = write_path_segmentation(db, geo)
        if err:
            print(err)
        else:
            print('  success!')


if __name__ == '__main__':
    descr = 'Generate an svproject folder'
    d, g, _ = input_args(descr)
    main(d, g)
