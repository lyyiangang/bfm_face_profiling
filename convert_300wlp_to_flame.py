'''
Max-Planck-Gesellschaft zur Foerderung der Wissenschaften e.V. (MPG) is holder of all proprietary rights on this
computer program.
You can only use this computer program if you have closed a license agreement with MPG or you get the right to use
the computer program from someone who is authorized to grant you that right.
Any use of the computer program without a valid license is prohibited and liable to prosecution.
Copyright 2019 Max-Planck-Gesellschaft zur Foerderung der Wissenschaften e.V. (MPG). acting on behalf of its
Max Planck Institute for Intelligent Systems and the Max Planck Institute for Biological Cybernetics.
All rights reserved.
More information about FLAME is available at http://flame.is.tue.mpg.de.
For comments or questions, please email us at flame@tue.mpg.de
'''

import os
import numpy as np
import chumpy as ch
from psbody.mesh import Mesh
import sys
sys.path.append('./BFM_to_FLAME')
from smpl_webuser.serialization import load_model

import config
import argparse
from pathlib import Path
import logging
lg = logging.getLogger(__name__)
lg.setLevel(logging.INFO)

def load_facefp_head_model(v, f):
    trim_path = 'photometric_optimization/data/trim_verts_face.npz'
    tp = np.load(trim_path, allow_pickle=True) # trim_path: 3487
    # tp['map_verts'] : (5023, )
    # tp['idx_faces'], (6860,)
    # tp['idx_verts'], (3487,)
    faces = tp['map_verts'][f[tp['idx_faces']]]
    verts = v[tp['idx_verts']]
    return Mesh(verts, faces)

def convert_mesh(mesh, corr_setup):
    v = np.vstack((mesh.v, np.zeros_like(mesh.v)))
    return Mesh(corr_setup['mtx'].dot(v), corr_setup['f_out'])

def convert_BFM_mesh_to_FLAME(FLAME_model_fname, BFM_mesh_fname, FLAME_out_fname):
    '''
    Convert Basel Face Model mesh to a FLAME mesh
    \param FLAME_model_fname        path of the FLAME model
    \param BFM_mesh_fname           path of the BFM mesh to be converted
    \param FLAME_out_fname          path of the output file
    '''

    # Regularizer weights for jaw pose (i.e. opening of mouth), shape, and facial expression.
    # Increase regularization in case of implausible output meshes. 
    w_pose = 1e-4
    w_shape = 1e-3
    w_exp = 1e-4

    if not os.path.exists(os.path.dirname(FLAME_out_fname)):
        os.makedirs(os.path.dirname(FLAME_out_fname))

    if not os.path.exists(BFM_mesh_fname):
        lg.info('BFM mesh not found %s' % BFM_mesh_fname)
        return
    BFM_mesh = Mesh(filename=BFM_mesh_fname)
    # BFM_mesh.show()
    if not os.path.exists(FLAME_model_fname):
        lg.info('FLAME model not found %s' % FLAME_model_fname)
        return
    flame_head_model = load_model(FLAME_model_fname)#verts:(5023,)
    cached_map_file = './BFM_to_FLAME/data/BFM_to_FLAME_corr.npz'
    if not os.path.exists(cached_map_file):
        lg.info('Cached mapping not found')
        return
    cached_data = np.load(cached_map_file, allow_pickle=True, encoding="latin1")

    BFM2017_corr = cached_data['BFM2017_corr'].item()
    BFM2009_corr = cached_data['BFM2009_corr'].item()
    BFM2009_cropped_corr = cached_data['BFM2009_cropped_corr'].item()
    # reverse coordiante system
    BFM_mesh.v[:, 1] *= -1
    # exit()
    if (2*BFM_mesh.v.shape[0] == BFM2017_corr['mtx'].shape[1]) and (BFM_mesh.f.shape[0] == BFM2017_corr['f_in'].shape[0]):
        lg.info(f'using bfm 2017 model')
        conv_mesh = convert_mesh(BFM_mesh, BFM2017_corr)
    elif (2*BFM_mesh.v.shape[0] == BFM2009_corr['mtx'].shape[1]) and (BFM_mesh.f.shape[0] == BFM2009_corr['f_in'].shape[0]):
        conv_mesh = convert_mesh(BFM_mesh, BFM2009_corr)
        lg.info(f'using bfm 2009 model')
    elif (2*BFM_mesh.v.shape[0] == BFM2009_cropped_corr['mtx'].shape[1]) and (BFM_mesh.f.shape[0] == BFM2009_cropped_corr['f_in'].shape[0]):
        conv_mesh = convert_mesh(BFM_mesh, BFM2009_cropped_corr) # (5023, 3), conv_mesh is simillar with flame head model
        lg.info(f'using bfm 2009 cropped model')
    else:
        lg.info('Conversion failed - input mesh does not match any setup')
        return

    FLAME_mask_ids = cached_data['FLAME_mask_ids'] # (2084,)
    scale = ch.ones(1)
    lg.info(f'conv_mesh.v.shape:{conv_mesh.v.shape}')
    v_target = scale*ch.array(conv_mesh.v)
    dist = v_target[FLAME_mask_ids]-flame_head_model[FLAME_mask_ids]
    pose_reg = flame_head_model.pose[3:]
    shape_reg = flame_head_model.betas[:300]
    exp_reg = flame_head_model.betas[300:]
    obj = {'dist': dist, 'pose_reg': w_pose*pose_reg, 'shape_reg': w_shape*shape_reg, 'exp_reg': w_exp*exp_reg}
    lg.info('initial estimate')
    ch.minimize(obj, x0=[scale, flame_head_model.trans, flame_head_model.pose[:3]])
    lg.info(f'estimate expression')
    ch.minimize(obj, x0=[scale, flame_head_model.trans, flame_head_model.pose[np.hstack((np.arange(3), np.arange(6,9)))], flame_head_model.betas])

    v_out = flame_head_model.r/scale.r
    # Mesh(v_out, flame_head_model.f).write_obj(FLAME_out_fname)

    facefp_mesh = load_facefp_head_model(v_out, flame_head_model.f)
    # facefp_mesh.write_obj('facefp.obj')

    param = {}
    param['verts0'] = facefp_mesh.v[None, :, :]
    lg.info(f'saving {FLAME_out_fname}')
    np.save(FLAME_out_fname, param, allow_pickle= True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='convert 300wlp model to flame head model')
    parser.add_argument('--obj', type=str, default ='./data/AFW_AFW_18489332_7_7_4.jpg.obj',  help='obj file')
    parser.add_argument('--output', type=str, default ='',  help='where to save fitted result')
    
    args = parser.parse_args()
    if args.output == '':
        output = 'outputs/fitted_flame/' + Path(args.obj).name + '.npy'
    convert_BFM_mesh_to_FLAME(config.flame_head_model,args.obj , output)
    lg.info('Conversion finished')