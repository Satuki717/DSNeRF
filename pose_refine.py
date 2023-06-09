import os, sys
import numpy as np
import imageio
import json
import random
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm, trange
from PIL import Image

import matplotlib.pyplot as plt

from run_nerf_7scenes import create_nerf, render_path
from run_nerf_helpers import *

from load_llff import load_llff_data, load_colmap_depth, load_colmap_llff, make_depths

from data import RayDataset
from torch.utils.data import DataLoader

from utils.generate_renderpath import generate_renderpath
import cv2

device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")

np.random.seed(0)
DEBUG = False

# define functions
def load_initial_pose(retrieval_list_path,database_dir):
    retrieval_list = np.loadtxt(retrieval_list_path, dtype=str, comments=['chess', 'fire', 'office', 'pumpkin', 'redkitchen', 'stairs'], usecols=(0,1))
    # print(retrieval_list[:5]) # N * 2

    coarse_pose_list = []
    for i in range(len(retrieval_list)):
        pose_path = os.path.join(database_dir,retrieval_list[i,1]).replace('color.png','pose.txt')
        pose = np.asarray(np.loadtxt(pose_path))
        coarse_pose_list.append(pose)
    coarse_pose_list = np.stack(coarse_pose_list, 0) # N * 4 * 4

    return retrieval_list, coarse_pose_list

# argument parse
def config_parser():

    import configargparse
    parser = configargparse.ArgumentParser()
    parser.add_argument('--config', is_config_file=True, 
                        help='config file path')
    parser.add_argument("--expname", type=str, 
                        help='experiment name')
    parser.add_argument("--basedir", type=str, default='./logs/', 
                        help='where to store ckpts and logs')
    parser.add_argument("--database_path", type=str, default='/home/kai/data/yyuan/7scenes', 
                        help='image database directory')
    parser.add_argument("--retrieve_list_path", type=str, default='/datasw/code/Camera_localization_diff_render_refine/inference_coarse_results/retrieval_coarse_estimation/retrieval_list.txt', 
                        help='input initial coarse pose file directory')

    # NeRF settings
    # training options
    parser.add_argument("--netdepth", type=int, default=8, 
                        help='layers in network')
    parser.add_argument("--netwidth", type=int, default=256, 
                        help='channels per layer')
    parser.add_argument("--netdepth_fine", type=int, default=8, 
                        help='layers in fine network')
    parser.add_argument("--netwidth_fine", type=int, default=256, 
                        help='channels per layer in fine network')
    parser.add_argument("--N_rand", type=int, default=32*32*4, 
                        help='batch size (number of random rays per gradient step)')
    parser.add_argument("--lrate", type=float, default=5e-4, 
                        help='learning rate')
    parser.add_argument("--lrate_decay", type=int, default=250, 
                        help='exponential learning rate decay (in 1000 steps)')
    parser.add_argument("--chunk", type=int, default=1024*8, 
                        help='number of rays processed in parallel, decrease if running out of memory')
    parser.add_argument("--netchunk", type=int, default=1024*64, 
                        help='number of pts sent through network in parallel, decrease if running out of memory')
    parser.add_argument("--no_batching", action='store_true', 
                        help='only take random rays from 1 image at a time')
    parser.add_argument("--no_reload", action='store_true', 
                        help='do not reload weights from saved ckpt')
    parser.add_argument("--ft_path", type=str, default=None, 
                        help='specific weights npy file to reload for coarse network')

    # rendering options
    parser.add_argument("--N_samples", type=int, default=64, 
                        help='number of coarse samples per ray')
    parser.add_argument("--N_importance", type=int, default=0,
                        help='number of additional fine samples per ray')
    parser.add_argument("--use_viewdirs", action='store_true', 
                        help='use full 5D input instead of 3D')
    parser.add_argument("--perturb", type=float, default=1.,
                        help='set to 0. for no jitter, 1. for jitter')
    parser.add_argument("--i_embed", type=int, default=0, 
                        help='set 0 for default positional encoding, -1 for none')
    parser.add_argument("--multires", type=int, default=10, 
                        help='log2 of max freq for positional encoding (3D location)')
    parser.add_argument("--multires_views", type=int, default=4, 
                        help='log2 of max freq for positional encoding (2D direction)')
    parser.add_argument("--raw_noise_std", type=float, default=0., 
                        help='std dev of noise added to regularize sigma_a output, 1e0 recommended')
    # training options
    # deepvoxels flags

    # dataset options
    parser.add_argument("--dataset_type", type=str, default='llff', 
                        help='options: llff / blender / deepvoxels')
    parser.add_argument("--testskip", type=int, default=8, 
                        help='will load 1/N images from test/val sets, useful for large datasets like deepvoxels')
    
    # blender flags
    parser.add_argument("--white_bkgd", action='store_true', 
                        help='set to render synthetic data on a white bkgd (always use for dvoxels)')
    parser.add_argument("--half_res", action='store_true', 
                        help='load blender synthetic data at 400x400 instead of 800x800')
    
    ## llff flags
    parser.add_argument("--factor", type=int, default=8, 
                        help='downsample factor for LLFF images')
    parser.add_argument("--no_ndc", action='store_true', 
                        help='do not use normalized device coordinates (set for non-forward facing scenes)')
    parser.add_argument("--lindisp", action='store_true', 
                        help='sampling linearly in disparity rather than depth')
    parser.add_argument("--spherify", action='store_true', 
                        help='set for spherical 360 scenes')
    parser.add_argument("--llffhold", type=int, default=8, 
                        help='will take every 1/N images as LLFF test set, paper uses 8')

    # logging/saving options
    # debug

    # new experiment by kangle
    parser.add_argument("--alpha_model_path", type=str, default=None,
                        help='predefined alpha model')
    parser.add_argument("--no_coarse", action='store_true',
                        help="Remove coarse network.")

    return parser

def optimize():

    # initial settings
    parser = config_parser()
    args = parser.parse_args()

    hwf = [480, 640, 585]

    # Create log dir and copy the config file
    basedir = args.basedir
    expname = args.expname
    os.makedirs(os.path.join(basedir, expname), exist_ok=True)
    f = os.path.join(basedir, expname, 'args.txt')
    with open(f, 'w') as file:
        for arg in sorted(vars(args)):
            attr = getattr(args, arg)
            file.write('{} = {}\n'.format(arg, attr))
    if args.config is not None:
        f = os.path.join(basedir, expname, 'config.txt')
        with open(f, 'w') as file:
            file.write(open(args.config, 'r').read())

    # load NeRF module
    render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer = create_nerf(args)

    global_step = start

    bds_dict = {
        'near' : near,
        'far' : far,
    }
    render_kwargs_train.update(bds_dict)
    render_kwargs_test.update(bds_dict)

    # load initial image and pose
    retrieval_list, pose_list = load_initial_pose(args.retrieve_list_path, args.database_path)

    for i, img_pair in enumerate(retrieval_list):
        query_img_name = img_pair[0]
        query_img_path = os.path.join(args.database_path, query_img_name)
        query_img = np.array(Image.open(query_img_path).convert("RGB"))

        initial_pose = pose_list[i]
        cam_pose = torch.clone(initial_pose.detach()).unsqueeze(0)
        cam_pose.requires_grad = True


        # create optimizer
        optimizer = torch.optim.Adam(params=[cam_pose], lr=args.lrate) # TODO:lrate needs to be set
        n_steps = args.N_steps + 1  #TODO: add args

        # Loss
        mse_loss = torch.nn.MSELoss()

        # Sampling
        n_rays = 1024
        sampling = 'random' # random/center/patch

        # Pose optimization
        predicted_poses = []
        fine_patches = []
        gt_patches = []

        for i_step in range(n_steps):
            # Short circuit if only rendering out from trained model
            # Change rendering only part to pose refine part
            with torch.no_grad():
                rgbs, disps = render_path(pose, hwf, args.chunk, render_kwargs_test, gt_imgs=images[i_test], savedir=testsavedir)


            loss = mse_loss(rgbs, query_img)

            optimizer.zero_grad()
            loss.backward()

            if i_step % 20 == 0:
                print(f"{i_step} step, Loss: {loss}.")
                print(cam_pose[0])
                predicted_poses.append(torch.clone(cam_pose[0]).detach().numpy())
                fine_patches.append(torch.clone(rgbs[0]).detach().cpu().numpy().reshape(32, 32, 3))
                gt_patches.append(torch.clone(query_img[idxs_sampled]).detach().cpu().numpy().reshape(32, 32, 3))
        
            optimizer.step()

        # rendering 


    # save

if __name__=='__main__':
    # torch.set_default_tensor_type('torch.cuda.FloatTensor')

    optimize()
