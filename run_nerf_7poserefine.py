import os, sys
import imageio.v2 as imageio
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
import pickle

import matplotlib.pyplot as plt

from run_nerf_helpers import *

# from data import RayDataset
# from torch.utils.data import DataLoader

import cv2

device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
np.random.seed(0)
DEBUG = False

def batchify(fn, chunk):
    """Constructs a version of 'fn' that applies to smaller batches.
    """
    if chunk is None:
        return fn
    def ret(inputs):
        return torch.cat([fn(inputs[i:i+chunk]) for i in range(0, inputs.shape[0], chunk)], 0)
    return ret

def run_network(inputs, viewdirs, fn, embed_fn, embeddirs_fn, netchunk=1024*64):
    """Prepares inputs and applies network 'fn'.
    """
    inputs_flat = torch.reshape(inputs, [-1, inputs.shape[-1]])
    embedded = embed_fn(inputs_flat)

    if viewdirs is not None:
        input_dirs = viewdirs[:,None].expand(inputs.shape)
        input_dirs_flat = torch.reshape(input_dirs, [-1, input_dirs.shape[-1]])
        embedded_dirs = embeddirs_fn(input_dirs_flat)
        embedded = torch.cat([embedded, embedded_dirs], -1)

    outputs_flat = batchify(fn, netchunk)(embedded)
    outputs = torch.reshape(outputs_flat, list(inputs.shape[:-1]) + [outputs_flat.shape[-1]])
    return outputs

def batchify_rays(rays_flat, chunk=1024*8, **kwargs):
    """Render rays in smaller minibatches to avoid OOM.
    """
    all_ret = {}
    # print(rays_flat.shape)
    for i in range(0, rays_flat.shape[0], chunk):
        ret = render_rays(rays_flat[i:i+chunk], **kwargs)
        for k in ret:
            if k not in all_ret:
                all_ret[k] = []
            all_ret[k].append(ret[k])

    all_ret = {k : torch.cat(all_ret[k], 0) for k in all_ret}
    return all_ret

def render(H, W, focal, chunk=1024*8, rays=None, c2w=None, ndc=True,
                  near=0., far=1.,
                  use_viewdirs=False, c2w_staticcam=None, depths=None,
                  **kwargs):
    """Render rays
    Args:
      H: int. Height of image in pixels.
      W: int. Width of image in pixels.
      focal: float. Focal length of pinhole camera.
      chunk: int. Maximum number of rays to process simultaneously. Used to
        control maximum memory usage. Does not affect final results.
      rays: array of shape [2, batch_size, 3]. Ray origin and direction for
        each example in batch.
      c2w: array of shape [3, 4]. Camera-to-world transformation matrix.
      ndc: bool. If True, represent ray origin, direction in NDC coordinates.
      near: float or array of shape [batch_size]. Nearest distance for a ray.
      far: float or array of shape [batch_size]. Farthest distance for a ray.
      use_viewdirs: bool. If True, use viewing direction of a point in space in model.
      c2w_staticcam: array of shape [3, 4]. If not None, use this transformation matrix for 
       camera while using other c2w argument for viewing directions.
    Returns:
      rgb_map: [batch_size, 3]. Predicted RGB values for rays.
      disp_map: [batch_size]. Disparity map. Inverse of depth.
      acc_map: [batch_size]. Accumulated opacity (alpha) along a ray.
      extras: dict with everything returned by render_rays().
    """
    if c2w is not None:
        # special case to render full image
        rays_o, rays_d = get_rays(H, W, focal, c2w)
    else:
        # use provided ray batch
        rays_o, rays_d = rays

    if use_viewdirs:
        # provide ray directions as input
        viewdirs = rays_d
        if c2w_staticcam is not None:
            # special case to visualize effect of viewdirs
            rays_o, rays_d = get_rays(H, W, focal, c2w_staticcam)
        viewdirs = viewdirs / torch.norm(viewdirs, dim=-1, keepdim=True)
        viewdirs = torch.reshape(viewdirs, [-1,3]).float()

    sh = rays_d.shape # [..., 3]
    if ndc:
        # for forward facing scenes
        rays_o, rays_d = ndc_rays(H, W, focal, 1., rays_o, rays_d)

    # Create ray batch
    rays_o = torch.reshape(rays_o, [-1,3]).float()
    rays_d = torch.reshape(rays_d, [-1,3]).float()

    near, far = near * torch.ones_like(rays_d[...,:1]), far * torch.ones_like(rays_d[...,:1])
    rays = torch.cat([rays_o, rays_d, near, far], -1) # B x 8
    if depths is not None:
        rays = torch.cat([rays, depths.reshape(-1,1)], -1)
    if use_viewdirs:
        rays = torch.cat([rays, viewdirs], -1)

    # Render and reshape
    all_ret = batchify_rays(rays, chunk, **kwargs)
    for k in all_ret:
        k_sh = list(sh[:-1]) + list(all_ret[k].shape[1:])
        all_ret[k] = torch.reshape(all_ret[k], k_sh)

    k_extract = ['rgb_map']
    ret_list = [all_ret[k] for k in k_extract]
    ret_dict = {k : all_ret[k] for k in all_ret if k not in k_extract}
    return ret_list + [ret_dict]

def render_path(render_poses, hwf, chunk, render_kwargs, gt_imgs=None, savedir=None, render_factor=0):

    H, W, focal = hwf

    if render_factor!=0:
        # Render downsampled for speed
        H = H//render_factor
        W = W//render_factor
        focal = focal/render_factor

    rgbs = []
    disps = []

    t = time.time()
    for i, c2w in enumerate(tqdm(render_poses)):
        print(i, time.time() - t)
        t = time.time()
        rgb, disp, acc, depth, extras = render(H, W, focal, chunk=chunk, c2w=c2w[:3,:4], retraw=True, **render_kwargs)
        rgbs.append(rgb.cpu().numpy())
        disps.append(disp.cpu().numpy())
        if i==0:
            print(rgb.shape, disp.shape)

        """
        if gt_imgs is not None and render_factor==0:
            p = -10. * np.log10(np.mean(np.square(rgb.cpu().numpy() - gt_imgs[i])))
            print(p)
        """

        if savedir is not None:
            rgb8 = to8b(rgbs[-1])
            rgb8[np.isnan(rgb8)] = 0
            filename = os.path.join(savedir, '{:03d}.png'.format(i))
            imageio.imwrite(filename, rgb8)
            depth = depth.cpu().numpy()
            maxdepth = np.nanmax(depth)
            print("max:", maxdepth)
            depth_color = depth.astype(np.float32) / np.max(depth) * 255
            depth_color = cv2.applyColorMap(depth_color.astype(np.uint8), cv2.COLORMAP_JET)[:,:,::-1]
            depth_color[np.isnan(depth_color)] = 0
            # depth = depth / 5 * 255
            # depth_color = cv2.applyColorMap(depth.astype(np.uint8), cv2.COLORMAP_JET)[:,:,::-1]
            # depth_color[np.isnan(depth_color)] = 0
            imageio.imwrite(os.path.join(savedir, '{:03d}_depth_normalized_{:.2f}.png'.format(i,maxdepth)), depth_color)
            # imageio.imwrite(os.path.join(savedir, '{:03d}_depth.png'.format(i)), depth.astype(np.float32))
            np.savez(os.path.join(savedir, '{:03d}.npz'.format(i)), rgb=rgb.cpu().numpy(), disp=disp.cpu().numpy(), acc=acc.cpu().numpy(), depth=depth)


    rgbs = np.stack(rgbs, 0)
    disps = np.stack(disps, 0)

    return rgbs, disps

def render_test_ray(rays_o, rays_d, hwf, ndc, near, far, use_viewdirs, N_samples, network, network_query_fn, **kwargs):
    H, W, focal = hwf
    if use_viewdirs:
        # provide ray directions as input
        viewdirs = rays_d
        viewdirs = viewdirs / torch.norm(viewdirs, dim=-1, keepdim=True)
        viewdirs = torch.reshape(viewdirs, [-1,3]).float()

    if ndc:
        # for forward facing scenes
        rays_o, rays_d = ndc_rays(H, W, focal, 1., rays_o, rays_d)

    # Create ray batch
    rays_o = torch.reshape(rays_o, [-1,3]).float()
    rays_d = torch.reshape(rays_d, [-1,3]).float()

    near, far = near * torch.ones_like(rays_d[...,:1]), far * torch.ones_like(rays_d[...,:1])

    t_vals = torch.linspace(0., 1., steps=N_samples).to(device)
    z_vals = near * (1.-t_vals) + far * (t_vals)

    z_vals = z_vals.reshape([rays_o.shape[0], N_samples])

    rgb, sigma, depth_maps, weights = sample_sigma(rays_o, rays_d, viewdirs, network, z_vals, network_query_fn)

    return rgb, sigma, z_vals, depth_maps, weights

def create_nerf(args):
    """Instantiate NeRF's MLP model.
    """
    embed_fn, input_ch = get_embedder(args.multires, args.i_embed)

    input_ch_views = 0
    embeddirs_fn = None
    if args.use_viewdirs:
        embeddirs_fn, input_ch_views = get_embedder(args.multires_views, args.i_embed)
    output_ch = 5 if args.N_importance > 0 else 4
    skips = [4]
    if args.alpha_model_path is None:
        model = NeRF(D=args.netdepth, W=args.netwidth,
                    input_ch=input_ch, output_ch=output_ch, skips=skips,
                    input_ch_views=input_ch_views, use_viewdirs=args.use_viewdirs).to(device)
        grad_vars = list(model.parameters())
    else:
        alpha_model = NeRF(D=args.netdepth_fine, W=args.netwidth_fine,
                            input_ch=input_ch, output_ch=output_ch, skips=skips,
                            input_ch_views=input_ch_views, use_viewdirs=args.use_viewdirs).to(device)
        print('Alpha model reloading from', args.alpha_model_path)
        ckpt = torch.load(args.alpha_model_path)
        alpha_model.load_state_dict(ckpt['network_fine_state_dict'])
        if not args.no_coarse:
            model = NeRF_RGB(D=args.netdepth, W=args.netwidth,
                        input_ch=input_ch, output_ch=output_ch, skips=skips,
                        input_ch_views=input_ch_views, use_viewdirs=args.use_viewdirs, alpha_model=alpha_model).to(device)
            grad_vars = list(model.parameters())
        else:
            model = None
            grad_vars = []
    

    model_fine = None
    if args.N_importance > 0:
        if args.alpha_model_path is None:
            model_fine = NeRF(D=args.netdepth_fine, W=args.netwidth_fine,
                            input_ch=input_ch, output_ch=output_ch, skips=skips,
                            input_ch_views=input_ch_views, use_viewdirs=args.use_viewdirs).to(device)
        else:
            model_fine = NeRF_RGB(D=args.netdepth_fine, W=args.netwidth_fine,
                            input_ch=input_ch, output_ch=output_ch, skips=skips,
                            input_ch_views=input_ch_views, use_viewdirs=args.use_viewdirs, alpha_model=alpha_model).to(device)
        grad_vars += list(model_fine.parameters())

    network_query_fn = lambda inputs, viewdirs, network_fn : run_network(inputs, viewdirs, network_fn,
                                                                embed_fn=embed_fn,
                                                                embeddirs_fn=embeddirs_fn,
                                                                netchunk=args.netchunk)

    # Create optimizer
    optimizer = torch.optim.Adam(params=grad_vars, lr=args.lrate, betas=(0.9, 0.999))

    start = 0
    basedir = args.basedir
    expname = args.expname

    ##########################

    # Load checkpoints
    if args.ft_path is not None and args.ft_path!='None':
        ckpts = [args.ft_path]
    else:
        ckpts = [os.path.join(basedir, expname, f) for f in sorted(os.listdir(os.path.join(basedir, expname))) if 'tar' in f]

    print('Found ckpts', ckpts)
    if len(ckpts) > 0 and not args.no_reload:
        ckpt_path = ckpts[-1]
        print('Reloading from', ckpt_path)
        ckpt = torch.load(ckpt_path)

        start = ckpt['global_step']
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])

        # Load model
        model.load_state_dict(ckpt['network_fn_state_dict'])
        if model_fine is not None:
            model_fine.load_state_dict(ckpt['network_fine_state_dict'])

    ##########################

    render_kwargs_train = {
        'network_query_fn' : network_query_fn,
        'perturb' : args.perturb,
        'N_importance' : args.N_importance,
        'network_fine' : model_fine,
        'N_samples' : args.N_samples,
        'network_fn' : model,
        'use_viewdirs' : args.use_viewdirs,
        'white_bkgd' : args.white_bkgd,
        'raw_noise_std' : args.raw_noise_std,
    }

    # NDC only good for LLFF-style forward facing data
    if args.dataset_type != 'llff' or args.no_ndc:
        print('Not ndc!')
        render_kwargs_train['ndc'] = False
        render_kwargs_train['lindisp'] = args.lindisp
    else:
        render_kwargs_train['ndc'] = True

    render_kwargs_test = {k : render_kwargs_train[k] for k in render_kwargs_train}
    render_kwargs_test['perturb'] = False
    render_kwargs_test['raw_noise_std'] = 0.

    # if args.sigma_loss:
    #     render_kwargs_train['sigma_loss'] = SigmaLoss(args.N_samples, args.perturb, args.raw_noise_std)

    ##########################


    return render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer

def render_rays(ray_batch,
                network_fn,
                network_query_fn,
                N_samples,
                retraw=False,
                lindisp=False,
                perturb=0.,
                N_importance=0,
                network_fine=None,
                white_bkgd=False,
                raw_noise_std=0.,
                verbose=False,
                pytest=False,
                sigma_loss=None):
    """Volumetric rendering.
    Args:
      ray_batch: array of shape [batch_size, ...]. All information necessary
        for sampling along a ray, including: ray origin, ray direction, min
        dist, max dist, and unit-magnitude viewing direction.
      network_fn: function. Model for predicting RGB and density at each point
        in space.
      network_query_fn: function used for passing queries to network_fn.
      N_samples: int. Number of different times to sample along each ray.
      retraw: bool. If True, include model's raw, unprocessed predictions.
      lindisp: bool. If True, sample linearly in inverse depth rather than in depth.
      perturb: float, 0 or 1. If non-zero, each ray is sampled at stratified
        random points in time.
      N_importance: int. Number of additional times to sample along each ray.
        These samples are only passed to network_fine.
      network_fine: "fine" network with same spec as network_fn.
      white_bkgd: bool. If True, assume a white background.
      raw_noise_std: ...
      verbose: bool. If True, print more debugging info.
    Returns:
      rgb_map: [num_rays, 3]. Estimated RGB color of a ray. Comes from fine model.
      disp_map: [num_rays]. Disparity map. 1 / depth.
      acc_map: [num_rays]. Accumulated opacity along each ray. Comes from fine model.
      raw: [num_rays, num_samples, 4]. Raw predictions from model.
      rgb0: See rgb_map. Output for coarse model.
      disp0: See disp_map. Output for coarse model.
      acc0: See acc_map. Output for coarse model.
      z_std: [num_rays]. Standard deviation of distances along ray for each
        sample.
    """
    N_rays = ray_batch.shape[0]
    rays_o, rays_d = ray_batch[:,0:3], ray_batch[:,3:6] # [N_rays, 3] each
    viewdirs = ray_batch[:,-3:] if ray_batch.shape[-1] > 9 else None
    bounds = torch.reshape(ray_batch[...,6:8], [-1,1,2])
    near, far = bounds[...,0], bounds[...,1] # [-1,1]

    t_vals = torch.linspace(0., 1., steps=N_samples).to(device)
    if not lindisp:
        z_vals = near * (1.-t_vals) + far * (t_vals)
    else:
        z_vals = 1./(1./near * (1.-t_vals) + 1./far * (t_vals))

    z_vals = z_vals.expand([N_rays, N_samples])

    if perturb > 0.:
        # get intervals between samples
        mids = .5 * (z_vals[...,1:] + z_vals[...,:-1])
        upper = torch.cat([mids, z_vals[...,-1:]], -1)
        lower = torch.cat([z_vals[...,:1], mids], -1)
        # stratified samples in those intervals
        t_rand = torch.rand(z_vals.shape).to(device)

        # Pytest, overwrite u with numpy's fixed random numbers
        if pytest:
            np.random.seed(0)
            t_rand = np.random.rand(*list(z_vals.shape))
            t_rand = torch.Tensor(t_rand).to(device)

        z_vals = lower + (upper - lower) * t_rand

    pts = rays_o[...,None,:] + rays_d[...,None,:] * z_vals[...,:,None] # [N_rays, N_samples, 3]


#     raw = run_network(pts)
    if network_fn is not None:
        # time1 = time.time()
        raw = network_query_fn(pts, viewdirs, network_fn)
        # time2 = time.time()
        # print('model time:{}s'.format(time2-time1))
        rgb_map, disp_map, acc_map, weights, depth_map = raw2outputs1(raw, z_vals, rays_d, raw_noise_std, white_bkgd, pytest=pytest)
        # time3 = time.time()
        # print('render time:{}s'.format(time3-time2))
    else:
        # rgb_map, disp_map, acc_map = None, None, None
        # raw2alpha = lambda raw, dists, act_fn=F.relu: 1.-torch.exp(-act_fn(raw)*dists)
        # noise = 0
        # alpha = network_query_fn(pts, viewdirs, network_fine.alpha_model)[...,3]
        if network_fine.alpha_model is not None:
            raw = network_query_fn(pts, viewdirs, network_fine.alpha_model)
            rgb_map, disp_map, acc_map, weights, depth_map = raw2outputs1(raw, z_vals, rays_d, raw_noise_std, white_bkgd, pytest=pytest)
        else:
            raw = network_query_fn(pts, viewdirs, network_fine)
            rgb_map, disp_map, acc_map, weights, depth_map = raw2outputs1(raw, z_vals, rays_d, raw_noise_std, white_bkgd, pytest=pytest)


    if N_importance > 0:

        rgb_map_0, disp_map_0, acc_map_0 = rgb_map, disp_map, acc_map

        z_vals_mid = .5 * (z_vals[...,1:] + z_vals[...,:-1])
        z_samples = sample_pdf(z_vals_mid, weights[...,1:-1], N_importance, det=(perturb==0.), pytest=pytest)
        z_samples = z_samples.detach()

        z_vals, _ = torch.sort(torch.cat([z_vals, z_samples], -1), -1)
        pts = rays_o[...,None,:] + rays_d[...,None,:] * z_vals[...,:,None] # [N_rays, N_samples + N_importance, 3]

        run_fn = network_fn if network_fine is None else network_fine
#         raw = run_network(pts, fn=run_fn)
        raw = network_query_fn(pts, viewdirs, run_fn)

        rgb_map, disp_map, acc_map, weights, depth_map = raw2outputs1(raw, z_vals, rays_d, raw_noise_std, white_bkgd, pytest=pytest)

    ret = {'rgb_map' : rgb_map}
    if retraw:
        ret['raw'] = raw
    if N_importance > 0:
        ret['rgb0'] = rgb_map_0

    # for k in ret:
    #     if (torch.isnan(ret[k]).any() or torch.isinf(ret[k]).any()) and DEBUG:
    #         print(f"! [Numerical Error] {k} contains nan or inf.")

    return ret


# define functions
def load_initial_pose(retrieval_list_path, database_dir, scene='heads'):
    comment_list = ['chess', 'fire', 'heads', 'office', 'pumpkin', 'redkitchen', 'stairs']
    comment_list.remove(scene)
    retrieval_list = np.loadtxt(retrieval_list_path, dtype=str, comments=comment_list, usecols=(0,1))
    # print(retrieval_list[:5]) # N * 2

    pred_pose_path = retrieval_list_path.replace('retrieval_list.txt', 'query_pose_prediction_dict.pkl')
    with open(pred_pose_path, 'rb') as file:
        pred_pose_dict = pickle.load(file)
 
    coarse_pose_list = []
    for i in range(len(retrieval_list)):
        pose_path = os.path.join(database_dir,retrieval_list[i,1]).replace('color.png','pose.txt')
        pose = np.asarray(np.loadtxt(pose_path))
        coarse_pose_list.append(pose)
    coarse_pose_list = np.stack(coarse_pose_list, 0) # N * 4 * 4

    gt_pose_list = []
    for i in range(len(retrieval_list)):
        pose_path = os.path.join(database_dir,retrieval_list[i,0]).replace('color.png','pose.txt')
        pose = np.asarray(np.loadtxt(pose_path))
        gt_pose_list.append(pose)
    gt_pose_list = np.stack(gt_pose_list, 0) # N * 4 * 4

    return retrieval_list, coarse_pose_list, gt_pose_list, pred_pose_dict

def calRelativePose(poseMatrix1, poseMatrix2):
    """Calculate the relative pose FROM matrix1 To matrix2 using in 7scenes
    # Return Value: relativeMatrix
    """
    assert (poseMatrix1.shape == (4, 4)), "shape of matrix 1 should be (4,4)"
    assert (poseMatrix2.shape == (4, 4)), "shape of matrix 1 should be (4,4)"

    rotMatrix1 = poseMatrix1[0:3, 0:3]
    rotMatrix2 = poseMatrix2[0:3, 0:3]
    transVec1 = poseMatrix1[0:3, 3]
    transVec2 = poseMatrix2[0:3, 3]

    poseMatrix2_inv = np.zeros((4, 4), dtype=np.float64)
    # [R|t]' = [R^T|-R^Tt]
    poseMatrix2_inv[0:3, 0:3] = rotMatrix2.T
    poseMatrix2_inv[0:3, 3]  = - np.matmul(rotMatrix2.T, transVec2)
    poseMatrix2_inv[3, 3]   = 1

    relativeMatrix = np.matmul(poseMatrix2_inv, poseMatrix1)
    return relativeMatrix

def calPoseError(gtRelativePose, predictionPose):
    """Calculate the error between ground-truth pose and prediction pose
    # Return Value: transError, rotError 
    """
    assert (gtRelativePose.shape == (4, 4)), "shape of ground truth pose matrix should be (4,4)"
    assert (predictionPose.shape == (4, 4)), "shape of prediction pose matrix should be (4,4)"
    
    gt_2_pred_pose = calRelativePose(gtRelativePose, predictionPose)

    rotError = np.linalg.norm(cv2.Rodrigues(gt_2_pred_pose[0:3, 0:3])[0])# / np.linalg.norm(cv2.Rodrigues(gtRelativePose[0:3, 0:3])[0])
    transError = np.linalg.norm(gt_2_pred_pose[0:3, 3])# / np.linalg.norm(gtRelativePose[0:3, 3])
    return rotError, transError

def num_count(arr,rot=5,trans=0.05):

    cond = (arr[0,:] < rot) & (arr[1,:] < trans)
    count_in_num = np.count_nonzero(cond)
    print(count_in_num, '/', arr.shape[1])

    return count_in_num / arr.shape[0]

def config_parser():

    import configargparse
    parser = configargparse.ArgumentParser()
    parser.add_argument('--config', is_config_file=True, 
                        help='config file path')
    parser.add_argument("--expname", type=str, 
                        help='experiment name')
    parser.add_argument("--basedir", type=str, default='./logs/', 
                        help='where to store ckpts and logs')
    parser.add_argument("--datadir", type=str, default='./data/llff/fern', 
                        help='input data directory')

    # training options
    parser.add_argument("--netdepth", type=int, default=8, 
                        help='layers in network')
    parser.add_argument("--netwidth", type=int, default=256, 
                        help='channels per layer')
    parser.add_argument("--netdepth_fine", type=int, default=8, 
                        help='layers in fine network')
    parser.add_argument("--netwidth_fine", type=int, default=256, 
                        help='channels per layer in fine network')
    parser.add_argument("--N_rand", type=int, default=32*32*1, 
                        help='batch size (number of random rays per gradient step)')
    parser.add_argument("--lrate", type=float, default=5e-4, 
                        help='learning rate')
    parser.add_argument("--lrate_decay", type=int, default=250, 
                        help='exponential learning rate decay (in 1000 steps)')
    parser.add_argument("--chunk", type=int, default=1024*1, 
                        help='number of rays processed in parallel, decrease if running out of memory')
    parser.add_argument("--netchunk", type=int, default=1024*8, 
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
    parser.add_argument("--perturb", type=float, default=1.,
                        help='set to 0. for no jitter, 1. for jitter')
    parser.add_argument("--use_viewdirs", action='store_true', 
                        help='use full 5D input instead of 3D')
    parser.add_argument("--i_embed", type=int, default=0, 
                        help='set 0 for default positional encoding, -1 for none')
    parser.add_argument("--multires", type=int, default=10, 
                        help='log2 of max freq for positional encoding (3D location)')
    parser.add_argument("--multires_views", type=int, default=4, 
                        help='log2 of max freq for positional encoding (2D direction)')
    parser.add_argument("--raw_noise_std", type=float, default=0., 
                        help='std dev of noise added to regularize sigma_a output, 1e0 recommended')

    parser.add_argument("--render_only", action='store_true', 
                        help='do not optimize, reload weights and render out render_poses path')
    parser.add_argument("--render_test", action='store_true', 
                        help='render the test set instead of render_poses path')
    parser.add_argument("--render_test_ray", action='store_true', 
                        help='render the test set instead of render_poses path')
    parser.add_argument("--render_train", action='store_true', 
                        help='render the train set instead of render_poses path')  
    parser.add_argument("--render_mypath", action='store_true', 
                        help='render the test path')         
    parser.add_argument("--render_factor", type=int, default=0, 
                        help='downsampling factor to speed up rendering, set 4 or 8 for fast preview')

    # training options
    parser.add_argument("--precrop_iters", type=int, default=0,
                        help='number of steps to train on central crops')
    parser.add_argument("--precrop_frac", type=float,
                        default=.5, help='fraction of img taken for central crops') 

    # dataset options
    parser.add_argument("--dataset_type", type=str, default='llff', 
                        help='options: llff / blender / deepvoxels')
    # parser.add_argument("--testskip", type=int, default=8, 
    #                     help='will load 1/N images from test/val sets, useful for large datasets like deepvoxels')

    ## deepvoxels flags
    parser.add_argument("--shape", type=str, default='greek', 
                        help='options : armchair / cube / greek / vase')

    ## blender flags
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
    parser.add_argument("--i_print",   type=int, default=100, 
                        help='frequency of console printout and metric loggin')
    parser.add_argument("--i_img",     type=int, default=500, 
                        help='frequency of tensorboard image logging')
    parser.add_argument("--i_weights", type=int, default=10000, 
                        help='frequency of weight ckpt saving')
    parser.add_argument("--i_testset", type=int, default=50000, 
                        help='frequency of testset saving')
    parser.add_argument("--i_video",   type=int, default=50000, 
                        help='frequency of render_poses video saving')
    
    # debug
    parser.add_argument("--debug",  action='store_true')

    # new experiment by kangle
    parser.add_argument("--N_iters", type=int, default=200000, 
                        help='number of iters')
    parser.add_argument("--alpha_model_path", type=str, default=None,
                        help='predefined alpha model')
    parser.add_argument("--no_coarse", action='store_true',
                        help="Remove coarse network.")
    parser.add_argument("--train_scene", nargs='+', type=int,
                        help='id of scenes used to train')
    parser.add_argument("--test_scene", nargs='+', type=int,
                        help='id of scenes used to test')
    parser.add_argument("--colmap_depth", action='store_true',
                        help="Use depth supervision by colmap.")
    parser.add_argument("--gt_depth", action='store_true',
                        help="Use depth supervision by groundtruth.")
    parser.add_argument("--depth_loss", action='store_true',
                        help="Use depth supervision by colmap - depth loss.")
    parser.add_argument("--depth_lambda", type=float, default=0.1,
                        help="Depth lambda used for loss.")
    parser.add_argument("--sigma_loss", action='store_true',
                        help="Use depth supervision by colmap - sigma loss.")
    parser.add_argument("--sigma_lambda", type=float, default=0.1,
                        help="Sigma lambda used for loss.")
    parser.add_argument("--weighted_loss", action='store_true',
                        help="Use weighted loss by reprojection error.")
    parser.add_argument("--relative_loss", action='store_true',
                        help="Use relative loss.")
    parser.add_argument("--depth_with_rgb", action='store_true',
                    help="single forward for both depth and rgb")
    parser.add_argument("--normalize_depth", action='store_true',
                    help="normalize depth before calculating loss")
    parser.add_argument("--depth_rays_prop", type=float, default=0.5,
                        help="Proportion of depth rays.")

    # new settings for 7scenes
    parser.add_argument("--loaddepth", action='store_true',
                        help="decide whether load depth from file or collect directly")
    parser.add_argument("--h", type=int, default=480, 
                        help="set image height in load 7scenes")
    parser.add_argument("--w", type=int, default=640, 
                        help="set image width in load 7scenes")
    parser.add_argument("--f", type=float, default=585., 
                        help="set focal length in load 7scenes")
    parser.add_argument("--dbdir", type=str, default='/mnt/datagrid1/yyuan/7scenes', 
                        help="set dataset path")
    parser.add_argument("--scene", type=str, default='heads', 
                        help="set scene to train or test")
    parser.add_argument("--trainskip", type=int, default=5, 
                        help="set the image sampling interval for training")
    parser.add_argument("--testskip", type=int, default=10, 
                        help="set the image sampling interval for testing")
    
    parser.add_argument("--render_refine", action='store_true', 
                        help='render to optimize poses')
    parser.add_argument("--retrieve_list_path", type=str, 
                        default='/mnt/datagrid1/yyuan/dockersw/code/Camera_localization_diff_render_refine/inference_coarse_results/retrieval_coarse_estimation/retrieval_list.txt', 
                        help='input initial coarse pose file directory')
     
    parser.add_argument("--use_coarse_estimation", action='store_true', 
                        help='set whether use coarse pose estimation result or database pose result')
    parser.add_argument("--refinelrate", type=float, default=2e-2)
    parser.add_argument("--refine_steps", type=int, default=200, help="Number of steps for pose optimization.")
    parser.add_argument("--refine_i_print",   type=int, default=10, 
                        help='frequency of console printout and metric loggin')
    parser.add_argument("--refine_i_test",   type=int, default=20, 
                        help='frequency of console printout and metric loggin')
    
    return parser


def pose_refine():

    parser = config_parser()
    args = parser.parse_args()

    if args.dataset_type == 'colmap_llff':
        pass
    elif args.dataset_type == 'llff':
        pass
    elif args.dataset_type == '7Scenes':
        pass
    # load pose to be refined
    elif args.dataset_type == '7ScenesRefine':
        retrieval_list, coarse_pose_list, gt_pose_list, pred_pose_dict = load_initial_pose(args.retrieve_list_path, args.dbdir, args.scene)
        print('list length:', len(retrieval_list))

        # opencv coordinate system -> NeRF coordinate system 
        coarse_pose_list = np.concatenate([coarse_pose_list[:, :, 0:1], -coarse_pose_list[:, :, 1:2], -coarse_pose_list[:, :, 2:3], coarse_pose_list[:, :, 3:4]], 2)
        gt_pose_list = np.concatenate([gt_pose_list[:, :, 0:1], -gt_pose_list[:, :, 1:2], -gt_pose_list[:, :, 2:3], gt_pose_list[:, :, 3:4]], 2)
        
        near = 0.4
        far = 2.5
    else:
        print('Unknown dataset type', args.dataset_type, 'exiting')
        return

    # Cast intrinsics to right types
    H = args.h
    W = args.w
    focal = args.f
    hwf = [H, W, focal]

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

    # Create nerf model
    render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer = create_nerf(args)

    global_step = start

    bds_dict = {
        'near' : near,
        'far' : far,
    }
    render_kwargs_train.update(bds_dict)
    render_kwargs_test.update(bds_dict)

    # pose refine

    print('------Start pose refine------')
    refine_result = []

    # for i in range(len(retrieval_list)):
    for i in tqdm(range(len(retrieval_list)), desc="processing"):
        query_img_path = os.path.join(args.dbdir, retrieval_list[i,0])
        query_img = imageio.imread(query_img_path)/255.0
        query_img = query_img.astype(np.float32)

        if args.use_coarse_estimation == True: # use the coarse pose estimation result from RelocNet
            scene, seq, index = retrieval_list[i,0].split('/')
            initial_pose = pred_pose_dict[scene][seq][index]
        else:   # use the database pose of the nearest image
            initial_pose = coarse_pose_list[i]
        gt_pose = gt_pose_list[i]

        print('currently refining image:', retrieval_list[i,0])
        print('ground truth pose:')
        print(gt_pose)
        print('initial pose from {}:'.format(retrieval_list[i,1]))
        print(initial_pose)

        # Move testing data to GPU
        query_img = torch.Tensor(query_img).to(device)
        cam_pose = torch.Tensor(initial_pose).to(device)
        # cam_pose = torch.clone(initial_pose.detach())
        cam_pose.requires_grad = True
        
        # Create optimizer
        pose_optimizer = torch.optim.Adam(params=[cam_pose], lr=args.refinelrate)
        n_steps = args.refine_steps + 1

        # Samlping
        N_rgb = args.N_rand
        sampling = 'random'

        # Pose optimization
        predicted_poses = []
        error_list = []
        i_print_time = 0.0
        testsavedir = os.path.join(basedir, expname, 'poserefine', 'refine_{:06d}'.format(i))
        os.makedirs(testsavedir, exist_ok=True)

        for i_step in range(n_steps):
            time0 = time.time()

            rays_o, rays_d = get_rays(H, W, focal, cam_pose)  # (H, W, 3), (H, W, 3)
            
            coords = torch.stack(torch.meshgrid(torch.linspace(0, H-1, H), torch.linspace(0, W-1, W)), -1)  # (H, W, 2)

            coords = torch.reshape(coords, [-1,2])  # (H * W, 2)
            select_inds = np.random.choice(coords.shape[0], size=[N_rgb], replace=False)  # (N_rand,)
            select_coords = coords[select_inds].long()  # (N_rand, 2)
            rays_o = rays_o[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
            rays_d = rays_d[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
            batch_rays = torch.stack([rays_o, rays_d], 0) # (2, N_rand, 3)
            target_s = query_img[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)

            #####  Core optimization loop  #####
            rgb, extras = render(H, W, focal, chunk=args.chunk, rays=batch_rays,
                                                    verbose=i < 10, retraw=True,
                                                    **render_kwargs_test)

            img_loss = img2mse(rgb, target_s)
            pose_optimizer.zero_grad()
            img_loss.backward()
            pose_optimizer.step()

            dt = time.time()-time0

            # Record data and rendering
            predicted_poses.append(torch.clone(cam_pose).detach().cpu().numpy())

            pred_pose = predicted_poses[-1].copy()
            rot_error, trans_error = calPoseError(gt_pose, pred_pose)
            error_list.append([i_step, img_loss.item(), rot_error*180/np.pi, trans_error])

            i_print_time += dt
            
            if i_step%args.refine_i_print==0:
                print('Step {}, loss:{:6f}, rot_error: {:6f}°, trans_error: {:6f}m time:{:6f}s'.format(i_step, img_loss.item(), rot_error*180/np.pi, trans_error, i_print_time))
                i_print_time = 0.0
            
            if i_step%args.refine_i_test==0:
                print('start render image from refined pose')
                print('pred_cam_pose:')
                print(pred_pose)
                c2w = torch.Tensor(pred_pose).to(device)
                with torch.no_grad():
                    rgb, extras = render(H, W, focal, chunk=args.chunk, c2w=c2w[:3,:4], retraw=True, **render_kwargs_test)
                    rgb8 = to8b(rgb.cpu().numpy())
                    rgb8[np.isnan(rgb8)] = 0
                    filename = os.path.join(testsavedir, '{:03d}.png'.format(i_step))
                    imageio.imwrite(filename, rgb8)
                print('image render complete')
        
        # draw the loss and error curves
        x_axis = [row[0] for row in error_list]
        loss = [row[1] for row in error_list]
        rot_err = [row[2] for row in error_list]
        trans_err = [row[3] for row in error_list]

        plt.figure(dpi=100)
        plt.plot(x_axis, loss, label='loss')
        plt.xlabel('iteration')
        plt.ylabel('loss')
        plt.title('loss of pose refinement for {}'.format(retrieval_list[i,0]))
        plt.legend()
        plt.grid()
        plt.savefig(os.path.join(testsavedir, 'image_loss.png'))
        plt.close()

        plt.figure(dpi=100)
        plt.plot(x_axis, rot_err, label='rot error')
        plt.xlabel('iteration')
        plt.ylabel('rot error (°)')
        plt.title('rotation error of pose refinement for {}'.format(retrieval_list[i,0]))
        plt.legend()
        plt.grid()
        plt.savefig(os.path.join(testsavedir, 'rot_error.png'))
        plt.close()
        
        plt.figure(dpi=100)
        plt.plot(x_axis, trans_err, label='trans error')
        plt.xlabel('iteration')
        plt.ylabel('trans error (m)')
        plt.title('translation error of pose refinement for {}'.format(retrieval_list[i,0]))
        plt.legend()
        plt.grid()
        plt.savefig(os.path.join(testsavedir, 'trans_error.png'))
        plt.close()

        # save gif
        
        # save refine result for each image
        np.save(os.path.join(testsavedir, 'predicted_poses_NeRF.npy'), np.array(predicted_poses))
        np.save(os.path.join(testsavedir, 'refine_loss.npy'), np.array(error_list))

        # record final refine result
        refine_result.append([retrieval_list[i,0], error_list[-1][1], error_list[-1][2], error_list[-1][3]])

    # calculate the mid error of the pose refinement for the whole scene
    scene_rot_error = np.array([row[2] for row in refine_result])
    scene_trans_error = np.array([row[3] for row in refine_result])
    scene_rot_median = np.median(scene_rot_error)
    scene_trans_median = np.median(scene_trans_error)
    array = np.vstack((scene_rot_error, scene_trans_error))
    CountInPercentage = num_count(array)
    print("median absolute error of rot = {}°, on {}".format(scene_rot_median, args.scene))
    print("median absolute error of trans = {}m, on {}".format(scene_trans_median, args.scene))
    print("Percentage of results within 5° and 0.05m = {:.2f}% on {}".format(CountInPercentage, args.scene))

    # save refine result for whole scene
    np.savetxt(os.path.join(basedir, expname, 'refine_result.txt'), refine_result, fmt='%s')

if __name__=='__main__':

    pose_refine()
