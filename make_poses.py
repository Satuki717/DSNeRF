import os
import numpy as np
import cv2
from PIL import Image

def cal_bds(posepath):
    depthpath = posepath.replace('pose.txt','depth.png')
    depth_img = cv2.imread(depthpath,-1).astype(float)
    depth_img /= 1000.  # depth is saved in 16-bit PNG in millimeters
    depth_img[depth_img == 65.535] = 0  # set invalid depth to 0 (specific to 7-scenes dataset)
    near = depth_img[depth_img > 0].min()
    far = depth_img.max()

    print(near,far)
    
    return near, far


def make_poses(scene, seq, posedir, outdir):
    imgdir = os.path.join(outdir, 'images')
    filename_list = sorted(os.listdir(imgdir))

    h, w, f = 480, 640, 585
    hwf = np.array([h,w,f]).reshape([3,1])

    pose_list = []
    bd_list = []
    for i in filename_list:
        posename = i.replace('color.png','pose.txt')
        posepath = os.path.join(posedir, scene, seq, posename)
        pose = np.loadtxt(posepath)
        pose_list.append(pose)
        
        near, far = cal_bds(posepath)
        bd_list.append([near, far])

        # argument_p = np.array(argument_p).reshape(-1)

        # save_arr.append(np.concatenate([argument_p,np.array([near, far])], 0))

    pose_list = np.stack(pose_list, 0)
    # print(pose_list.shape)
    # print(pose_list[0])
    poses = pose_list[:, :3, :4].transpose([1,2,0])
    # print(poses.shape)
    # print(poses[0])
    poses = np.concatenate([poses, np.tile(hwf[..., np.newaxis], [1,1,poses.shape[-1]])], 1)
    # print(poses.shape)
    # print(poses[0])

    # must switch to [-u, r, -t] from [r, -u, t], NOT [r, u, -t] to LLFF from opencv/colmap not NeRF/opengl
    poses = np.concatenate([poses[:, 1:2, :], poses[:, 0:1, :], -poses[:, 2:3, :], poses[:, 3:4, :], poses[:, 4:5, :]], 1)
    # print(poses.shape)
    # print(poses[0])

    save_arr = []
    for i in range(poses.shape[-1]):
        save_arr.append(np.concatenate([poses[..., i].ravel(), np.array(bd_list[i])], 0))

    save_arr = np.array(save_arr)
    np.save(os.path.join(outdir, 'poses_bounds_gt.npy'), save_arr)

def depthloss(rendered_depth, gt_depth):
    mask = gt_depth == 0
    error = (rendered_depth[mask] - gt_depth[mask]) ** 2
    return torch.mean(error)

def sample_depth(img, N_samples=500):
    ERR = 0.01

    H, W = img.shape
    sampled_depth = []
    index = []
    coord = []
    error = []
    while len(sampled_depth) < N_samples:
        idx = np.random.randint(0, H * W)
        row = idx // W
        col = idx % W
        depth = img[row, col]
        if depth==0 and idx in index:
            continue
        sampled_depth.append(depth)
        index.append(idx)
        coord.append([row, col])
        error.append(ERR)

    return sampled_depth, coord, error

def sample_all_depth(img):
    ERR = 0.01

    H, W = img.shape
    sampled_depth = []
    index = []
    coord = []
    error = []
    for i in range(H):
        for j in range(W):
            depth = img[i, j]
            if depth==0:
                continue
            sampled_depth.append(depth)
            index.append(H * i + j)
            coord.append([i, j])
            error.append(ERR)
    print(sampled_depth.shape)

    return sampled_depth, coord, error

def make_depths(scene, seq, imgdir, depthdir, outdir, factor=2, bd_factor=.75):
    filename_list = sorted(os.listdir(imgdir))

    _, bds_raw, _ = _load_data(basedir, factor=factor) # factor=8 downsamples original imgs by 8x
    bds_raw = np.moveaxis(bds_raw, -1, 0).astype(np.float32)
    # print(bds_raw.shape)
    # Rescale if bd_factor is provided
    sc = 1. if bd_factor is None else 1./(bds_raw.min() * bd_factor)

    depth_gts = []
    for i in filename_list:
        depthname = i.replace('color.png', 'depth.png')
        depthpath = os.path.join(depthdir, scene, seq, depthname)
        depth = Image.open(depthpath)
        depth = np.array(depth)
        depth[depth==65535] = 0
        depth = depth.astype(np.float32)*0.001
        # print(depth.shape)
        depth, coord, error = sample_all_depth(depth)
        depth_gts.append({"depth":np.array(depth * sc), "coord":np.array(coord / factor), "error":np.array(error)})

    np.save(os.path.join(outdir, 'gt_all_depth.npy'), depth_gts)

    return depth_gts

def make_render_pose(scene, seq, imgdir, posedir, outdir):

    filename_list = sorted(os.listdir(imgdir))

    render_poses = []
    for i in filename_list:
        posename = i.replace('color.png','pose.txt')
        posepath = os.path.join(posedir, scene, seq, posename)
        pose = np.asarray(np.loadtxt(posepath))
        h, w, f = 480, 640, 250
        # h, w, f = 480, 640, 585
        # print(pose.shape)
        argument_p = np.zeros((3,5))
        # poses_reset = np.concatenate([poses_reset[:,:3,:4], np.broadcast_to(poses[0,:3,-1:], poses_reset[:,:3,-1:].shape)], -1)
        argument_p[:3,:4] = pose[:3,:4]
        argument_p[0,4] = h
        argument_p[1,4] = w
        argument_p[2,4] = f
        # print(argument_p)
        render_poses.append(argument_p)
        savepath = os.path.join(outdir, 'render_pose.npy')
        np.save(savepath, render_poses)

    return render_poses

def sample_depth1(img, factor, N_samples=500):
    ERR = 0.01

    H, W = img.shape
    sampled_depth = []
    index = []
    coord = []
    error = []
    while len(sampled_depth) < N_samples:
        idx = np.random.randint(0, H * W)
        row = idx // W
        col = idx % W
        depth = img[row, col]
        if depth==0 or idx in index:
            continue
        sampled_depth.append(depth)
        index.append(idx)
        coord.append([row / factor, col / factor])
        error.append(ERR)

    return sampled_depth, coord, error

def make_depths1(scene, seq, imgdir, depthdir, basedir, factor=1):
    filename_list = sorted(os.listdir(imgdir))

    depth_gts = []
    for i in filename_list:
        depthname = i.replace('color.png', 'depth.png')
        depthpath = os.path.join(depthdir, scene, seq, depthname)
        depth = Image.open(depthpath)
        depth = np.array(depth)
        depth[depth==65535] = 0
        depth = depth.astype(np.float32)*0.001
        # print(depth.shape)
        depth, coord, error = sample_depth1(depth, factor)
        depth_gts.append({"depth":np.array(depth), "coord":np.array(coord), "error":np.array(error)})

    np.save(os.path.join(basedir, 'gt_depth.npy'), depth_gts)

    return depth_gts


if __name__=='__main__':
    make_poses('heads', 'seq-02','/mnt/datagrid1/yyuan/7scenes', 'data/heads3')