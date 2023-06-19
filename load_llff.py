from copy import error
import numpy as np
import os, imageio
from pathlib import Path
from colmapUtils.read_write_model import *
from colmapUtils.read_write_dense import *
import json
from PIL import Image

########## Slightly modified version of LLFF data loading code 
##########  see https://github.com/Fyusion/LLFF for original

def _minify(basedir, factors=[], resolutions=[]):
    needtoload = False
    for r in factors:
        imgdir = os.path.join(basedir, 'images_{}'.format(r))
        if not os.path.exists(imgdir):
            needtoload = True
    for r in resolutions:
        imgdir = os.path.join(basedir, 'images_{}x{}'.format(r[1], r[0]))
        if not os.path.exists(imgdir):
            needtoload = True
    if not needtoload:
        return
    
    from shutil import copy
    from subprocess import check_output
    
    imgdir = os.path.join(basedir, 'images')
    imgs = [os.path.join(imgdir, f) for f in sorted(os.listdir(imgdir))]
    imgs = [f for f in imgs if any([f.endswith(ex) for ex in ['JPG', 'jpg', 'png', 'jpeg', 'PNG']])]
    imgdir_orig = imgdir
    
    wd = os.getcwd()

    for r in factors + resolutions:
        if isinstance(r, int):
            name = 'images_{}'.format(r)
            resizearg = '{}%'.format(100./r)
        else:
            name = 'images_{}x{}'.format(r[1], r[0])
            resizearg = '{}x{}'.format(r[1], r[0])
        imgdir = os.path.join(basedir, name)
        if os.path.exists(imgdir):
            continue
            
        print('Minifying', r, basedir)
        
        os.makedirs(imgdir)
        check_output('cp {}/* {}'.format(imgdir_orig, imgdir), shell=True)
        
        ext = imgs[0].split('.')[-1]
        args = ' '.join(['mogrify', '-resize', resizearg, '-format', 'png', '*.{}'.format(ext)])
        print(args)
        os.chdir(imgdir)
        check_output(args, shell=True)
        os.chdir(wd)
        
        if ext != 'png':
            check_output('rm {}/*.{}'.format(imgdir, ext), shell=True)
            print('Removed duplicates')
        print('Done')
            
        
        
        
def _load_data(basedir, factor=None, width=None, height=None, load_imgs=True):
    
    poses_arr = np.load(os.path.join(basedir, 'poses_bounds_gt.npy'))
    poses = poses_arr[:, :-2].reshape([-1, 3, 5]).transpose([1,2,0]) # 3 x 5 x N
    bds = poses_arr[:, -2:].transpose([1,0])
    
    img0 = [os.path.join(basedir, 'images', f) for f in sorted(os.listdir(os.path.join(basedir, 'images'))) \
            if f.endswith('JPG') or f.endswith('jpg') or f.endswith('png')][0]
    sh = imageio.imread(img0).shape
    
    sfx = ''
    
    if factor is not None:
        sfx = '_{}'.format(factor)
        _minify(basedir, factors=[factor])
        factor = factor
    elif height is not None:
        factor = sh[0] / float(height)
        width = int(sh[1] / factor)
        _minify(basedir, resolutions=[[height, width]])
        sfx = '_{}x{}'.format(width, height)
    elif width is not None:
        factor = sh[1] / float(width)
        height = int(sh[0] / factor)
        _minify(basedir, resolutions=[[height, width]])
        sfx = '_{}x{}'.format(width, height)
    else:
        factor = 1
    
    imgdir = os.path.join(basedir, 'images' + sfx)
    if not os.path.exists(imgdir):
        print( imgdir, 'does not exist, returning' )
        return
    
    imgfiles = [os.path.join(imgdir, f) for f in sorted(os.listdir(imgdir)) if f.endswith('JPG') or f.endswith('jpg') or f.endswith('png')]
    if poses.shape[-1] != len(imgfiles):
        print( 'Mismatch between imgs {} and poses {} !!!!'.format(len(imgfiles), poses.shape[-1]) )
        return
    
    sh = imageio.imread(imgfiles[0]).shape
    poses[:2, 4, :] = np.array(sh[:2]).reshape([2, 1])
    poses[2, 4, :] = poses[2, 4, :] * 1./factor
    
    if not load_imgs:
        return poses, bds
    
    def imread(f):
        if f.endswith('png'):
            # return imageio.imread(f, ignoregamma=True)
            return imageio.imread(f)
        else:
            return imageio.imread(f)
        
    imgs = imgs = [imread(f)[...,:3]/255. for f in imgfiles]
    imgs = np.stack(imgs, -1)  
    
    print('Loaded image data', imgs.shape, poses[:,-1,0])
    return poses, bds, imgs

    
            
            
    

def normalize(x):
    return x / np.linalg.norm(x)

def viewmatrix(z, up, pos):
    vec2 = normalize(z)
    vec1_avg = up
    vec0 = normalize(np.cross(vec1_avg, vec2))
    vec1 = normalize(np.cross(vec2, vec0))
    m = np.stack([vec0, vec1, vec2, pos], 1)
    return m

def ptstocam(pts, c2w):
    tt = np.matmul(c2w[:3,:3].T, (pts-c2w[:3,3])[...,np.newaxis])[...,0]
    return tt

def poses_avg(poses):

    hwf = poses[0, :3, -1:]

    center = poses[:, :3, 3].mean(0)
    vec2 = normalize(poses[:, :3, 2].sum(0))
    up = poses[:, :3, 1].sum(0)
    c2w = np.concatenate([viewmatrix(vec2, up, center), hwf], 1)
    
    return c2w



def render_path_spiral(c2w, up, rads, focal, zdelta, zrate, rots, N):
    render_poses = []
    rads = np.array(list(rads) + [1.])
    hwf = c2w[:,4:5]
    
    for theta in np.linspace(0., 2. * np.pi * rots, N+1)[:-1]:
        c = np.dot(c2w[:3,:4], np.array([np.cos(theta), -np.sin(theta), -np.sin(theta*zrate), 1.]) * rads) 
        z = normalize(c - np.dot(c2w[:3,:4], np.array([0,0,-focal, 1.])))
        render_poses.append(np.concatenate([viewmatrix(z, up, c), hwf], 1))
    return render_poses
    


def recenter_poses(poses):

    poses_ = poses+0
    bottom = np.reshape([0,0,0,1.], [1,4])
    c2w = poses_avg(poses)
    c2w = np.concatenate([c2w[:3,:4], bottom], -2)
    bottom = np.tile(np.reshape(bottom, [1,1,4]), [poses.shape[0],1,1])
    poses = np.concatenate([poses[:,:3,:4], bottom], -2)

    poses = np.linalg.inv(c2w) @ poses
    poses_[:,:3,:4] = poses[:,:3,:4]
    poses = poses_
    return poses


#####################


def spherify_poses(poses, bds):
    
    p34_to_44 = lambda p : np.concatenate([p, np.tile(np.reshape(np.eye(4)[-1,:], [1,1,4]), [p.shape[0], 1,1])], 1)
    
    rays_d = poses[:,:3,2:3]
    rays_o = poses[:,:3,3:4]

    def min_line_dist(rays_o, rays_d):
        A_i = np.eye(3) - rays_d * np.transpose(rays_d, [0,2,1])
        b_i = -A_i @ rays_o
        pt_mindist = np.squeeze(-np.linalg.inv((np.transpose(A_i, [0,2,1]) @ A_i).mean(0)) @ (b_i).mean(0))
        return pt_mindist

    pt_mindist = min_line_dist(rays_o, rays_d)
    
    center = pt_mindist
    up = (poses[:,:3,3] - center).mean(0)

    vec0 = normalize(up)
    vec1 = normalize(np.cross([.1,.2,.3], vec0))
    vec2 = normalize(np.cross(vec0, vec1))
    pos = center
    c2w = np.stack([vec1, vec2, vec0, pos], 1)

    poses_reset = np.linalg.inv(p34_to_44(c2w[None])) @ p34_to_44(poses[:,:3,:4])

    rad = np.sqrt(np.mean(np.sum(np.square(poses_reset[:,:3,3]), -1)))
    
    sc = 1./rad
    poses_reset[:,:3,3] *= sc
    bds *= sc
    rad *= sc
    
    centroid = np.mean(poses_reset[:,:3,3], 0)
    zh = centroid[2]
    radcircle = np.sqrt(rad**2-zh**2)
    new_poses = []
    
    for th in np.linspace(0.,2.*np.pi, 120):

        camorigin = np.array([radcircle * np.cos(th), radcircle * np.sin(th), zh])
        up = np.array([0,0,-1.])

        vec2 = normalize(camorigin)
        vec0 = normalize(np.cross(vec2, up))
        vec1 = normalize(np.cross(vec2, vec0))
        pos = camorigin
        p = np.stack([vec0, vec1, vec2, pos], 1)

        new_poses.append(p)

    new_poses = np.stack(new_poses, 0)
    
    new_poses = np.concatenate([new_poses, np.broadcast_to(poses[0,:3,-1:], new_poses[:,:3,-1:].shape)], -1)
    poses_reset = np.concatenate([poses_reset[:,:3,:4], np.broadcast_to(poses[0,:3,-1:], poses_reset[:,:3,-1:].shape)], -1)
    
    return poses_reset, new_poses, bds
    

def load_llff_data(basedir, factor=8, recenter=True, bd_factor=.75, spherify=False, path_zflat=False):
    

    poses, bds, imgs = _load_data(basedir, factor=factor) # factor=8 downsamples original imgs by 8x
    print('Loaded', basedir, bds.min(), bds.max())
    
    # print('poses_bound.npy:\n', poses[:,:,0])

    # Correct rotation matrix ordering and move variable dim to axis 0
    poses = np.concatenate([poses[:, 1:2, :], -poses[:, 0:1, :], poses[:, 2:, :]], 1) # [-u, r, -t] -> [r, u, -t]
    poses = np.moveaxis(poses, -1, 0).astype(np.float32)
    imgs = np.moveaxis(imgs, -1, 0).astype(np.float32)
    images = imgs
    bds = np.moveaxis(bds, -1, 0).astype(np.float32)
    print("bds:", bds[0])
    
    # Rescale if bd_factor is provided
    sc = 1. if bd_factor is None else 1./(bds.min() * bd_factor)
    poses[:,:3,3] *= sc
    # print("what is rescaled:", poses[:,:3,3])
    bds *= sc
    
    # print('before recenter:\n', poses[0])

    if recenter:
        poses = recenter_poses(poses)
        
    if spherify:
        poses, render_poses, bds = spherify_poses(poses, bds)

    else:
        
        c2w = poses_avg(poses)
        print('recentered', c2w.shape)
        print(c2w[:3,:4])

        ## Get spiral
        # Get average pose
        up = normalize(poses[:, :3, 1].sum(0))

        # Find a reasonable "focus depth" for this dataset
        close_depth, inf_depth = bds.min()*.9, bds.max()*5.
        dt = .75
        mean_dz = 1./(((1.-dt)/close_depth + dt/inf_depth))
        focal = mean_dz

        # Get radii for spiral path
        shrink_factor = .8
        zdelta = close_depth * .2
        tt = poses[:,:3,3] # ptstocam(poses[:3,3,:].T, c2w).T
        rads = np.percentile(np.abs(tt), 90, 0)
        c2w_path = c2w
        N_views = 120
        N_rots = 2
        if path_zflat:
#             zloc = np.percentile(tt, 10, 0)[2]
            zloc = -close_depth * .1
            c2w_path[:3,3] = c2w_path[:3,3] + zloc * c2w_path[:3,2]
            rads[2] = 0.
            N_rots = 1
            N_views/=2

        # Generate poses for spiral path
        render_poses = render_path_spiral(c2w_path, up, rads, focal, zdelta, zrate=.5, rots=N_rots, N=N_views)
        
        
    render_poses = np.array(render_poses).astype(np.float32)

    c2w = poses_avg(poses)
    print('Data:')
    print(poses.shape, images.shape, bds.shape)
    
    dists = np.sum(np.square(c2w[:3,3] - poses[:,:3,3]), -1)
    i_test = np.argmin(dists)
    print('HOLDOUT view is', i_test)
    
    images = images.astype(np.float32)
    poses = poses.astype(np.float32)

    return images, poses, bds, render_poses, i_test


def get_poses(images):
    poses = []
    for i in images:
        R = images[i].qvec2rotmat()
        t = images[i].tvec.reshape([3,1])
        bottom = np.array([0,0,0,1.]).reshape([1,4])
        w2c = np.concatenate([np.concatenate([R, t], 1), bottom], 0)
        c2w = np.linalg.inv(w2c)
        poses.append(c2w)
    return np.array(poses)

def load_colmap_depth(basedir, factor=8, bd_factor=.75):
    data_file = Path(basedir) / 'colmap_depth.npy'
    
    images = read_images_binary(Path(basedir) / 'sparse' / '0' / 'images.bin')
    points = read_points3d_binary(Path(basedir) / 'sparse' / '0' / 'points3D.bin')

    Errs = np.array([point3D.error for point3D in points.values()])
    Err_mean = np.mean(Errs)
    print("Mean Projection Error:", Err_mean)
    
    poses = get_poses(images)
    _, bds_raw, _ = _load_data(basedir, factor=factor) # factor=8 downsamples original imgs by 8x
    bds_raw = np.moveaxis(bds_raw, -1, 0).astype(np.float32)
    # print(bds_raw.shape)
    # Rescale if bd_factor is provided
    sc = 1. if bd_factor is None else 1./(bds_raw.min() * bd_factor)
    
    near = np.ndarray.min(bds_raw) * .9 * sc
    far = np.ndarray.max(bds_raw) * 1. * sc
    print('near/far:', near, far)

    data_list = []
    for id_im in range(1, len(images)+1):
        depth_list = []
        coord_list = []
        weight_list = []
        for i in range(len(images[id_im].xys)):
            point2D = images[id_im].xys[i]
            id_3D = images[id_im].point3D_ids[i]
            if id_3D == -1:
                continue
            point3D = points[id_3D].xyz
            depth = (poses[id_im-1,:3,2].T @ (point3D - poses[id_im-1,:3,3])) * sc
            if depth < bds_raw[id_im-1,0] * sc or depth > bds_raw[id_im-1,1] * sc:
                continue
            err = points[id_3D].error
            weight = 2 * np.exp(-(err/Err_mean)**2)
            depth_list.append(depth)
            coord_list.append(point2D/factor)
            weight_list.append(weight)
        if len(depth_list) > 0:
            print(id_im, len(depth_list), np.min(depth_list), np.max(depth_list), np.mean(depth_list))
            data_list.append({"depth":np.array(depth_list), "coord":np.array(coord_list), "error":np.array(weight_list)})
        else:
            print(id_im, len(depth_list))
    # json.dump(data_list, open(data_file, "w"))
    np.save(data_file, data_list)
    return data_list

def sample_depth(img, sc, factor, N_samples=5000):
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
        sampled_depth.append(depth * sc)
        index.append(idx)
        coord.append([col / factor, row / factor])
        error.append(ERR)

    return sampled_depth, coord, error

def sample_all_depth(img, sc, factor):
    ERR = 0.01

    H, W = img.shape
    sampled_depth = []
    coord = []
    error = []
    for i in range(H):
        for j in range(W):
            depth = img[i, j]
            if depth==0:
                continue
            sampled_depth.append(depth * sc)
            # see img as matrix i is row j is col, see img as coordinate i is y j is x
            coord.append([j / factor, i / factor])
            error.append(ERR)
    print(len(sampled_depth))

    return sampled_depth, coord, error

def make_depths(scene, seq, depthdir, basedir, factor=2, bd_factor=.75, loadfromfile=False):
    filepath = os.path.join(basedir, 'gt_all_depth_bd{}.npy'.format(bd_factor))
    if loadfromfile == True and os.path.exists(filepath):
        print('load depth from file...')
        depth_gts = np.load(filepath, allow_pickle=True)
    else:
        print('collect depth from database...')
        imgdir = os.path.join(basedir, 'images')
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
            depth, coord, error = sample_all_depth(depth, sc, factor)
            depth_gts.append({"depth":np.array(depth), "coord":np.array(coord), "error":np.array(error)})

        np.save(os.path.join(basedir, 'gt_all_depth_bd{}.npy'.format(bd_factor)), depth_gts)

    return depth_gts

def load_7Scenes_data(args):

    h = args.h # 480
    w = args.w # 640
    f = args.f # 585.
    hwf = np.array([h,w,f]).reshape([3,1])

    # decide path to load data
    data_dir = os.path.join(args.dbdir, args.scene)
    train_split = os.path.join(data_dir, 'TrainSplit.txt')
    test_split = os.path.join(data_dir, 'TestSplit.txt')
    with open(train_split, 'r') as f:
        train_seqs = [int(l.split('sequence')[-1]) for l in f if not l.startswith('#')]
    with open(test_split, 'r') as f:
        test_seqs = [int(l.split('sequence')[-1]) for l in f if not l.startswith('#')]
    

    # collect poses, color images and depth images for train set
    pose_list = []
    train_imgs = []
    bd_list = []
    depthfilepath = os.path.join(args.datadir, 'gt_depth_train.npy')
    if args.loaddepth == True and os.path.exists(depthfilepath):
        print('load depth from file...')
        depth_gts = np.load(depthfilepath, allow_pickle=True)
    else:
        print('collect depth from database...')
        depth_gts = []
    for seq in train_seqs:
        seq_data_dir = os.path.join(data_dir, 'seq-{:02d}'.format(seq))

        p_filenames = [n for n in os.listdir(seq_data_dir) if n.find('pose') >= 0]
        idxes = [int(n[6:12]) for n in p_filenames]
        frame_idx = np.array(sorted(idxes))

        if args.trainskip > 1:
            frame_idx_tmp = frame_idx[::args.trainskip]
            frame_idx = frame_idx_tmp

        # load pose
        for i in frame_idx:
            posepath = os.path.join(seq_data_dir, 'frame-{:06d}.pose.txt'.format(i))
            pose = np.loadtxt(posepath)
            pose_list.append(pose)
            bd_list.append([0.4, 2.5])

        # load color images
        c_imgs = [imageio.imread(os.path.join(seq_data_dir, 'frame-{:06d}.color.png'.format(i)))[...,:3]/255. for i in frame_idx]
        train_imgs.extend(c_imgs)

        # load depth images
        if args.loaddepth == False:
            for i in frame_idx:
                depthpath = os.path.join(seq_data_dir, 'frame-{:06d}.depth.png'.format(i))
                d_img = Image.open(depthpath)
                d_img = np.array(d_img)
                d_img[d_img==65535] = 0
                d_img = d_img.astype(np.float32)*0.001
                # get near far original method is to get 0.5% and 99.5% as the near and far
                # near = d_img[d_img > 0].min()
                # far = d_img.max()
                # bd_list.append([near, far])
                # print('original near/far:', near, far)
                # sample depths
                depth, coord, error = sample_depth(d_img, sc=1.0, factor=1.0)
                depth_gts.append({"depth":np.array(depth), "coord":np.array(coord), "error":np.array(error)})

    # preprocess pose
    # prepare for gt_pose_bounds.npy
    pose_list = np.stack(pose_list, 0)
    poses = pose_list[:, :3, :4].transpose([1,2,0])
    poses = np.concatenate([poses, np.tile(hwf[..., np.newaxis], [1,1,poses.shape[-1]])], 1)
    poses = np.concatenate([poses[:, 1:2, :], poses[:, 0:1, :], -poses[:, 2:3, :], poses[:, 3:4, :], poses[:, 4:5, :]], 1)

    # save train pose bounds
    poses_arr = []
    for i in range(poses.shape[-1]):
        poses_arr.append(np.concatenate([poses[..., i].ravel(), np.array(bd_list[i])], 0))

    poses_arr = np.array(poses_arr)
    np.save(os.path.join(args.datadir, 'gt_poses_bounds_train.npy'), poses_arr)

    # save sampled depths
    if args.loaddepth == True:
        pass
    else:
        np.save(os.path.join(args.datadir, 'gt_depth_train.npy'), depth_gts)

    # load pose, images as the required data format in _load_data()
    # omit rescale by factor
    train_poses = poses_arr[:, :-2].reshape([-1, 3, 5]).transpose([1,2,0]) # 3 x 5 x N
    bds = poses_arr[:, -2:].transpose([1,0])
    train_imgs = np.stack(train_imgs, -1)

    # load pose, images as the required data format in load_llff_data()
    # omit rescale by bd_factor
    train_poses = np.concatenate([train_poses[:, 1:2, :], -train_poses[:, 0:1, :], train_poses[:, 2:, :]], 1) # [-u, r, -t] -> [r, u, -t]
    train_poses = np.moveaxis(train_poses, -1, 0).astype(np.float32)
    train_imgs = np.moveaxis(train_imgs, -1, 0).astype(np.float32)
    bds = np.moveaxis(bds, -1, 0).astype(np.float32)

    # collect poses, color images and depth images for test set
    pose_list_t = []
    test_imgs = []
    bd_list_t = [] # near far for test set are supposed not to be obtained
    for seq in test_seqs:
        seq_data_dir = os.path.join(data_dir, 'seq-{:02d}'.format(seq))

        p_filenames = [n for n in os.listdir(seq_data_dir) if n.find('pose') >= 0]
        idxes = [int(n[6:12]) for n in p_filenames]
        frame_idx = np.array(sorted(idxes))

        if args.testskip > 1:
            frame_idx_tmp = frame_idx[::args.testskip]
            frame_idx = frame_idx_tmp

        # load pose
        for i in frame_idx:
            posepath = os.path.join(seq_data_dir, 'frame-{:06d}.pose.txt'.format(i))
            pose = np.loadtxt(posepath)
            pose_list_t.append(pose)
            bd_list_t.append([0.4, 2.5])

        # load color images
        c_imgs_t = [imageio.imread(os.path.join(seq_data_dir, 'frame-{:06d}.color.png'.format(i)))[...,:3]/255. for i in frame_idx]
        test_imgs.extend(c_imgs_t)

    # preprocess pose
    # prepare for gt_pose_bounds.npy
    pose_list_t = np.stack(pose_list_t, 0)
    poses_t = pose_list_t[:, :3, :4].transpose([1,2,0])
    poses_t = np.concatenate([poses_t, np.tile(hwf[..., np.newaxis], [1,1,poses_t.shape[-1]])], 1)
    poses_t = np.concatenate([poses_t[:, 1:2, :], poses_t[:, 0:1, :], -poses_t[:, 2:3, :], poses_t[:, 3:4, :], poses_t[:, 4:5, :]], 1)

    # save test pose bounds
    poses_arr_t = []
    for i in range(poses_t.shape[-1]):
        poses_arr_t.append(np.concatenate([poses_t[..., i].ravel(), np.array(bd_list_t[i])], 0))

    poses_arr_t = np.array(poses_arr_t)
    np.save(os.path.join(args.datadir, 'gt_poses_bounds_test.npy'), poses_arr_t)

    # load pose, images as the required data format in _load_data()
    # omit rescale by factor
    test_poses = poses_arr_t[:, :-2].reshape([-1, 3, 5]).transpose([1,2,0]) # 3 x 5 x N
    test_imgs = np.stack(test_imgs, -1)
    
    # load pose, images as the required data format in load_llff_data()
    # omit rescale by bd_factor
    test_poses = np.concatenate([test_poses[:, 1:2, :], -test_poses[:, 0:1, :], test_poses[:, 2:, :]], 1) # [-u, r, -t] -> [r, u, -t]
    test_poses = np.moveaxis(test_poses, -1, 0).astype(np.float32)
    test_imgs = np.moveaxis(test_imgs, -1, 0).astype(np.float32)

    # set render poses
    render_poses = test_poses

    return train_imgs, test_imgs, train_poses, test_poses, render_poses, depth_gts, bds

def load_sensor_depth(basedir, factor=8, bd_factor=.75):
    data_file = Path(basedir) / 'colmap_depth.npy'
    
    images = read_images_binary(Path(basedir) / 'sparse' / '0' / 'images.bin')
    points = read_points3d_binary(Path(basedir) / 'sparse' / '0' / 'points3D.bin')

    Errs = np.array([point3D.error for point3D in points.values()])
    Err_mean = np.mean(Errs)
    print("Mean Projection Error:", Err_mean)
    
    poses = get_poses(images)
    _, bds_raw, _ = _load_data(basedir, factor=factor) # factor=8 downsamples original imgs by 8x
    bds_raw = np.moveaxis(bds_raw, -1, 0).astype(np.float32)
    # print(bds_raw.shape)
    # Rescale if bd_factor is provided
    sc = 1. if bd_factor is None else 1./(bds_raw.min() * bd_factor)
    
    near = np.ndarray.min(bds_raw) * .9 * sc
    far = np.ndarray.max(bds_raw) * 1. * sc
    print('near/far:', near, far)

    depthfiles = [Path(basedir) / 'depth' / f for f in sorted(os.listdir(Path(basedir) / 'depth')) if f.endswith('JPG') or f.endswith('jpg') or f.endswith('png')]
    depths = [imageio.imread(f) for f in depthfiles]
    depths = np.stack(depths, 0)

    data_list = []
    for id_im in range(1, len(images)+1):
        depth_list = []
        coord_list = []
        weight_list = []
        for i in range(len(images[id_im].xys)):
            point2D = images[id_im].xys[i]
            id_3D = images[id_im].point3D_ids[i]
            if id_3D == -1:
                continue
            point3D = points[id_3D].xyz
            depth = (poses[id_im-1,:3,2].T @ (point3D - poses[id_im-1,:3,3])) * sc
            if depth < bds_raw[id_im-1,0] * sc or depth > bds_raw[id_im-1,1] * sc:
                continue
            err = points[id_3D].error
            weight = 2 * np.exp(-(err/Err_mean)**2)
            depth_list.append(depth)
            coord_list.append(point2D/factor)
            weight_list.append(weight)
        if len(depth_list) > 0:
            print(id_im, len(depth_list), np.min(depth_list), np.max(depth_list), np.mean(depth_list))
            data_list.append({"depth":np.array(depth_list), "coord":np.array(coord_list), "weight":np.array(weight_list)})
        else:
            print(id_im, len(depth_list))
    # json.dump(data_list, open(data_file, "w"))
    np.save(data_file, data_list)
    return data_list

def load_colmap_llff(basedir):
    basedir = Path(basedir)

    train_imgs = np.load(basedir / 'train_images.npy')
    test_imgs = np.load(basedir / 'test_images.npy')
    train_poses = np.load(basedir / 'train_poses.npy')
    test_poses = np.load(basedir / 'test_poses.npy')
    video_poses = np.load(basedir / 'video_poses.npy')
    depth_data = np.load(basedir / 'train_depths.npy', allow_pickle=True)
    bds = np.load(basedir / 'bds.npy')

    return train_imgs, test_imgs, train_poses, test_poses, video_poses, depth_data, bds

    

