from email.mime import base
import os
import cv2
import numpy as np
import pickle

basedir = '/mnt/datagrid1/yyuan/7scenes/heads/seq-02'
outdir = 'data/heads3'
retrieval_list_path = '/mnt/datagrid1/yyuan/dockersw/code/Camera_localization_diff_render_refine/inference_coarse_results/retrieval_coarse_estimation/query_pose_prediction_dict.pkl'
database_dir = '/mnt/datagrid1/yyuan/7scenes'


def render_depth(basedir, outdir):
    '''render normalized depth map from original depth map
    basedir: database path
    outdir: output path
    '''
    imgdir = os.path.join(outdir, 'images')
    filename_list = sorted(os.listdir(imgdir))

    for i in filename_list:
        
        depthname = i.replace('color.png','depth.png')
        depth = cv2.imread(os.path.join(basedir, depthname), cv2.IMREAD_UNCHANGED)

        depth = np.array(depth)
        depth[depth==65535]=0
        print('max depth:', np.max(depth))
        depth = depth.astype(np.float32) / np.max(depth) * 255
        # print(depth[100:110,100:110])

        colordepth = cv2.applyColorMap(depth.astype(np.uint8), cv2.COLORMAP_JET)

        cv2.imwrite(os.path.join(outdir, 'depth', depthname), colordepth)

def load_initial_pose1(retrieval_list_path, database_dir, scene='heads'):

    with open(retrieval_list_path, 'rb') as file:
        coarse_pose_list = pickle.load(file)
    
    for key in coarse_pose_list.keys():
        print(key)
    
    test_pose_list = []
    heads_pose_list = coarse_pose_list['chess']

    for key in heads_pose_list.keys():
        print(key)
        print(heads_pose_list[key].shape) # 1000*4*4
        test_pose_list.append(heads_pose_list[key])
        print(len(test_pose_list))

    test_pose_list = np.stack(test_pose_list, 0)
    print(len(test_pose_list))
    
    # print(len(coarse_pose_list), len(coarse_pose_list.keys()), len(coarse_pose_list.values()))

    # coarse_pose_list = []
    # for i in range(len(retrieval_list)):
    #     pose_path = os.path.join(database_dir,retrieval_list[i,1]).replace('color.png','pose.txt')
    #     pose = np.asarray(np.loadtxt(pose_path))
    #     coarse_pose_list.append(pose)
    # coarse_pose_list = np.stack(coarse_pose_list, 0) # N * 4 * 4

    # gt_pose_list = []
    # for i in range(len(retrieval_list)):
    #     pose_path = os.path.join(database_dir,retrieval_list[i,0]).replace('color.png','pose.txt')
    #     pose = np.asarray(np.loadtxt(pose_path))
    #     gt_pose_list.append(pose)
    # gt_pose_list = np.stack(gt_pose_list, 0) # N * 4 * 4

    return

if __name__=='__main__':
    # load_initial_pose1(retrieval_list_path, database_dir)
    for x in range(640):
        for y in range(480):
            if 2*x*y - y**2 ==1024:
                print(x,y)