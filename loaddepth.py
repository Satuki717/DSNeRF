from email.mime import base
import os
import cv2
import numpy as np

basedir = '/mnt/datagrid1/yyuan/7scenes/heads/seq-02'
outdir = 'data/heads3'

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

if __name__=='__main__':
    render_depth(basedir, outdir)