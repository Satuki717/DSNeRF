import os
import numpy as np

datadir = '/mnt/datagrid3/yyuan/code/DSNeRF/logs/test/heads4_gt_sparse/poserefine'

def num_count(arr,rot=5,trans=0.05):

    cond = (arr[0,:] < rot) & (arr[1,:] < trans)
    count_in_num = np.count_nonzero(cond)
    # print(arr.shape)
    print(count_in_num, '/', arr.shape[1])

    return count_in_num / arr.shape[0]

def calculate_error(datadir):
    filename_list = sorted(os.listdir(datadir))

    print(filename_list[:5])
    roterror_list = []
    transerror_list = []
    for dir in filename_list[:100]:
        error_path = os.path.join(datadir, dir, 'refine_loss.npy')
        refine_loss = np.load(error_path)
        # print(refine_loss)
        roterror_list.append(refine_loss[-1][2])
        transerror_list.append(refine_loss[-1][3])
    
    rot_error = np.array(roterror_list)
    trans_error = np.array(transerror_list)
    # print(roterror_list[:5])
    # print(transerror_list[:5])
    rot_median = np.median(rot_error)
    trans_median = np.median(trans_error)
    array = np.vstack((rot_error, trans_error))
    CountInPercentage = num_count(array)
    print("median absolute error of rot = {}°".format(rot_median))
    print("median absolute error of trans = {}m".format(trans_median))
    print("Percentage of results within 5° and 0.05m = {:.2f}%".format(CountInPercentage))


if __name__=='__main__':
    calculate_error(datadir)