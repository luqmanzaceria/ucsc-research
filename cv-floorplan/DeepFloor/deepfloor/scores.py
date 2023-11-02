import argparse
import os
import time

import pandas as pd
from scipy.spatial import distance_matrix
from scipy.spatial.distance import cosine
from scipy.spatial.distance import euclidean
from skimage.io import imread
from skimage.transform import resize as imresize

from floor_plan_model.utils.rgb_ind_convertor import *
from floor_plan_model.utils.util import fast_hist

parser = argparse.ArgumentParser()

parser.add_argument('--dataset', type=str, default='R3D',
                    help='define the benchmark')

parser.add_argument('--result_dir', type=str, default='./out',
                    help='define the storage folder of network prediction')

def evaluate_semantic(benchmark_path, result_dir, num_of_classes=11, need_merge_result=True, im_downsample=True,
                      gt_downsample=True):
    """Calculate accuracy scores
    
    Parameters
    ----------
    benchmark_path : path
        path to benchmark ground-truth .txt file (i.e.: r3d_train.txt)
    result_dir : path
        path to output directory
    num_of_classes : int
        default is 11
    need_merge_results : boolean
        funtion runs when True
    im_downsample : boolean
        function runs when True
    gt_downsample : boolean
        function runs when True
    """
    gt_paths = open(benchmark_path, 'r').read().splitlines()
    d_paths = [p.split('\t')[2] for p in gt_paths]  # 1 denote wall, 2 denote door, 3 denote room
    r_paths = [p.split('\t')[3] for p in gt_paths]  # 1 denote wall, 2 denote door, 3 denote room
    cw_paths = [p.split('\t')[-1] for p in
                gt_paths]  # 1 denote wall, 2 denote door, 3 denote room, last one denote close wall
    im_names = [p.split('/')[-1].split('.')[0] for p in gt_paths]
    im_paths = [os.path.join(result_dir, p.split('/')[-1].split('.')[0] + '_pred.png') for p in r_paths]
    if need_merge_result:
        im_d_paths = [os.path.join(result_dir, p.split('/')[-1].split('.')[0].replace('close', 'doors_windows.png')) for
                      p in d_paths]
        im_cw_paths = [os.path.join(result_dir, p.split('/')[-1].split('.')[0].replace('close_wall', 'close_walls.png'))
                       for p in cw_paths]

    n = len(im_paths)
    # n = 1
    hist = np.zeros((num_of_classes, num_of_classes))
    for i in range(n):
        im = imread(im_paths[i], pilmode='RGB')
        if need_merge_result:
            im_d = imread(im_d_paths[i], pilmode='L')
            im_cw = imread(im_cw_paths[i], pilmode='L')
        # create fuse semantic label
        cw = imread(cw_paths[i], pilmode='L')
        dd = imread(d_paths[i], pilmode='L')
        rr = imread(r_paths[i], pilmode='RGB')

        if im_downsample:  # really means resize, not downsample
            im = imresize(im, (512, 512, 3), preserve_range=True)
            if need_merge_result:
                im_d = imresize(im_d, (512, 512), preserve_range=True)
                im_cw = imresize(im_cw, (512, 512), preserve_range=True)
                im_d = im_d / 255
                im_cw = im_cw / 255

        if gt_downsample:  # really means resize, not downsample
            cw = imresize(cw, (512, 512), preserve_range=True)
            dd = imresize(dd, (512, 512), preserve_range=True)
            rr = imresize(rr, (512, 512, 3), preserve_range=True)

        # normalize
        cw = cw / 255
        dd = dd / 255

        im_ind = rgb2ind(im, color_map=floorplan_fuse_map)
        if im_ind.sum() == 0:
            im_ind = rgb2ind(im + 1)
        rr_ind = rgb2ind(rr, color_map=floorplan_fuse_map)
        if rr_ind.sum() == 0:
            rr_ind = rgb2ind(rr + 1)

        if need_merge_result:
            im_d = (im_d > 0.5).astype(np.uint8)
            im_cw = (im_cw > 0.5).astype(np.uint8)
            im_ind[im_cw == 1] = 10
            im_ind[im_d == 1] = 9

        # merge the label and produce
        cw = (cw > 0.5).astype(np.uint8)
        dd = (dd > 0.5).astype(np.uint8)
        rr_ind[cw == 1] = 10
        rr_ind[dd == 1] = 9

        name = im_paths[i].split('/')[-1]
        r_name = r_paths[i].split('/')[-1]

        print('Evaluating {}(im) <=> {}(gt)...'.format(name, r_name))

        hist += fast_hist(im_ind.flatten(), rr_ind.flatten(), num_of_classes)

    print('*' * 60)
    # overall accuracy
    acc = np.diag(hist).sum() / hist.sum()
    print('overall accuracy {:.4}'.format(acc))
    # per-class accuracy, avoid div zero
    acc = np.diag(hist) / (hist.sum(1) + 1e-6)
    print('room-type: mean accuracy {:.4}, room-type+bd: mean accuracy {:.4}'.format(np.nanmean(acc[:7]), (
            np.nansum(acc[:7]) + np.nansum(acc[-2:])) / 9.))
    for t in range(0, acc.shape[0]):
        if t not in [7, 8]:
            print('room type {}th, accuracy = {:.4}'.format(t, acc[t]))

    print('*' * 60)
    # per-class IU, avoid div zero
    iu = np.diag(hist) / (hist.sum(1) + 1e-6 + hist.sum(0) - np.diag(hist))
    print('room-type: mean IoU {:.4}, room-type+bd: mean IoU {:.4}'.format(np.nanmean(iu[:7]), (
            np.nansum(iu[:7]) + np.nansum(iu[-2:])) / 9.))
    for t in range(iu.shape[0]):
        if t not in [7, 8]:  # ignore class 7 & 8
            print('room type {}th, IoU = {:.4}'.format(t, iu[t]))


if __name__ == '__main__':
    FLAGS, unparsed = parser.parse_known_args()

    if FLAGS.dataset.lower() == 'r2v':
        benchmark_path = './dataset/r2v_test.txt'
    else:
        benchmark_path = './dataset/r3d_test.txt'

    result_dir = FLAGS.result_dir

    tic = time.time()
    evaluate_semantic(benchmark_path, result_dir, need_merge_result=False, im_downsample=False,
                      gt_downsample=True)  # same as previous line but 11 classes by combining the opening and wall line

    print("*" * 60)
    print("Evaluate time: {} sec".format(time.time() - tic))
