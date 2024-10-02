import argparse
import glob
import math
import os
import os.path as osp

import mmcv
import numpy as np
from mmengine.utils import ProgressBar, mkdir_or_exist


def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert Inria dataset to mmsegmentation format')
    parser.add_argument('dataset_path', help='Inria folder path')
    parser.add_argument('-o', '--out_dir', help='output path')
    parser.add_argument(
        '--clip_size',
        type=int,
        help='clipped size of image after preparation',
        default=1000)
    parser.add_argument(
        '--stride_size',
        type=int,
        help='stride of clipping original images',
        default=1000)
    args = parser.parse_args()
    return args


def clip_big_image(image_path, clip_save_dir, args, to_label=False):
    # Original image of Potsdam dataset is very large, thus pre-processing
    # of them is adopted. Given fixed clip size and stride size to generate
    # clipped image, the intersection　of width and height is determined.
    # For example, given one 5120 x 5120 original image, the clip size is
    # 512 and stride size is 256, thus it would generate 20x20 = 400 images
    # whose size are all 512x512.
    image = mmcv.imread(image_path, 'color') if not to_label \
        else mmcv.imread(image_path, 'grayscale')
    
    if to_label:
        image[image < 128] = 0
        image[image >= 128] = 1

    h, w = image.shape[:2]
    clip_size = args.clip_size
    stride_size = args.stride_size

    num_rows = math.ceil((h - clip_size) / stride_size) if math.ceil(
        (h - clip_size) /
        stride_size) * stride_size + clip_size >= h else math.ceil(
            (h - clip_size) / stride_size) + 1
    num_cols = math.ceil((w - clip_size) / stride_size) if math.ceil(
        (w - clip_size) /
        stride_size) * stride_size + clip_size >= w else math.ceil(
            (w - clip_size) / stride_size) + 1

    x, y = np.meshgrid(np.arange(num_cols + 1), np.arange(num_rows + 1))
    xmin = x * clip_size
    ymin = y * clip_size

    xmin = xmin.ravel()
    ymin = ymin.ravel()
    xmin_offset = np.where(xmin + clip_size > w, w - xmin - clip_size,
                           np.zeros_like(xmin))
    ymin_offset = np.where(ymin + clip_size > h, h - ymin - clip_size,
                           np.zeros_like(ymin))
    boxes = np.stack([
        xmin + xmin_offset, ymin + ymin_offset,
        np.minimum(xmin + clip_size, w),
        np.minimum(ymin + clip_size, h)
    ],
                     axis=1)

    for box in boxes:
        start_x, start_y, end_x, end_y = box
        clipped_image = image[start_y:end_y,
                              start_x:end_x] if to_label else image[
                                  start_y:end_y, start_x:end_x, :]
        base_name = osp.basename(image_path).split('.')[0].replace('_24label', '')
        mmcv.imwrite(
            clipped_image.astype(np.uint8),
            osp.join(
                clip_save_dir,
                f'{base_name}_{start_x}_{start_y}_{end_x}_{end_y}.png'))


def main():
    args = parse_args()
    splits = {
        'split_val': ['chicago24.tif', 'austin27.tif', 'kitsap22.tif', 'chicago28.tif', 'tyrol-w6.tif', 'austin8.tif', 
                'kitsap5.tif', 'vienna28.tif', 'kitsap30.tif', 'austin15.tif', 'kitsap13.tif', 'austin19.tif', 'chicago6.tif', 
                'austin9.tif', 'chicago12.tif', 'vienna7.tif', 'kitsap18.tif', 'tyrol-w24.tif', 'vienna36.tif', 'austin29.tif', 
                'tyrol-w19.tif', 'vienna29.tif', 'chicago1.tif', 'tyrol-w12.tif', 'tyrol-w22.tif', 'austin17.tif', 'tyrol-w10.tif'],
        'split_test': ['vienna3.tif', 'chicago10.tif', 'vienna20.tif', 'austin1.tif', 'tyrol-w25.tif', 'vienna9.tif', 'chicago14.tif', 
                     'vienna18.tif','kitsap36.tif','kitsap35.tif','austin22.tif','kitsap34.tif','austin7.tif','vienna11.tif',
                     'kitsap14.tif','kitsap25.tif','austin12.tif','vienna15.tif','kitsap9.tif','austin36.tif','vienna6.tif',
                     'chicago23.tif','tyrol-w3.tif','chicago26.tif','austin24.tif','tyrol-w1.tif','tyrol-w36.tif']
    }

    dataset_path = args.dataset_path
    if args.out_dir is None:
        out_dir = osp.join('data', 'Inria')
    else:
        out_dir = args.out_dir

    print('Making directories...')
    mkdir_or_exist(osp.join(out_dir, 'img_dir', 'split_test'))
    mkdir_or_exist(osp.join(out_dir, 'img_dir', 'split_val'))
    mkdir_or_exist(osp.join(out_dir, 'ann_dir', 'split_test'))
    mkdir_or_exist(osp.join(out_dir, 'ann_dir', 'split_val'))

    rgb_8bit_dir = osp.join(dataset_path, 'images')
    ann_idx_dir = osp.join(dataset_path, 'gt')
    src_path_list = glob.glob(os.path.join(rgb_8bit_dir, '*.tif'))

    prog_bar = ProgressBar(len(src_path_list))
    for i, src_path in enumerate(src_path_list):
        ann_path = osp.join(ann_idx_dir, osp.basename(src_path))
        if osp.basename(src_path) in splits['split_val']:
            data_type = 'split_val'
        elif osp.basename(src_path) in splits['split_test']:
            data_type = 'split_test'
        else:
            data_type = 'split_train'
        
        if data_type == 'split_train':
            continue

        dst_dir = osp.join(out_dir, 'ann_dir', data_type)
        clip_big_image(ann_path, dst_dir, args, to_label=True)

        dst_dir = osp.join(out_dir, 'img_dir', data_type)
        clip_big_image(src_path, dst_dir, args, to_label=False)
        prog_bar.update()

    print('Done!')


if __name__ == '__main__':
    main()