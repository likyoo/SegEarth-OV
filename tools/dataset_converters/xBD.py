"""
python datasets/cvt_xBD.py data/rs_seg/Building/xBD/test -o data/rs_seg/Building/xBD/test

Then, move the images and labels with "pre" in the filename to a new directory.

"""
import argparse
import os
import os.path as osp
import cv2

from mmengine.utils import mkdir_or_exist


def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert xBD dataset to mmsegmentation format')
    parser.add_argument('dataset_path', help='xBD folder path')
    parser.add_argument('-o', '--out_dir', help='output path')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    dataset_path = args.dataset_path
    if args.out_dir is None:
        out_dir = osp.join('data', 'xBD')
    else:
        out_dir = args.out_dir

    print('Making directories...')
    mkdir_or_exist(out_dir)
    mkdir_or_exist(osp.join(out_dir, 'targets_cvt'))

    dataset_path = osp.join(dataset_path, 'targets')
    for img_name in os.listdir(dataset_path):
        img = cv2.imread(osp.join(dataset_path, img_name), cv2.IMREAD_GRAYSCALE)
        img[img >= 1] = 1
        cv2.imwrite(osp.join(out_dir, 'targets_cvt', img_name.replace('_target', '')), img)

    print('Done!')


if __name__ == '__main__':
    main()