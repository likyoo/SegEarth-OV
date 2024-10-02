import argparse
import os
import os.path as osp
import shutil
from pathlib import Path

import numpy as np
from mmengine.utils import mkdir_or_exist


def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert OpenEarthMap dataset to mmsegmentation format')
    parser.add_argument('dataset_path', help='OpenEarthMap folder path')
    parser.add_argument('-o', '--out_dir', help='output path')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    dataset_path = args.dataset_path
    if args.out_dir is None:
        out_dir = osp.join('data', 'OpenEarthMap')
    else:
        out_dir = args.out_dir

    print('Making directories...')
    mkdir_or_exist(out_dir)
    mkdir_or_exist(osp.join(out_dir, 'img_dir'))
    mkdir_or_exist(osp.join(out_dir, 'img_dir', 'val'))
    mkdir_or_exist(osp.join(out_dir, 'ann_dir', 'val'))
    
    all_images = [f for f in Path(dataset_path).rglob("*.tif") if "/images/" in str(f)]
    val_images = [str(f) for f in all_images if f.name in \
                 np.loadtxt(osp.join(dataset_path, 'val.txt'), dtype=str)]

    for img_path in val_images:
        shutil.copy(img_path, osp.join(out_dir, 'img_dir', 'val', osp.basename(img_path)))
        shutil.copy(img_path.replace('images', 'labels'), \
                    osp.join(out_dir, 'ann_dir', 'val', osp.basename(img_path)))

    print('Done!')


if __name__ == '__main__':
    main()