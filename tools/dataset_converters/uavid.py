import argparse
import glob
import os
import os.path as osp

import mmcv
import numpy as np
from mmengine.utils import ProgressBar, mkdir_or_exist
from PIL import Image

UAVid_palette = \
    {
        0: (0, 0, 0),
        1: (128, 0, 0),
        2: (128, 64, 128),
        3: (192, 0, 192), # Static_Car
        4: (0, 128, 0),
        5: (128, 128, 0),
        6: (64, 64, 0),
        7: (64, 0, 128) # Moving_Car
    }

UAVid_invert_palette = {v: k for k, v in UAVid_palette.items()}


def UAVid_convert_from_color(arr_3d, palette=UAVid_invert_palette):
    """RGB-color encoding to grayscale labels."""
    arr_2d = np.zeros((arr_3d.shape[0], arr_3d.shape[1]), dtype=np.uint8)

    for c, i in palette.items():
        m = np.all(arr_3d == np.array(c).reshape(1, 1, 3), axis=2)
        arr_2d[m] = i if i != 7 else 3 # convert Moving_Car to Static_Car

    return arr_2d


def slide_crop_image(src_path, out_dir, mode, patch_H, patch_W, overlap):
    img = np.asarray(Image.open(src_path).convert('RGB'))

    img_H, img_W, _ = img.shape

    if img_H < patch_H and img_W > patch_W:

        img = mmcv.impad(img, shape=(patch_H, img_W), pad_val=0)

        img_H, img_W, _ = img.shape

    elif img_H > patch_H and img_W < patch_W:

        img = mmcv.impad(img, shape=(img_H, patch_W), pad_val=0)

        img_H, img_W, _ = img.shape

    elif img_H < patch_H and img_W < patch_W:

        img = mmcv.impad(img, shape=(patch_H, patch_W), pad_val=0)

        img_H, img_W, _ = img.shape

    for x in range(0, img_W, patch_W - overlap):
        for y in range(0, img_H, patch_H - overlap):
            x_str = x
            x_end = x + patch_W
            if x_end > img_W:
                diff_x = x_end - img_W
                x_str -= diff_x
                x_end = img_W
            y_str = y
            y_end = y + patch_H
            if y_end > img_H:
                diff_y = y_end - img_H
                y_str -= diff_y
                y_end = img_H

            img_patch = img[y_str:y_end, x_str:x_end, :]
            img_patch = Image.fromarray(img_patch.astype(np.uint8))
            pre_name = src_path.split('/')[-3] + '_'
            image = pre_name + osp.basename(src_path).split('.')[0] + '_' + str(
                y_str) + '_' + str(y_end) + '_' + str(x_str) + '_' + str(
                    x_end) + '.png'
            
            save_path_image = osp.join(out_dir, 'img_dir', mode, str(image))
            img_patch.save(save_path_image, format='BMP')


def slide_crop_label(src_path, out_dir, mode, patch_H, patch_W, overlap):
    label = mmcv.imread(src_path, channel_order='rgb')
    label = UAVid_convert_from_color(label)
    img_H, img_W = label.shape

    if img_H < patch_H and img_W > patch_W:

        label = mmcv.impad(label, shape=(patch_H, img_W), pad_val=255)

        img_H = patch_H

    elif img_H > patch_H and img_W < patch_W:

        label = mmcv.impad(label, shape=(img_H, patch_W), pad_val=255)

        img_W = patch_W

    elif img_H < patch_H and img_W < patch_W:

        label = mmcv.impad(label, shape=(patch_H, patch_W), pad_val=255)

        img_H = patch_H
        img_W = patch_W

    for x in range(0, img_W, patch_W - overlap):
        for y in range(0, img_H, patch_H - overlap):
            x_str = x
            x_end = x + patch_W
            if x_end > img_W:
                diff_x = x_end - img_W
                x_str -= diff_x
                x_end = img_W
            y_str = y
            y_end = y + patch_H
            if y_end > img_H:
                diff_y = y_end - img_H
                y_str -= diff_y
                y_end = img_H

            lab_patch = label[y_str:y_end, x_str:x_end]
            lab_patch = Image.fromarray(lab_patch.astype(np.uint8), mode='P')

            pre_name = src_path.split('/')[-3] + '_'
            image = pre_name + osp.basename(src_path).split('.')[0] + '_' + str(
                y_str) + '_' + str(y_end) + '_' + str(x_str) + '_' + str(
                    x_end) + '.png'
            lab_patch.save(osp.join(out_dir, 'ann_dir', mode, str(image)))


def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert UAVid dataset to mmsegmentation format')
    parser.add_argument('dataset_path', help='UAVid folder path')
    parser.add_argument('-o', '--out_dir', help='output path')

    parser.add_argument(
        '--patch_width',
        default=1080,
        type=int,
        help='Width of the cropped image patch')
    parser.add_argument(
        '--patch_height',
        default=1280,
        type=int,
        help='Height of the cropped image patch')
    parser.add_argument(
        '--overlap_area', default=0, type=int, help='Overlap area')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    dataset_path = args.dataset_path
    # image patch width and height
    patch_H, patch_W = args.patch_width, args.patch_height

    overlap = args.overlap_area  # overlap area

    if args.out_dir is None:
        out_dir = osp.join('data', 'UAVid')
    else:
        out_dir = args.out_dir

    print('Making directories...')
    mkdir_or_exist(osp.join(out_dir, 'img_dir', 'test'))
    mkdir_or_exist(osp.join(out_dir, 'ann_dir', 'test'))

    test_dir = osp.join(dataset_path, 'test_gt')
    src_path_list = glob.glob(os.path.join(test_dir, '**', '*.png'), recursive=True)

    prog_bar = ProgressBar(len(src_path_list))
    for i, src_path in enumerate(src_path_list):
        dataset_mode = 'test'
        if 'Labels' in src_path:
            slide_crop_label(src_path, out_dir, dataset_mode, patch_H,
                                    patch_W, overlap)
        else:
            slide_crop_image(src_path, out_dir, dataset_mode, patch_H,
                                    patch_W, overlap)

        prog_bar.update()

    print('Done!')


if __name__ == '__main__':
    main()