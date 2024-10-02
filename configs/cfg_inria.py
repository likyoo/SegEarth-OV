_base_ = './base_config.py'

# model settings
model = dict(
    name_path='./configs/cls_inria.txt',
    prob_thd=0.6,
)

# dataset settings
dataset_type = 'InriaDataset'
data_root = ''

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(448, 448), keep_ratio=True),
    # add loading annotation after ``Resize`` because ground truth
    # does not need to do resize data transform
    dict(type='LoadAnnotations'),
    dict(type='PackSegInputs')
]

test_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path='data/Inria/AerialImageDataset/train/img_dir/split_test',
            seg_map_path='data/Inria/AerialImageDataset/train/ann_dir/split_test'),
        pipeline=test_pipeline))