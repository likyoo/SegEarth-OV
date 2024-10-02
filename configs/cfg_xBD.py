_base_ = './base_config.py'

# model settings
model = dict(
    name_path='./configs/cls_xBD.txt',
    prob_thd=0.0,
    cls_token_lambda=0,
)

# dataset settings
dataset_type = 'xBDDataset'
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
            img_path='data/xBD/test/images_pre',
            seg_map_path='data/xBD/test/targets_cvt_pre'),
        pipeline=test_pipeline))