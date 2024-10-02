_base_ = './base_config.py'

# model settings
model = dict(
    name_path='./configs/cls_roadval.txt',
    prob_thd=0.7,
)

# dataset settings
dataset_type = 'RoadValDataset'
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
            img_path='data/GlobalRoadSet_Val/SpaceNet_test_567/img',
            seg_map_path='data/GlobalRoadSet_Val/SpaceNet_test_567/label_cvt'),
        pipeline=test_pipeline))