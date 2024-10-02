_base_ = './base_config.py'

# model settings
model = dict(
    name_path='./configs/cls_whu.txt',
    prob_thd=0.7,
)

# dataset settings
dataset_type = 'WHUDataset'
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
        img_suffix='.tif',
        seg_map_suffix='.tif',
        data_prefix=dict(
            img_path='data/WHU_Sat_II/Satellite_dataset_Ⅱ_East_Asia/1.cropped/test/image', 
            seg_map_path='data/WHU_Sat_II/Satellite_dataset_Ⅱ_East_Asia/1.cropped/test/label_cvt'),
        pipeline=test_pipeline))