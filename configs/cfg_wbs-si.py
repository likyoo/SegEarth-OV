_base_ = './base_config.py'

# model settings
model = dict(
    name_path='./configs/cls_wbs-si.txt',
    prob_thd=0.6,
)

# dataset settings
dataset_type = 'WaterDataset'
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
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='tools/dataset_converters/wbs-si_val.txt',
        data_prefix=dict(
            img_path='data/water-body-segmentation-in-satellite-images/WaterBodiesDatasetPreprocessed/WaterBodiesDatasetPreprocessed/Images',
            seg_map_path='data/water-body-segmentation-in-satellite-images/WaterBodiesDatasetPreprocessed/WaterBodiesDatasetPreprocessed/Masks_cvt'),
        pipeline=test_pipeline))