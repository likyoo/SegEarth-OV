import os.path as osp
import mmengine.fileio as fileio

from mmseg.registry import DATASETS
from mmseg.datasets import BaseSegDataset


@DATASETS.register_module()
class OpenEarthMapDataset(BaseSegDataset):
    """OpenEarthMap dataset.

    In segmentation map annotation for OpenEarthMap, 0 is the ignore index.
    ``reduce_zero_label`` should be set to False. The ``img_suffix`` and
    ``seg_map_suffix`` are both fixed to '.tif'.
    """
    METAINFO = dict(
        classes=('background', 'bareland', 'grass', 'pavement', 'road', 'tree',
                 'water', 'cropland', 'building'),
        palette=[[0, 0, 0], [128, 0, 0], [0, 255, 36], [148, 148, 148],
                 [255, 255, 255], [34, 97, 38], [0, 69, 255], [75, 181, 73],
                 [222, 31, 7]])

    def __init__(self,
                 img_suffix='.tif',
                 seg_map_suffix='.tif',
                 reduce_zero_label=False,
                 **kwargs) -> None:
        super().__init__(
            img_suffix=img_suffix,
            seg_map_suffix=seg_map_suffix,
            reduce_zero_label=reduce_zero_label,
            **kwargs)


@DATASETS.register_module()
class WHUDataset(BaseSegDataset):
    """WHU dataset.

    """
    METAINFO = dict(
        classes=('background', 'building'),
        palette=[[0, 0, 0], [255, 255, 255]])

    def __init__(self,
                 img_suffix='.png',
                 seg_map_suffix='.png',
                 reduce_zero_label=False,
                 **kwargs) -> None:
        super().__init__(
            img_suffix=img_suffix,
            seg_map_suffix=seg_map_suffix,
            reduce_zero_label=reduce_zero_label,
            **kwargs)


@DATASETS.register_module()
class xBDDataset(BaseSegDataset):
    """xBD dataset.

    """
    METAINFO = dict(
        classes=('background', 'building'),
        palette=[[0, 0, 0], [255, 255, 255]])

    def __init__(self,
                 img_suffix='.png',
                 seg_map_suffix='.png',
                 reduce_zero_label=False,
                 **kwargs) -> None:
        super().__init__(
            img_suffix=img_suffix,
            seg_map_suffix=seg_map_suffix,
            reduce_zero_label=reduce_zero_label,
            **kwargs)


@DATASETS.register_module()
class CHN6_CUGDataset(BaseSegDataset):
    """CHN6-CUG dataset.

    """
    METAINFO = dict(
        classes=('background', 'road'),
        palette=[[0, 0, 0], [255, 255, 255]])

    def __init__(self,
                 img_suffix='.jpg',
                 seg_map_suffix='.png',
                 reduce_zero_label=False,
                 **kwargs) -> None:
        super().__init__(
            img_suffix=img_suffix,
            seg_map_suffix=seg_map_suffix,
            reduce_zero_label=reduce_zero_label,
            **kwargs)


@DATASETS.register_module()
class RoadValDataset(BaseSegDataset):
    """RoadVal dataset.

    """
    METAINFO = dict(
        classes=('background', 'road'),
        palette=[[0, 0, 0], [255, 255, 255]])

    def __init__(self,
                 img_suffix='.jpg',
                 seg_map_suffix='.png',
                 reduce_zero_label=False,
                 **kwargs) -> None:
        super().__init__(
            img_suffix=img_suffix,
            seg_map_suffix=seg_map_suffix,
            reduce_zero_label=reduce_zero_label,
            **kwargs)


@DATASETS.register_module()
class UAVidDataset(BaseSegDataset):
    """UAVid dataset.

    convert Moving_Car to Static_Car
    """
    METAINFO = dict(
    classes=('background', 'building', 'road', 'car', 'tree', 
             'vegetation', 'human'),
    palette=[[0, 0, 0], [128, 0, 0], [128, 64, 128], [192, 0, 192], 
             [0, 128, 0], [128, 128, 0], [64, 64, 0]])

    def __init__(self,
                 img_suffix='.png',
                 seg_map_suffix='.png',
                 reduce_zero_label=False,
                 ignore_index=255,
                 **kwargs) -> None:
        super().__init__(
            img_suffix=img_suffix,
            seg_map_suffix=seg_map_suffix,
            reduce_zero_label=reduce_zero_label,
            ignore_index=ignore_index,
            **kwargs)


@DATASETS.register_module()
class UDD5Dataset(BaseSegDataset):
    """UDD5 dataset.
    
    """
    METAINFO = dict(
    classes=('vegetation', 'building', 'road', 'vehicle',
             'other'),
    palette=[[107, 142, 35], [102,102,156], [128,64,128],
             [0, 0, 142], [0, 0, 0]])

    def __init__(self,
                 img_suffix='.JPG',
                 seg_map_suffix='.png',
                 reduce_zero_label=False,
                 ignore_index=255,
                 **kwargs) -> None:
        super().__init__(
            img_suffix=img_suffix,
            seg_map_suffix=seg_map_suffix,
            reduce_zero_label=reduce_zero_label,
            ignore_index=ignore_index,
            **kwargs)


@DATASETS.register_module()
class VDDDataset(BaseSegDataset):
    """VDD dataset.
    
    """
    METAINFO = dict(
    classes=('other', 'wall', 'road', 'vegetation', 'vehicle',
             'roof', 'water'))

    def __init__(self,
                 img_suffix='.JPG',
                 seg_map_suffix='.png',
                 reduce_zero_label=False,
                 ignore_index=255,
                 **kwargs) -> None:
        super().__init__(
            img_suffix=img_suffix,
            seg_map_suffix=seg_map_suffix,
            reduce_zero_label=reduce_zero_label,
            ignore_index=ignore_index,
            **kwargs)


@DATASETS.register_module()
class InriaDataset(BaseSegDataset):
    """Inria dataset.

    """
    METAINFO = dict(
        classes=('background', 'building'),
        palette=[[0, 0, 0], [255, 255, 255]])

    def __init__(self,
                 img_suffix='.png',
                 seg_map_suffix='.png',
                 reduce_zero_label=False,
                 **kwargs) -> None:
        super().__init__(
            img_suffix=img_suffix,
            seg_map_suffix=seg_map_suffix,
            reduce_zero_label=reduce_zero_label,
            **kwargs)


@DATASETS.register_module()
class WaterDataset(BaseSegDataset):
    """Water dataset.

    """
    METAINFO = dict(
        classes=('background', 'water'),
        palette=[[0, 0, 0], [0, 235, 255]])

    def __init__(self,
                 img_suffix='.jpg',
                 seg_map_suffix='.jpg',
                 reduce_zero_label=False,
                 **kwargs) -> None:
        super().__init__(
            img_suffix=img_suffix,
            seg_map_suffix=seg_map_suffix,
            reduce_zero_label=reduce_zero_label,
            **kwargs)
