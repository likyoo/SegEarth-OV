<div align="center">

<h1>SegEarth-OV: Towards Training-Free Open-Vocabulary Segmentation for Remote Sensing Images</h1>

<div>
    <strong>Make OVSS possible in remote sensing contexts</strong>
</div>

<div>
    <a href='https://likyoo.github.io/' target='_blank'>Kaiyu Li</a><sup>1</sup>&emsp;
    <a href='https://scholar.google.com/citations?user=WTleRV8AAAAJ' target='_blank'>Ruixun Liu</a><sup>1</sup>&emsp;
    <a href='https://gr.xjtu.edu.cn/en/web/caoxiangyong' target='_blank'>Xiangyong Cao</a><sup>✉1</sup>&emsp;
    <a href='https://web.xidian.edu.cn/xrbai' target='_blank'>Xueru Bai</a><sup>2</sup>&emsp;
    <a href='https://faculty.xidian.edu.cn/ZF3' target='_blank'>Feng Zhou</a><sup>2</sup>&emsp;
    <a href='https://gr.xjtu.edu.cn/en/web/dymeng' target='_blank'>Deyu Meng</a><sup>1</sup>&emsp;
    <a href='https://gr.xjtu.edu.cn/en/web/zhiwang' target='_blank'>Zhi Wang</a><sup>1</sup>&emsp;
</div>
<div>
    <sup>1</sup>Xi'an Jiaotong University&emsp;
    <sup>2</sup>Xidian University&emsp;
</div>

<div>
    <h4 align="center">
        • <a href="https://likyoo.github.io/SegEarth-OV/" target='_blank'>[Project]</a> • <a href="https://arxiv.org/abs/2410.01768" target='_blank'>[arXiv]</a> • <a href="https://colab.research.google.com/drive/1a-NNz_2maesvszk4Xff5PKY02_moPqt6#scrollTo=Pz9QGEcFBGtK" target='_blank'>[Colab]</a> •
    </h4>
</div>

<img src="https://github.com/user-attachments/assets/28675180-02de-476e-ad01-7c2138a2a943" width="100%"/>
Visualization and performance of SegEarth-OV on open-vocabulary semantic segmentation of remote sensing images. We evaluate on 17 remote sensing datasets (including semantic segmentation, building extraction, road extraction, and flood detection tasks), and our SegEarth-OV consistently generates high-quality segmentation masks.

</div>

## Abstract
> *Remote sensing image plays an irreplaceable role in fields such as agriculture, water resources, military, and disaster relief. Pixel-level interpretation is a critical aspect of remote sensing image applications; however, a prevalent limitation remains the need for extensive manual annotation. For this, we try to introduce open-vocabulary semantic segmentation (OVSS) into the remote sensing context. However, due to the sensitivity of remote sensing images to low-resolution features, distorted target shapes and ill-fitting boundaries are exhibited in the prediction mask. To tackle this issue, we propose a simple and general upsampler, SimFeatUp, to restore lost spatial information in deep features in a training-free style. Further, based on the observation of the abnormal response of local patch tokens to [CLS] token in CLIP, we propose to execute a straightforward subtraction operation to alleviate the global bias in patch tokens. Extensive experiments are conducted on 17 remote sensing datasets spanning semantic segmentation, building extraction, road detection, and flood detection tasks. Our method achieves an average of 5.8%, 8.2%, 4%, and 15.3% improvement over state-of-the-art methods on 4 tasks. All codes are released.*

## Dependencies and Installation


```
# 1. install SimFeatUp
# refer to https://github.com/likyoo/SimFeatUp

# 2. git clone this repository
git clone https://github.com/likyoo/SegEarth-OV.git
cd SegEarth-OV

# 3. create new anaconda env
conda create -n SegEarth python=3.9
conda activate SegEarth

# install torch and dependencies
pip install -r requirements.txt
# The dependent versions are not strict, and in general you only need to pay attention to mmcv and mmsegmentation.
```


## Datasets
We include the following dataset configurations in this repo: 
1) `Semantic Segmentation`: OpenEarthMap, LoveDA, iSAID, Potsdam, Vaihingen, UAVid<sup>img</sup>, UDD5, VDD
2) `Building Extraction`: WHU<sup>Aerial</sup>, WHU<sup>Sat.Ⅱ</sup>, Inria, xBD<sup>pre</sup>
4) `Road Extraction`: CHN6-CUG, DeepGlobe, Massachusetts, SpaceNet
5) `Water Extraction`: WBS-SI

Please refer to [dataset_prepare.md](https://github.com/likyoo/SegEarth-OV/blob/main/dataset_prepare.md) for dataset preparation.


## Quick Inference
```
python demo.py
```

## Model evaluation
Single-GPU:

```
python eval.py --config ./configs/cfg_DATASET.py --workdir YOUR_WORK_DIR
```

Multi-GPU:
```
bash ./dist_test.sh ./config/cfg_DATASET.py
```

Evaluation on all datasets:
```
python eval_all.py
```
Results will be saved in `results.xlsx`.

## Results

<div>
<img src="https://github.com/user-attachments/assets/445afb44-447b-4aa2-b6e7-c2e7bf2b5450" width="100%"/>
</div>

<div>
<img src="https://github.com/user-attachments/assets/7270efea-2c8e-485c-a47c-48be8718000f" width="100%"/>
</div>

## Citation

```
@article{li2024segearth,
  title={SegEarth-OV: Towards Training-Free Open-Vocabulary Segmentation for Remote Sensing Images},
  author={Li, Kaiyu and Liu, ruixun and Cao, Xiangyong and Bai, Xueru and Zhou, Feng and Meng, Deyu and Wang, Zhi},
  journal={arXiv preprint arXiv:2410.01768},
  year={2024}
}
```

## Acknowledgement
This implementation is based on [ClearCLIP](https://github.com/mc-lan/ClearCLIP) and [FeatUp](https://github.com/mhamilton723/FeatUp). Thanks for the awesome work.

