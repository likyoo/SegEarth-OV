import logging
from typing import Any, Union, List, Optional, Tuple, Dict
import open_clip
from open_clip.constants import OPENAI_DATASET_MEAN, OPENAI_DATASET_STD

import torch
from torchvision import transforms
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import cv2 as cv2

from .gem_wrapper import GEMWrapper


_MODELS = {
    # B/32
    "ViT-B/32": [
        "openai",
        "laion400m_e31",
        "laion400m_e32",
        "laion2b_e16",
        "laion2b_s34b_b79k",
    ],

    "ViT-B/32-quickgelu": [
        "metaclip_400m",
        "metaclip_fullcc"
    ],
    # B/16
    "ViT-B/16": [
        "openai",
        "laion400m_e31",
        "laion400m_e32",
        "laion2b_s34b_b88k",
    ],
    "ViT-B/16-quickgelu": [
        "metaclip_400m",
        "metaclip_fullcc",
    ],
    "ViT-B/16-plus-240": [
        "laion400m_e31",
        "laion400m_e32"
    ],
    # L/14
    "ViT-L/14": [
        "openai",
        "laion400m_e31",
        "laion400m_e32",
        "laion2b_s32b_b82k",
    ],
    "ViT-L/14-quickgelu": [
        "metaclip_400m",
        "metaclip_fullcc"
    ],
    "ViT-L/14-336": [
        "openai",
    ]
}

def available_models() -> List[str]:
    """Returns the names of available GEM-VL models"""
    # _str = "".join([": ".join([key, value]) + "\n" for key, values in _MODELS2.items() for value in values])
    _str = "".join([": ".join([key + " "*(20 - len(key)), value]) + "\n" for key, values in _MODELS.items() for value in values])
    return _str

def get_tokenizer(
        model_name: str = '',
        context_length: Optional[int] = None,
        **kwargs,
):
    """ Wrapper around openclip get_tokenizer function """
    return open_clip.get_tokenizer(model_name=model_name, context_length=context_length, **kwargs)


def get_gem_img_transform(
        img_size:  Union[int, Tuple[int, int]] = (448, 448),
        mean: Optional[Tuple[float, ...]] = None,
        std: Optional[Tuple[float, ...]] = None,
):
    mean = mean or OPENAI_DATASET_MEAN
    std = std or OPENAI_DATASET_STD
    transform = transforms.Compose([
        transforms.Resize(size=img_size, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    return transform


def create_gem_model(
        model_name: str,
        pretrained: Optional[str] = None,
        gem_depth: int = 7,
        ss_attn_iter: int = 1,
        ss_attn_temp: Optional[float] = None,
        precision: str = 'fp32',
        device: Union[str, torch.device] = 'cpu',
        jit: bool = False,
        force_quick_gelu: bool = False,
        force_custom_text: bool = False,
        force_patch_dropout: Optional[float] = None,
        force_image_size: Optional[Union[int, Tuple[int, int]]] = None,
        force_preprocess_cfg: Optional[Dict[str, Any]] = None,
        pretrained_image: bool = False,
        pretrained_hf: bool = True,
        cache_dir: Optional[str] = None,
        output_dict: Optional[bool] = None,
        require_pretrained: bool = False,
        ignore_residual: bool = False,
        **model_kwargs,
):
    model_name = model_name.replace("/", "-")
    logging.info(f'Loading pretrained {model_name} from pretrained weights {pretrained}...')
    open_clip_model = open_clip.create_model(model_name, pretrained, precision, device, jit, force_quick_gelu, force_custom_text,
                                  force_patch_dropout, force_image_size, force_preprocess_cfg, pretrained_image,
                                  pretrained_hf, cache_dir, output_dict, require_pretrained, **model_kwargs)
    tokenizer = open_clip.get_tokenizer(model_name=model_name)

    gem_model = GEMWrapper(model=open_clip_model, tokenizer=tokenizer, depth=gem_depth,
                           ss_attn_iter=ss_attn_iter, ss_attn_temp=ss_attn_temp, ignore_residual=ignore_residual)
    logging.info(f'Loaded GEM-{model_name} from pretrained weights {pretrained}!')
    return gem_model

def create_model_and_transforms(
        model_name: str,
        pretrained: Optional[str] = None,
        gem_depth: int = 7,
        precision: str = 'fp32',
        device: Union[str, torch.device] = 'cpu',
        jit: bool = False,
        force_quick_gelu: bool = False,
        force_custom_text: bool = False,
        force_patch_dropout: Optional[float] = None,
        force_image_size: Optional[Union[int, Tuple[int, int]]] = None,
        force_preprocess_cfg: Optional[Dict[str, Any]] = None,
        pretrained_image: bool = False,
        pretrained_hf: bool = True,
        cache_dir: Optional[str] = None,
        output_dict: Optional[bool] = None,
        require_pretrained: bool = False,
        **model_kwargs,
):
    gem_model = create_gem_model(model_name, pretrained, gem_depth, precision, device, jit, force_quick_gelu, force_custom_text,
                                 force_patch_dropout, force_image_size, force_preprocess_cfg, pretrained_image,
                                 pretrained_hf, cache_dir, output_dict, require_pretrained, **model_kwargs)

    transform = get_gem_img_transform(**model_kwargs)
    return gem_model, transform

def visualize(image, text, logits, alpha=0.6, save_path=None):
    W, H = logits.shape[-2:]
    if isinstance(image, Image.Image):
        image = image.resize((W, H))
    elif isinstance(image, torch.Tensor):
        if image.ndim > 3:
            image = image.squeeze(0)
        image_unormed = (image.detach().cpu() * torch.Tensor(OPENAI_DATASET_STD)[:, None, None]) \
                        + torch.Tensor(OPENAI_DATASET_MEAN)[:, None, None]  # undo the normalization
        image = Image.fromarray((image_unormed.permute(1, 2, 0).numpy() * 255).astype('uint8'))  # convert to PIL
    else:
        raise f'image should be either of type PIL.Image.Image or torch.Tensor but found {type(image)}'

    # plot image
    plt.imshow(image)
    plt.axis('off')
    plt.tight_layout()
    plt.show()

    if logits.ndim > 3:
        logits = logits.squeeze(0)
    logits = logits.detach().cpu().numpy()


    img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    logits = (logits * 255).astype('uint8')
    heat_maps = [cv2.applyColorMap(logit, cv2.COLORMAP_JET) for logit in logits]

    vizs = [(1 - alpha) * img_cv + alpha * heat_map for heat_map in heat_maps]
    for viz, cls_name in zip(vizs, text):

        viz = cv2.cvtColor(viz.astype('uint8'), cv2.COLOR_BGR2RGB)
        plt.imshow(viz)
        plt.title(cls_name)
        plt.axis('off')
        plt.tight_layout()
        plt.show()
        if save_path is not None:
            plt.savefig(f'heatmap_{cls_name}.png')
