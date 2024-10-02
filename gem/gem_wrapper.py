import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from open_clip.transformer import VisionTransformer

from .gem_utils import SelfSelfAttention, GEMResidualBlock, modified_vit_forward


class GEMWrapper(nn.Module):
    def __init__(self, model, tokenizer, depth=7, ss_attn_iter=1, ss_attn_temp=None, ignore_residual=False):
        super(GEMWrapper, self).__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.depth = depth
        self.ss_attn_iter = ss_attn_iter
        self.ss_attn_temp = ss_attn_temp
        self.patch_size = self.model.visual.patch_size[0]
        self.ignore_residual = ignore_residual
        self.apply_gem()


    def apply_gem(self):
        for i in range(1, self.depth):
            # Extract info from the original ViT
            num_heads = self.model.visual.transformer.resblocks[-i].attn.num_heads
            dim = int(self.model.visual.transformer.resblocks[-i].attn.head_dim * num_heads)
            qkv_bias = True
            # Init the self-self attention layer
            ss_attn = SelfSelfAttention(dim=dim, num_heads=num_heads, qkv_bias=qkv_bias,
                                        ss_attn_iter=self.ss_attn_iter, ss_attn_temp=self.ss_attn_temp)
            # Copy necessary weights
            ss_attn.qkv.weight.data = self.model.visual.transformer.resblocks[-i].attn.in_proj_weight.clone()
            ss_attn.qkv.bias.data = self.model.visual.transformer.resblocks[-i].attn.in_proj_bias.clone()
            ss_attn.proj.weight.data = self.model.visual.transformer.resblocks[-i].attn.out_proj.weight.clone()
            ss_attn.proj.bias.data = self.model.visual.transformer.resblocks[-i].attn.out_proj.bias.clone()
            # Swap the original Attention with our SelfSelfAttention
            self.model.visual.transformer.resblocks[-i].attn = ss_attn
            # Wrap Residual block to handle SelfSelfAttention outputs
            self.model.visual.transformer.resblocks[-i] = GEMResidualBlock(self.model.visual.transformer.resblocks[-i],
                                                                           self.ignore_residual)
        # Modify ViT's forward function
        self.model.visual.forward = modified_vit_forward.__get__(self.model.visual, VisionTransformer)
        return

    def encode_text(self, text: list):
        prompts = [f'a photo of a {cls}.' for cls in text]
        tokenized_prompts = self.tokenizer(prompts).to(self.model.visual.proj.device)
        text_embedding = self.model.encode_text(tokenized_prompts)
        text_embedding = F.normalize(text_embedding, dim=-1)
        return text_embedding.unsqueeze(0)

    def min_max(self, logits):
        B, num_prompt = logits.shape[:2]
        logits_min = logits.reshape(B, num_prompt, -1).min(dim=-1, keepdim=True)[0].unsqueeze(-1)
        logits_max = logits.reshape(B, num_prompt, -1).max(dim=-1, keepdim=True)[0].unsqueeze(-1)
        logits = (logits - logits_min) / (logits_max - logits_min)
        return logits

    def forward(self, image: torch.Tensor, text: list, normalize: bool = True, return_ori: bool =False, ignore_residual: bool = False):
        """
        :param image: torch.Tensor [1, 3, H, W]
        :param text: list[]
        :param normalize: bool - if True performs min-max normalization
        :param return_ori: bool - if True uses the features from the original visual encoder
        """
        # Image
        W, H = image.shape[-2:]
        feat_gem, feat_ori = self.model.visual(image, ignore_residual)
        image_feat = feat_ori if return_ori else feat_gem
        image_feat = F.normalize(image_feat, dim=-1)  # [1, N, dim]

        # Text
        text_embeddings = self.encode_text(text)  # [1, num_prompt, dim]

        # Image-Text matching
        img_txt_matching = image_feat[:, 1:] @ text_embeddings.transpose(-1, -2)  # [1, N, num_prompt]
        img_txt_matching = rearrange(img_txt_matching, 'b (w h) c -> b c w h',
                                     w=W//self.patch_size, h=H//self.patch_size)  # [1, num_prompt, w, h]

        # Interpolate
        img_txt_matching = F.interpolate(img_txt_matching, size=(W, H), mode='bilinear')  # [1, num_prompt, W, H]

        # Heat Maps
        if normalize:
            img_txt_matching = self.min_max(img_txt_matching)
        return img_txt_matching

    def batched_forward(self, image: torch.Tensor, text: list, normalize: bool = True, return_ori: bool =False):
        """
        :param image: torch.Tensor [B, 3, H, W]
        :param text: list[list[]]
        :param normalize: bool - if True performs min-max normalization
        :param return_ori: bool - if True uses the features from the original visual encoder
        """
        L = len(text)
        cumm_idx = np.cumsum([len(t) for t in text]).tolist()
        B, _, W, H = image.shape
        assert B == L, f'Number of prompts L: {L} should be the same as number of images B: {B}.'

        # Image
        feat_gem, feat_ori = self.model.visual(image)
        image_feat = feat_ori if return_ori else feat_gem
        image_feat = F.normalize(image_feat, dim=-1)  # [B, N, dim]

        # Text
        flatten_text = [t for sub_text in text for t in sub_text]
        text_embeddings = self.encode_text(flatten_text)  # [B, num_prompt, dim]

        # Image-Text matching
        img_txt_matching = 100 * image_feat[:, 1:] @ text_embeddings.transpose(-1, -2)  # [B, N, num_prompt]
        img_txt_matching = rearrange(img_txt_matching, 'b (w h) c -> b c w h',
                                     w=W // self.patch_size, h=H // self.patch_size)  # [B, num_prompt, w, h]

        # Interpolate
        img_txt_matching = F.interpolate(img_txt_matching, size=(W, H), mode='bilinear')  # [B,num_prompt, W, H]

        # Heat Maps
        if normalize:
            img_txt_matching = self.min_max(img_txt_matching)  # [B,num_prompt, W, H]

        # unflatten
        img_txt_matching = torch.tensor_split(img_txt_matching, cumm_idx[:-1], dim=1)
        img_txt_matching = [itm[i] for i, itm in enumerate(img_txt_matching)]
        return img_txt_matching
