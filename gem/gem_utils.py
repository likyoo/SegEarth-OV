from typing import Optional, List
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from open_clip.transformer import _expand_token, to_2tuple



def resample_abs_pos_embed(
        posemb,
        new_size: List[int],
        old_size: Optional[List[int]] = None,
        num_prefix_tokens: int = 1,
        interpolation: str = 'bicubic',
        antialias: bool = True
):
    # sort out sizes, assume square if old size not provided
    new_size = to_2tuple(new_size)
    new_ntok = new_size[0] * new_size[1]
    if not old_size:
        old_size = int(math.sqrt(posemb.shape[1] - num_prefix_tokens))
    old_size = to_2tuple(old_size)
    if new_size == old_size:  # might not both be same container type
        return posemb

    if num_prefix_tokens:
        posemb_prefix, posemb = posemb[:, :num_prefix_tokens], posemb[:, num_prefix_tokens:]
    else:
        posemb_prefix, posemb = None, posemb

    # do the interpolation
    posemb = posemb.reshape(1, old_size[0], old_size[1], -1).permute(0, 3, 1, 2)
    posemb = F.interpolate(posemb, size=new_size, mode=interpolation, antialias=antialias)
    posemb = posemb.permute(0, 2, 3, 1).reshape(1, new_ntok, -1)


    # add back extra (class, etc) prefix tokens
    if posemb_prefix is not None:
        posemb = torch.cat([posemb_prefix, posemb], dim=1)
    return posemb

class SelfSelfAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., ss_attn_iter=1,
                 ss_attn_temp=None):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.ss_attn_iter = ss_attn_iter
        self.ss_attn_temp = ss_attn_temp

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, attn_bias=None, prev_attn=None):
        x = x.transpose(0, 1)
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        self.v_values = v
        # original self-attention for the original path
        attn_ori_return = (q @ k.transpose(-2, -1)) * self.scale
        attn_ori = attn_ori_return.softmax(dim=-1)
        attn_ori = self.attn_drop(attn_ori)

        x_ori = (attn_ori @ v).transpose(1, 2).reshape(B, N, C)
        x_ori = self.proj_drop(self.proj(x_ori))

        # GEM
        xs1 = v
        xs2 = k
        xs3 = q

        if self.ss_attn_temp is None:
            pre_norm = torch.norm(x, dim=-1).mean(dim=-1, keepdim=True).unsqueeze(1).unsqueeze(-1)
            inv_temp = pre_norm * self.scale
        else:
            inv_temp = self.ss_attn_temp

        for it in range(self.ss_attn_iter):
            xs1 = F.normalize(xs1, dim=-1)
            xs2 = F.normalize(xs2, dim=-1)
            xs3 = F.normalize(xs3, dim=-1)

            attn_return1 = (xs1 @ xs1.transpose(-2, -1)) * inv_temp
            attn_return2 = (xs2 @ xs2.transpose(-2, -1)) * inv_temp
            attn_return3 = (xs3 @ xs3.transpose(-2, -1)) * inv_temp

            attn1 = (attn_return1).softmax(dim=-1)
            attn2 = (attn_return2).softmax(dim=-1)
            attn3 = (attn_return3).softmax(dim=-1)

            xs1 = attn1 @ xs1
            xs2 = attn2 @ xs2
            xs3 = attn3 @ xs3

        # Assigment to V
        xs1 = F.normalize(xs1, dim=-1)
        xs2 = F.normalize(xs2, dim=-1)
        xs3 = F.normalize(xs3, dim=-1)

        attn_return1 = (xs1 @ xs1.transpose(-2, -1)) * inv_temp
        attn_return2 = (xs2 @ xs2.transpose(-2, -1)) * inv_temp
        attn_return3 = (xs3 @ xs3.transpose(-2, -1)) * inv_temp

        attn1 = (attn_return1).softmax(dim=-1)
        attn2 = (attn_return2).softmax(dim=-1)
        attn3 = (attn_return3).softmax(dim=-1)

        xs1 = attn1 @ v
        xs2 = attn2 @ v
        xs3 = attn3 @ v
        xs = (xs1 + xs2 + xs3) / 3

        x = xs.transpose(1, 2).reshape(B, N, C)
        x = self.proj_drop(self.proj(x))

        return [x.transpose(0, 1), x_ori.transpose(0, 1)]


class GEMResidualBlock(nn.Module):
    def __init__(self, res_block, ignore_residual):
        super(GEMResidualBlock, self).__init__()
        self.res_block = res_block
        self.ignore_residual = ignore_residual

    def forward(self,
                q_x: torch.Tensor,
                k_x: Optional[torch.Tensor] = None,
                v_x: Optional[torch.Tensor] = None,
                attn_mask: Optional[torch.Tensor] = None,
                ):
        if isinstance(q_x, list):
            x_gem, q_x = q_x
        else:
            x_gem = q_x

        x_gem_res, x_ori_res = self.res_block.attn(x=self.res_block.ln_1(q_x))
        x_gem_res, x_ori_res = self.res_block.ls_1(x_gem_res), self.res_block.ls_1(x_ori_res)
        # Original
        x_ori = q_x + x_ori_res
        x_ori = x_ori + self.res_block.ls_2(self.res_block.mlp(self.res_block.ln_2(x_ori)))
        # GEM
        if self.ignore_residual:
            x_gem = x_gem_res
        else:
            x_gem = x_gem + x_gem_res
        return [x_gem, x_ori]

class GEMViT(nn.Module):
    def __init__(self, vit):
        self.vit = vit

def modified_vit_forward(self, x: torch.Tensor):
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        grid_h, grid_w = x.shape[2:]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]

        # class embeddings and positional embeddings
        x = torch.cat([_expand_token(self.class_embedding, x.shape[0]).to(x.dtype), x], dim=1)
        # shape = [*, grid ** 2 + 1, width]

        if x.shape[1] != self.positional_embedding.shape[1]:
            pos_emb = resample_abs_pos_embed(self.positional_embedding.unsqueeze(0),
                                             new_size=[grid_h, grid_w],
                                             # old_size=list(self.grid_size),
                                             num_prefix_tokens=1,
                                             interpolation='bicubic',
                                             antialias=True)

        else:
            pos_emb = self.positional_embedding

        x = x + pos_emb.to(x.dtype)
        # x = x + self.positional_embedding.to(x.dtype)

        x = self.patch_dropout(x)
        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x_gem, x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x_gem = x_gem.permute(1, 0, 2)  # LND -> NLD

        # Apply proj
        x = self.ln_post(x)
        x_gem = self.ln_post(x_gem)
        if self.proj is not None:
            x = x @ self.proj
            x_gem = x_gem @ self.proj

        # return [x_gem, x]
        return x_gem[:, 1:, :]
