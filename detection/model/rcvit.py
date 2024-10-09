"""
Code for Additive Vision Transformer
"""

import torch
import torch.nn as nn
from torch.cuda.amp import autocast

import numpy as np
from einops import rearrange, repeat
import itertools
import os
import copy

from timm.models.layers import DropPath, trunc_normal_, to_2tuple
from timm.models.registry import register_model

from mmdet.utils import get_root_logger
from mmdet.models import BACKBONES
from torch.nn.modules.batchnorm import _BatchNorm
from mmcv.runner import _load_checkpoint

# ======================================================================================================================
def stem(in_chs, out_chs):
    return nn.Sequential(
        nn.Conv2d(in_chs, out_chs // 2, kernel_size=3, stride=2, padding=1),
        nn.BatchNorm2d(out_chs // 2),
        nn.ReLU(),
        nn.Conv2d(out_chs // 2, out_chs, kernel_size=3, stride=2, padding=1),
        nn.BatchNorm2d(out_chs),
        nn.ReLU(), )

class Embedding(nn.Module):
    """
    Patch Embedding that is implemented by a layer of conv.
    Input: tensor in shape [B, C, H, W]
    Output: tensor in shape [B, C, H/stride, W/stride]
    """

    def __init__(self, patch_size=16, stride=16, padding=0,
                 in_chans=3, embed_dim=768, norm_layer=nn.BatchNorm2d):
        super().__init__()
        patch_size = to_2tuple(patch_size)
        stride = to_2tuple(stride)
        padding = to_2tuple(padding)
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size,
                              stride=stride, padding=padding)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        x = self.proj(x)
        x = self.norm(x)
        return x

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class SpatialOperation(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1, groups=dim),
            nn.BatchNorm2d(dim),
            nn.ReLU(True),
            nn.Conv2d(dim, 1, 1, 1, 0, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.block(x)

class ChannelOperation(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(dim, dim, 1, 1, 0, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.block(x)

class LocalIntegration(nn.Module):
    """
    """
    def __init__(self, dim, ratio=1, act_layer=nn.ReLU, norm_layer=nn.GELU):
        super().__init__()
        mid_dim = round(ratio * dim)
        self.network = nn.Sequential(
            nn.Conv2d(dim, mid_dim, 1, 1, 0),
            norm_layer(mid_dim),
            nn.Conv2d(mid_dim, mid_dim, 3, 1, 1, groups=mid_dim),
            act_layer(),
            nn.Conv2d(mid_dim, dim, 1, 1, 0),
        )

    def forward(self, x):
        return self.network(x)


class AdditiveTokenMixer(nn.Module):
    """
    改变了proj函数的输入，不对q+k卷积，而是对融合之后的结果proj
    """
    def __init__(self, dim=512, attn_bias=False, proj_drop=0.):
        super().__init__()
        self.qkv = nn.Conv2d(dim, 3 * dim, 1, stride=1, padding=0, bias=attn_bias)
        self.oper_q = nn.Sequential(
            SpatialOperation(dim),
            ChannelOperation(dim),
        )
        self.oper_k = nn.Sequential(
            SpatialOperation(dim),
            ChannelOperation(dim),
        )
        self.dwc = nn.Conv2d(dim, dim, 3, 1, 1, groups=dim)

        self.proj = nn.Conv2d(dim, dim, 3, 1, 1, groups=dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        q, k, v = self.qkv(x).chunk(3, dim=1)
        q = self.oper_q(q)
        k = self.oper_k(k)
        out = self.proj(self.dwc(q + k) * v)
        out = self.proj_drop(out)
        return out



class AdditiveBlock(nn.Module):
    """
    """
    def __init__(self, dim, mlp_ratio=4., attn_bias=False, drop=0., drop_path=0.,
                 act_layer=nn.ReLU, norm_layer=nn.GELU):
        super().__init__()
        self.local_perception = LocalIntegration(dim, ratio=1, act_layer=act_layer, norm_layer=norm_layer)
        self.norm1 = norm_layer(dim)
        self.attn = AdditiveTokenMixer(dim, attn_bias=attn_bias, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.local_perception(x)
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

def Stage(dim, index, layers, mlp_ratio=4., act_layer=nn.GELU, attn_bias=False, drop=0., drop_path_rate=0.):
    """
    """
    blocks = []
    for block_idx in range(layers[index]):
        block_dpr = drop_path_rate * (block_idx + sum(layers[:index])) / (sum(layers) - 1)

        blocks.append(
            AdditiveBlock(
                dim, mlp_ratio=mlp_ratio, attn_bias=attn_bias, drop=drop, drop_path=block_dpr,
                act_layer=act_layer, norm_layer=nn.BatchNorm2d)
        )
    blocks = nn.Sequential(*blocks)
    return blocks

class RCViT(nn.Module):
    def __init__(self, layers, embed_dims, mlp_ratios=4, downsamples=[True, True, True, True], norm_layer=nn.BatchNorm2d, attn_bias=False,
                 act_layer=nn.GELU, num_classes=1000, drop_rate=0., drop_path_rate=0., fork_feat=False,
                 init_cfg=None, pretrained=None, distillation=True, **kwargs):
        super().__init__()

        if not fork_feat:
            self.num_classes = num_classes
        self.fork_feat = fork_feat

        self.patch_embed = stem(3, embed_dims[0])

        network = []
        for i in range(len(layers)):
            stage = Stage(embed_dims[i], i, layers, mlp_ratio=mlp_ratios, act_layer=act_layer,
                          attn_bias=attn_bias, drop=drop_rate, drop_path_rate=drop_path_rate)

            network.append(stage)
            if i >= len(layers) - 1:
                break
            if downsamples[i] or embed_dims[i] != embed_dims[i + 1]:
                # downsampling between two stages
                network.append(
                    Embedding(
                        patch_size=3, stride=2, padding=1, in_chans=embed_dims[i],
                        embed_dim=embed_dims[i+1], norm_layer=nn.BatchNorm2d)
                )

        self.network = nn.ModuleList(network)

        if self.fork_feat:
            # add a norm layer for each output
            self.out_indices = [0, 2, 4, 6]
            for i_emb, i_layer in enumerate(self.out_indices):
                if i_emb == 0 and os.environ.get('FORK_LAST3', None):
                    layer = nn.Identity()
                else:
                    layer = norm_layer(embed_dims[i_emb])
                layer_name = f'norm{i_layer}'
                self.add_module(layer_name, layer)
        else:
            # Classifier head
            self.norm = norm_layer(embed_dims[-1])
            self.head = nn.Linear(
                embed_dims[-1], num_classes) if num_classes > 0 \
                else nn.Identity()
            self.dist = distillation
            if self.dist:
                self.dist_head = nn.Linear(
                    embed_dims[-1], num_classes) if num_classes > 0 \
                    else nn.Identity()

        self.apply(self.cls_init_weights)

        self.init_cfg = copy.deepcopy(init_cfg)
        # load pre-trained model
        if self.fork_feat and (
                self.init_cfg is not None or pretrained is not None):
            self.init_weights()

    # init for classification
    def cls_init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    # init for mmdetection or mmsegmentation by loading
    # imagenet pre-trained weights
    def init_weights(self, pretrained=None):
        logger = get_root_logger()
        if self.init_cfg is None and pretrained is None:
            logger.warn(f'No pre-trained weights for '
                        f'{self.__class__.__name__}, '
                        f'training start from scratch')
            pass
        else:
            assert 'checkpoint' in self.init_cfg, f'Only support ' \
                                                  f'specify `Pretrained` in ' \
                                                  f'`init_cfg` in ' \
                                                  f'{self.__class__.__name__} '
            if self.init_cfg is not None:
                ckpt_path = self.init_cfg['checkpoint']
            elif pretrained is not None:
                ckpt_path = pretrained

            ckpt = _load_checkpoint(
                ckpt_path, logger=logger, map_location='cpu')
            if 'state_dict' in ckpt:
                _state_dict = ckpt['state_dict']
            elif 'model' in ckpt:
                _state_dict = ckpt['model']
            else:
                _state_dict = ckpt

            state_dict = _state_dict
            missing_keys, unexpected_keys = \
                self.load_state_dict(state_dict, False)
    
    @torch.no_grad()
    def train(self, mode=True):
        super().train(mode)
        for m in self.modules():
            # trick: eval have effect on BatchNorm only
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def forward_tokens(self, x):
        outs = []
        for idx, block in enumerate(self.network):
            x = block(x)
            if self.fork_feat and idx in self.out_indices:
                norm_layer = getattr(self, f'norm{idx}')
                x_out = norm_layer(x)
                outs.append(x_out)
        if self.fork_feat:
            return outs
        return x

    def forward(self, x):
        x = self.patch_embed(x)
        x = self.forward_tokens(x)
        if self.fork_feat:
            # otuput features of four stages for dense prediction
            return x
        x = self.norm(x)
        if self.dist:
            cls_out = self.head(x.flatten(2).mean(-1)), self.dist_head(x.flatten(2).mean(-1))
            if not self.training:
                cls_out = (cls_out[0] + cls_out[1]) / 2
        else:
            cls_out = self.head(x.flatten(2).mean(-1))
        # for image classification
        return cls_out

# ======================================================================================================================

@BACKBONES.register_module()
def rcvit_xs(**kwargs):
    model = RCViT(
        layers=[2, 2, 4, 2], embed_dims=[48, 56, 112, 220], mlp_ratios=4, downsamples=[True, True, True, True],
        norm_layer=nn.BatchNorm2d, attn_bias=False, act_layer=nn.GELU, drop_rate=0., **kwargs)
    return model

@BACKBONES.register_module()
def rcvit_s(**kwargs):
    model = RCViT(
        layers=[3, 3, 6, 3], embed_dims=[48, 64, 128, 256], mlp_ratios=4, downsamples=[True, True, True, True],
        norm_layer=nn.BatchNorm2d, attn_bias=False, act_layer=nn.GELU, drop_rate=0., **kwargs)
    return model

@BACKBONES.register_module()
def rcvit_m(**kwargs):
    model = RCViT(
        layers=[3, 3, 6, 3], embed_dims=[64, 96, 192, 384], mlp_ratios=4, downsamples=[True, True, True, True],
        norm_layer=nn.BatchNorm2d, attn_bias=False, act_layer=nn.GELU, drop_rate=0., **kwargs)
    return model

@BACKBONES.register_module()
def rcvit_t(**kwargs):
    model = RCViT(
        layers=[3, 3, 6, 3], embed_dims=[96, 128, 256, 512], mlp_ratios=4, downsamples=[True, True, True, True],
        norm_layer=nn.BatchNorm2d, attn_bias=False, act_layer=nn.GELU, drop_rate=0., **kwargs)
    return model

# ======================================================================================================================
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    from thop import profile, clever_format

    net = rcvit_xs()
    x = torch.rand((1, 3, 224, 224))
    out = net(x)

    print('Net Params: {:d}'.format(int(count_parameters(net))))

    flops, params = profile(net, inputs=(x,))
    flops, params = clever_format([flops, params], "%.3f")
    print('Params: ', params, '\tFlops: ', flops)
