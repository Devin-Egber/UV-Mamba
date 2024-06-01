import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import ModuleList
import math
import numpy as np
from einops import rearrange, repeat, einsum
from mamba_ssm import Mamba

from models.uvmamba.utils import nchw_to_nlc, nlc_to_nchw
from models.uvmamba.utils import trunc_normal_init, constant_init, normal_init
from models.uvmamba.module import DropPath
from models.uvmamba.module import PatchEmbed, PatchEmbedDeform, PatchEmbedModulatedDeform
from models.uvmamba.ossm import OSSM

class MixFFN(nn.Module):
    """An implementation of MixFFN of Segformer.

    The differences between MixFFN & FFN:
        1. Use 1X1 Conv to replace Linear layer.
        2. Introduce 3X3 Conv to encode positional information.
    Args:
        embed_dims (int): The feature dimension. Same as
            `MultiheadAttention`. Defaults: 256.
        feedforward_channels (int): The hidden dimension of FFNs.
            Defaults: 1024.
        ffn_drop (float, optional): Probability of an element to be
            zeroed in FFN. Default 0.0.
        dropout_layer (obj:`ConfigDict`): The dropout_layer used
            when adding the shortcut.
    """

    def __init__(self,
                 embed_dims,
                 feedforward_channels,
                 ffn_drop=0.,
                 dropout_layer=None):
        super().__init__()

        self.embed_dims = embed_dims
        self.feedforward_channels = feedforward_channels
        self.activate = nn.GELU()

        in_channels = embed_dims

        fc1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=feedforward_channels,
            kernel_size=1,
            stride=1,
            bias=True)

        # 3x3 depth wise conv to provide positional encode information
        pe_conv = nn.Conv2d(
            in_channels=feedforward_channels,
            out_channels=feedforward_channels,
            kernel_size=3,
            stride=1,
            padding=(3 - 1) // 2,
            bias=True,
            groups=feedforward_channels)

        fc2 = nn.Conv2d(
            in_channels=feedforward_channels,
            out_channels=in_channels,
            kernel_size=1,
            stride=1,
            bias=True)

        drop = nn.Dropout(ffn_drop)
        layers = [fc1, pe_conv, self.activate, drop, fc2, drop]
        self.layers = nn.Sequential(*layers)
        self.dropout_layer = DropPath(
            dropout_layer['drop_prob']) if dropout_layer else torch.nn.Identity()

    def forward(self, x, hw_shape, identity=None):
        out = nlc_to_nchw(x, hw_shape)
        out = self.layers(out)
        out = nchw_to_nlc(out)
        if identity is None:
            identity = x
        return identity + self.dropout_layer(out)


class MambaEncoderLayer(nn.Module):
    """
    Implements one encoder layer compose of mamba and DeformMixFFN in UVMamba.

    Args:
        embed_dims (int): The feature dimension.
        num_heads (int): Parallel attention heads.
        feedforward_channels (int): The hidden dimension for FFNs.
        drop_rate (float): Probability of an element to be zeroed.
            after the feed forward layer. Default 0.0.
        attn_drop_rate (float): The drop out rate for attention layer.
            Default 0.0.
        drop_path_rate (float): stochastic depth rate. Default 0.0
        batch_first (bool): Key, Query and Value are shape of
            (batch, n, embed_dim)
            or (n, batch, embed_dim). Default: False.
        init_cfg (dict, optional): Initialization config dict.
            Default:None.
        sr_ratio (int): The ratio of spatial reduction of Efficient Multi-head
            Attention of Segformer. Default: 1.
        with_cp (bool): Use checkpoint or not. Using checkpoint will save
            some memory while slowing down the training speed. Default: False.
    """

    def __init__(self,
                 embed_dims,
                 feedforward_channels,
                 drop_rate=0.,
                 drop_path_rate=0.,
                 proj_drop=0.,
                 cur_index=None,
                 dropout_layer=dict(type='Dropout', drop_prob=0.),
                 depth=2):
        super().__init__()

        self.norm1 = nn.LayerNorm(embed_dims)

        self.mamba_layer = OSSM(
                        d_model=embed_dims,
                        d_state=16,
                        ssm_ratio=2.0,
                        dt_rank="auto",
                        # act_layer=nn.SiLU,
                        # ==========================
                        d_conv=3,
                        conv_bias=True,
                        # ==========================
                        dropout=0,
                        # bias=False,
                        # ==========================
                        # dt_min=0.001,
                        # dt_max=0.1,
                        # dt_init="random",
                        # dt_scale="random",
                        # dt_init_floor=1e-4,
                        initialize="v0",
                        # ==========================
                        forward_type="v2",
                        )

        self.norm2 = nn.LayerNorm(embed_dims)

        self.ffn = MixFFN(
            embed_dims=embed_dims,
            feedforward_channels=feedforward_channels,
            ffn_drop=drop_rate,
            dropout_layer=dict(type='DropPath', drop_prob=drop_path_rate))

        self.proj_drop = nn.Dropout(proj_drop)
        self.dropout_layer = DropPath(
            dropout_layer['drop_prob']) if dropout_layer else torch.nn.Identity()

        gate_out_dim = 3
        self.gate_layers = nn.Sequential(
                nn.Linear(gate_out_dim * embed_dims, gate_out_dim, bias=False),
                nn.Softmax(dim=-1))

    def forward(self, x, hw_shape, identity=None):
        if identity is None:
            identity = x

        B = x.shape[0]
        x = nlc_to_nchw(x, hw_shape)
        # x = x.permute(0, 2, 3, 1).contiguous()
        x = self.mamba_layer(x)
        # x = x.flatten(2, 3).permute(0, 2, 1)
        x = nchw_to_nlc(x)
        x = identity + self.dropout_layer(self.proj_drop(x))

        x = self.ffn(self.norm2(x), hw_shape, identity=x)
        return x


class MixVisionSamba(nn.Module):
    """The backbone of uvmamba.
    Args:
        in_channels (int): Number of input channels. Default: 3.
        embed_dims (int): Embedding dimension. Default: 768.
        num_stags (int): The num of stages. Default: 4.
        num_layers (Sequence[int]): The layer number of each transformer encode
            layer. Default: [3, 4, 6, 3].
        num_heads (Sequence[int]): The attention heads of each transformer
            encode layer. Default: [1, 2, 4, 8].
        patch_sizes (Sequence[int]): The patch_size of each overlapped patch
            embedding. Default: [7, 3, 3, 3].
        strides (Sequence[int]): The stride of each overlapped patch embedding.
            Default: [4, 2, 2, 2].
        sr_ratios (Sequence[int]): The spatial reduction rate of each
            transformer encode layer. Default: [8, 4, 2, 1].
        out_indices (Sequence[int] | int): Output from which stages.
            Default: (0, 1, 2, 3).
        mlp_ratio (int): ratio of mlp hidden dim to embedding dim.
            Default: 4.
        qkv_bias (bool): Enable bias for qkv if True. Default: True.
        drop_rate (float): Probability of an element to be zeroed.
            Default 0.0
        attn_drop_rate (float): The drop out rate for attention layer.
            Default 0.0
        drop_path_rate (float): stochastic depth rate. Default 0.0
    """

    def __init__(self,
                 in_channels=3,
                 embed_dims=64,
                 num_stages=4,
                 num_layers=[3, 4, 6, 3],
                 num_heads=[1, 2, 4, 8],
                 patch_sizes=[7, 3, 3, 3],
                 strides=[4, 2, 2, 2],
                 sr_ratios=[8, 4, 2, 1],
                 out_indices=(0, 1, 2, 3),
                 mlp_ratio=4,
                 drop_rate=0.,
                 drop_path_rate=0.):

        super().__init__()

        self.embed_dims = embed_dims
        self.num_stages = num_stages
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.patch_sizes = patch_sizes
        self.strides = strides
        self.sr_ratios = sr_ratios

        assert num_stages == len(num_layers) == len(num_heads) \
               == len(patch_sizes) == len(strides) == len(sr_ratios)

        self.out_indices = out_indices
        assert max(out_indices) < self.num_stages

        # transformer encoder
        dpr = [
            x.item()
            for x in torch.linspace(0, drop_path_rate, sum(num_layers))
        ]  # stochastic num_layer decay rule

        cur = 0
        self.layers = ModuleList()
        for i, num_layer in enumerate(num_layers):
            embed_dims_i = embed_dims * num_heads[i]
            patch_embed = PatchEmbed(
                in_channels=in_channels,
                embed_dims=embed_dims_i,
                kernel_size=patch_sizes[i],
                stride=strides[i],
                padding=patch_sizes[i] // 2)

            layer = ModuleList([
                MambaEncoderLayer(
                    embed_dims=embed_dims_i,
                    feedforward_channels=mlp_ratio * embed_dims_i,
                    drop_rate=drop_rate,
                    cur_index=cur+idx,
                    drop_path_rate=dpr[cur + idx]) for idx in range(num_layer)
            ])

            in_channels = embed_dims_i
            norm = nn.LayerNorm(embed_dims_i)
            self.layers.append(ModuleList([patch_embed, layer, norm]))
            cur += num_layer

    def forward(self, x):
        outs = []

        for i, layer in enumerate(self.layers):
            x, hw_shape = layer[0](x)
            for block in layer[1]:
                x = block(x, hw_shape)
            x = layer[2](x)
            x = nlc_to_nchw(x, hw_shape)
            if i in self.out_indices:
                outs.append(x)

        return outs
