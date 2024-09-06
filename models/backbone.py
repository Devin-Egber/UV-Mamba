import torch
import torch.nn as nn
from torch.nn import ModuleList

from models.ossm import OSSM
from models.utils import nchw_to_nlc, nlc_to_nchw
from models.module import DropPath, Stem
from models.module import PatchEmbed
from DCNv4 import DCNv4


class MixFFN(nn.Module):

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


class DeformMixFFN(nn.Module):

    def __init__(self,
                 embed_dims,
                 feedforward_channels,
                 ffn_drop=0.,
                 dropout_layer=None):
        super().__init__()

        self.embed_dims = embed_dims
        self.feedforward_channels = feedforward_channels
        self.activate = nn.GELU()

        self.norm1 = nn.LayerNorm(embed_dims)
        self.norm2 = nn.LayerNorm(embed_dims)

        in_channels = embed_dims

        fc1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=feedforward_channels,
            kernel_size=1,
            stride=1)

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
            stride=1)

        self.dcnv4 = DCNv4(
                    channels=embed_dims,
                    kernel_size=3,
                    stride=1,
                    padding=(3 - 1) // 2,
                    group=2)

        drop = nn.Dropout(ffn_drop)
        layers = [fc1, pe_conv, self.activate, drop, fc2, drop]
        self.layers = nn.Sequential(*layers)
        self.dropout_layer = DropPath(
            dropout_layer['drop_prob']) if dropout_layer else torch.nn.Identity()

    def forward(self, x, hw_shape, identity=None):
        x = x + self.dropout_layer(self.norm1(self.dcnv4(x)))
        out = nlc_to_nchw(x, hw_shape)
        out = self.layers(out)
        out = nchw_to_nlc(out)
        out = x + self.dropout_layer(self.norm2(out))
        return out


class DeformMambaEncoderLayer(nn.Module):
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

        self.mamba_layer = nn.ModuleList()

        self.mamba_layer = OSSM(
            d_model=embed_dims,
            d_state=16,
            ssm_ratio=2.0,
            dt_rank="auto",
            d_conv=3,
            conv_bias=True,
            dropout=0,
            initialize="v0",
            forward_type="v2",
        )
        self.norm2 = nn.LayerNorm(embed_dims)

        self.deform_mix_ffn = DeformMixFFN(
            embed_dims=embed_dims,
            feedforward_channels=feedforward_channels,
            ffn_drop=drop_rate,
            dropout_layer=dict(type='DropPath', drop_prob=drop_path_rate))

        self.mix_ffn = MixFFN(
            embed_dims=embed_dims,
            feedforward_channels=feedforward_channels,
            ffn_drop=drop_rate,
            dropout_layer=dict(type='DropPath', drop_prob=drop_path_rate))

        self.proj_drop = nn.Dropout(proj_drop)
        self.dropout_layer = DropPath(
            dropout_layer['drop_prob']) if dropout_layer else torch.nn.Identity()

    def forward(self, x, hw_shape, identity=None):
        if identity is None:
            identity = x

        B = x.shape[0]
        x = self.norm1(x)
        x = self.deform_mix_ffn(x, hw_shape, identity=x)
        x = self.mamba_layer(x, hw_shape)
        x = self.norm2(x)
        x = self.mix_ffn(x, hw_shape, identity=x)
        x = identity + self.dropout_layer(self.proj_drop(x))
        return x


class DeformMixVisionMamba(nn.Module):
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

        dpr = [
            x.item()
            for x in torch.linspace(0, drop_path_rate, sum(num_layers))
        ]  # stochastic num_layer decay rule

        cur = 0

        self.stem = Stem(in_channels=in_channels, stem_hidden_dim=embed_dims*2, out_channels=embed_dims)

        self.layers = ModuleList()
        for i, num_layer in enumerate(num_layers):
            embed_dims_i = self.embed_dims * num_heads[i]
            patch_embed = PatchEmbed(
                in_channels=embed_dims,
                embed_dims=embed_dims_i,
                kernel_size=patch_sizes[i],
                stride=strides[i],
                padding=patch_sizes[i] // 2)

            layer = ModuleList([
                DeformMambaEncoderLayer(
                    embed_dims=embed_dims_i,
                    feedforward_channels=mlp_ratio * embed_dims_i,
                    drop_rate=drop_rate,
                    cur_index=cur+idx,
                    drop_path_rate=dpr[cur + idx]) for idx in range(num_layer)
            ])

            embed_dims = embed_dims_i
            norm = nn.LayerNorm(embed_dims_i)
            self.layers.append(ModuleList([patch_embed, layer, norm]))
            cur += num_layer

    def forward(self, x):
        outs = []
        x = self.stem(x)

        for i, layer in enumerate(self.layers):
            x, hw_shape = layer[0](x)
            for block in layer[1]:
                x = block(x, hw_shape)
            x = layer[2](x)
            x = nlc_to_nchw(x, hw_shape)
            if i in self.out_indices:
                outs.append(x)

        return outs


# ====================================
# 消融实验一：去掉DCNV4
# ====================================

class MambaEncoderLayer(nn.Module):
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

        self.mamba_layer = nn.ModuleList()

        self.mamba_layer = OSSM(
            d_model=embed_dims,
            d_state=16,
            ssm_ratio=2.0,
            dt_rank="auto",
            d_conv=3,
            conv_bias=True,
            dropout=0,
            initialize="v0",
            forward_type="v2",
        )
        self.norm2 = nn.LayerNorm(embed_dims)

        self.mix_ffn = MixFFN(
            embed_dims=embed_dims,
            feedforward_channels=feedforward_channels,
            ffn_drop=drop_rate,
            dropout_layer=dict(type='DropPath', drop_prob=drop_path_rate))

        self.proj_drop = nn.Dropout(proj_drop)
        self.dropout_layer = DropPath(
            dropout_layer['drop_prob']) if dropout_layer else torch.nn.Identity()

    def forward(self, x, hw_shape, identity=None):
        if identity is None:
            identity = x

        B = x.shape[0]
        x = self.norm1(x)
        x = self.mamba_layer(x, hw_shape)
        x = self.norm2(x)
        x = self.mix_ffn(x, hw_shape, identity=x)
        x = identity + self.dropout_layer(self.proj_drop(x))
        return x


class MixVisionMamba(nn.Module):
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

        dpr = [
            x.item()
            for x in torch.linspace(0, drop_path_rate, sum(num_layers))
        ]  # stochastic num_layer decay rule

        cur = 0

        self.stem = Stem(in_channels=in_channels, stem_hidden_dim=embed_dims*2, out_channels=embed_dims)

        self.layers = ModuleList()
        for i, num_layer in enumerate(num_layers):
            embed_dims_i = self.embed_dims * num_heads[i]
            patch_embed = PatchEmbed(
                in_channels=embed_dims,
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

            embed_dims = embed_dims_i
            norm = nn.LayerNorm(embed_dims_i)
            self.layers.append(ModuleList([patch_embed, layer, norm]))
            cur += num_layer

    def forward(self, x):
        outs = []
        x = self.stem(x)

        for i, layer in enumerate(self.layers):
            x, hw_shape = layer[0](x)
            for block in layer[1]:
                x = block(x, hw_shape)
            x = layer[2](x)
            x = nlc_to_nchw(x, hw_shape)
            if i in self.out_indices:
                outs.append(x)

        return outs

# ====================================
# 消融实验二: 去除SSM
# ====================================

class DeformEncoderLayer(nn.Module):
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

        self.norm2 = nn.LayerNorm(embed_dims)

        self.deform_mix_ffn = DeformMixFFN(
            embed_dims=embed_dims,
            feedforward_channels=feedforward_channels,
            ffn_drop=drop_rate,
            dropout_layer=dict(type='DropPath', drop_prob=drop_path_rate))

        self.proj_drop = nn.Dropout(proj_drop)
        self.dropout_layer = DropPath(
            dropout_layer['drop_prob']) if dropout_layer else torch.nn.Identity()

    def forward(self, x, hw_shape, identity=None):
        if identity is None:
            identity = x

        B = x.shape[0]
        x = self.norm1(x)
        x = self.deform_mix_ffn(x, hw_shape, identity=x)
        x = identity + self.dropout_layer(self.proj_drop(x))
        return x


class DeformMixVision(nn.Module):
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

        dpr = [
            x.item()
            for x in torch.linspace(0, drop_path_rate, sum(num_layers))
        ]  # stochastic num_layer decay rule

        cur = 0

        self.stem = Stem(in_channels=in_channels, stem_hidden_dim=embed_dims*2, out_channels=embed_dims)

        self.layers = ModuleList()
        for i, num_layer in enumerate(num_layers):
            embed_dims_i = self.embed_dims * num_heads[i]
            patch_embed = PatchEmbed(
                in_channels=embed_dims,
                embed_dims=embed_dims_i,
                kernel_size=patch_sizes[i],
                stride=strides[i],
                padding=patch_sizes[i] // 2)

            layer = ModuleList([
                DeformEncoderLayer(
                    embed_dims=embed_dims_i,
                    feedforward_channels=mlp_ratio * embed_dims_i,
                    drop_rate=drop_rate,
                    cur_index=cur+idx,
                    drop_path_rate=dpr[cur + idx]) for idx in range(num_layer)
            ])

            embed_dims = embed_dims_i
            norm = nn.LayerNorm(embed_dims_i)
            self.layers.append(ModuleList([patch_embed, layer, norm]))
            cur += num_layer

    def forward(self, x):
        outs = []
        x = self.stem(x)

        for i, layer in enumerate(self.layers):
            x, hw_shape = layer[0](x)
            for block in layer[1]:
                x = block(x, hw_shape)
            x = layer[2](x)
            x = nlc_to_nchw(x, hw_shape)
            if i in self.out_indices:
                outs.append(x)

        return outs


# ====================================
# 消融实验三：DCN与Mamba并行
# ====================================
class ParallelDeformMambaEncoderLayer(nn.Module):

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

        self.mamba_layer = nn.ModuleList()

        self.mamba_layer = OSSM(
            d_model=embed_dims,
            d_state=16,
            ssm_ratio=2.0,
            dt_rank="auto",
            d_conv=3,
            conv_bias=True,
            dropout=0,
            initialize="v0",
            forward_type="v2",
        )
        self.norm2 = nn.LayerNorm(embed_dims)

        self.deform_mix_ffn = DeformMixFFN(
            embed_dims=embed_dims,
            feedforward_channels=feedforward_channels,
            ffn_drop=drop_rate,
            dropout_layer=dict(type='DropPath', drop_prob=drop_path_rate))

        self.mix_ffn = MixFFN(
            embed_dims=embed_dims,
            feedforward_channels=feedforward_channels,
            ffn_drop=drop_rate,
            dropout_layer=dict(type='DropPath', drop_prob=drop_path_rate))

        self.proj_drop = nn.Dropout(proj_drop)
        self.dropout_layer = DropPath(
            dropout_layer['drop_prob']) if dropout_layer else torch.nn.Identity()

    def forward(self, x, hw_shape, identity=None):
        if identity is None:
            identity = x

        B = x.shape[0]
        x = self.norm1(x)
        x_deform = self.deform_mix_ffn(x, hw_shape, identity=x)
        x = self.mamba_layer(x, hw_shape)
        x = self.norm2(x)
        x_ssm = self.mix_ffn(x, hw_shape, identity=x)
        x = self.norm2(x_deform + x_ssm)
        x = identity + self.dropout_layer(self.proj_drop(x))
        return x


class ParallelDeformMixVisionMamba(nn.Module):

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

        dpr = [
            x.item()
            for x in torch.linspace(0, drop_path_rate, sum(num_layers))
        ]  # stochastic num_layer decay rule

        cur = 0

        self.stem = Stem(in_channels=in_channels, stem_hidden_dim=embed_dims*2, out_channels=embed_dims)

        self.layers = ModuleList()
        for i, num_layer in enumerate(num_layers):
            embed_dims_i = self.embed_dims * num_heads[i]
            patch_embed = PatchEmbed(
                in_channels=embed_dims,
                embed_dims=embed_dims_i,
                kernel_size=patch_sizes[i],
                stride=strides[i],
                padding=patch_sizes[i] // 2)

            layer = ModuleList([
                ParallelDeformMambaEncoderLayer(
                    embed_dims=embed_dims_i,
                    feedforward_channels=mlp_ratio * embed_dims_i,
                    drop_rate=drop_rate,
                    cur_index=cur+idx,
                    drop_path_rate=dpr[cur + idx]) for idx in range(num_layer)
            ])

            embed_dims = embed_dims_i
            norm = nn.LayerNorm(embed_dims_i)
            self.layers.append(ModuleList([patch_embed, layer, norm]))
            cur += num_layer

    def forward(self, x):
        outs = []
        x = self.stem(x)

        for i, layer in enumerate(self.layers):
            x, hw_shape = layer[0](x)
            for block in layer[1]:
                x = block(x, hw_shape)
            x = layer[2](x)
            x = nlc_to_nchw(x, hw_shape)
            if i in self.out_indices:
                outs.append(x)

        return outs


# ====================================
# 消融实验：SSM --> DCN
# ====================================

class MambaDeformEncoderLayer(nn.Module):

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

        self.mamba_layer = nn.ModuleList()

        self.mamba_layer = OSSM(
            d_model=embed_dims,
            d_state=16,
            ssm_ratio=2.0,
            dt_rank="auto",
            d_conv=3,
            conv_bias=True,
            dropout=0,
            initialize="v0",
            forward_type="v2",
        )
        self.norm2 = nn.LayerNorm(embed_dims)

        self.deform_mix_ffn = DeformMixFFN(
            embed_dims=embed_dims,
            feedforward_channels=feedforward_channels,
            ffn_drop=drop_rate,
            dropout_layer=dict(type='DropPath', drop_prob=drop_path_rate))

        self.mix_ffn = MixFFN(
            embed_dims=embed_dims,
            feedforward_channels=feedforward_channels,
            ffn_drop=drop_rate,
            dropout_layer=dict(type='DropPath', drop_prob=drop_path_rate))

        self.proj_drop = nn.Dropout(proj_drop)
        self.dropout_layer = DropPath(
            dropout_layer['drop_prob']) if dropout_layer else torch.nn.Identity()

    def forward(self, x, hw_shape, identity=None):
        if identity is None:
            identity = x

        B = x.shape[0]

        x = self.norm1(x)
        x = self.mamba_layer(x, hw_shape)
        x = self.norm2(x)
        x = self.mix_ffn(x, hw_shape, identity=x)
        x = self.deform_mix_ffn(x, hw_shape, identity=x)
        x = identity + self.dropout_layer(self.proj_drop(x))
        return x


class MambaMixVisionDeform(nn.Module):

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

        dpr = [
            x.item()
            for x in torch.linspace(0, drop_path_rate, sum(num_layers))
        ]  # stochastic num_layer decay rule

        cur = 0

        self.stem = Stem(in_channels=in_channels, stem_hidden_dim=embed_dims*2, out_channels=embed_dims)

        self.layers = ModuleList()
        for i, num_layer in enumerate(num_layers):
            embed_dims_i = self.embed_dims * num_heads[i]
            patch_embed = PatchEmbed(
                in_channels=embed_dims,
                embed_dims=embed_dims_i,
                kernel_size=patch_sizes[i],
                stride=strides[i],
                padding=patch_sizes[i] // 2)

            layer = ModuleList([
                MambaDeformEncoderLayer(
                    embed_dims=embed_dims_i,
                    feedforward_channels=mlp_ratio * embed_dims_i,
                    drop_rate=drop_rate,
                    cur_index=cur+idx,
                    drop_path_rate=dpr[cur + idx]) for idx in range(num_layer)
            ])

            embed_dims = embed_dims_i
            norm = nn.LayerNorm(embed_dims_i)
            self.layers.append(ModuleList([patch_embed, layer, norm]))
            cur += num_layer

    def forward(self, x):
        outs = []
        x = self.stem(x)

        for i, layer in enumerate(self.layers):
            x, hw_shape = layer[0](x)
            for block in layer[1]:
                x = block(x, hw_shape)
            x = layer[2](x)
            x = nlc_to_nchw(x, hw_shape)
            if i in self.out_indices:
                outs.append(x)

        return outs














