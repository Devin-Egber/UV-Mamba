import math
import torch
import torch.nn as nn
from einops import repeat
from models.utils import nchw_to_nlc, nlc_to_nchw
from functools import partial
from timm.models.layers import DropPath

try:
    import selective_scan_cuda_core
except Exception as e:
    print(f"WARNING: can not import selective_scan_cuda_core.")


class SelectiveScanCore(torch.autograd.Function):
    @staticmethod
    def forward(ctx, u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=False, nrows=1, backnrows=1, oflex=True):
        ctx.delta_softplus = delta_softplus
        out, x, *rest = selective_scan_cuda_core.fwd(u, delta, A, B, C, D, delta_bias, delta_softplus, 1)
        ctx.save_for_backward(u, delta, A, B, C, D, delta_bias, x)
        return out

    @staticmethod
    def backward(ctx, dout, *args):
        u, delta, A, B, C, D, delta_bias, x = ctx.saved_tensors
        if dout.stride(-1) != 1:
            dout = dout.contiguous()
        du, ddelta, dA, dB, dC, dD, ddelta_bias, *rest = selective_scan_cuda_core.bwd(
            u, delta, A, B, C, D, delta_bias, dout, x, ctx.delta_softplus, 1
        )
        return (du, ddelta, dA, dB, dC, dD, ddelta_bias, None, None, None, None)


def antidiagonal_gather(tensor):
    # 取出矩阵所有反斜向的元素并拼接
    B, C, H, W = tensor.size()
    shift = torch.arange(H, device=tensor.device).unsqueeze(1)  # 创建一个列向量[H, 1]
    index = (torch.arange(W, device=tensor.device) - shift) % W  # 利用广播创建索引矩阵[H, W]
    # 扩展索引以适应B和C维度
    expanded_index = index.unsqueeze(0).unsqueeze(0).expand(B, C, -1, -1)
    # 使用gather进行索引选择
    return tensor.gather(3, expanded_index).transpose(-1, -2).reshape(B, C, H * W)


def diagonal_gather(tensor):
    # 取出矩阵所有反斜向的元素并拼接
    B, C, H, W = tensor.size()
    shift = torch.arange(H, device=tensor.device).unsqueeze(1)  # 创建一个列向量[H, 1]
    index = (shift + torch.arange(W, device=tensor.device)) % W  # 利用广播创建索引矩阵[H, W]
    # 扩展索引以适应B和C维度
    expanded_index = index.unsqueeze(0).unsqueeze(0).expand(B, C, -1, -1)
    # 使用gather进行索引选择
    return tensor.gather(3, expanded_index).transpose(-1, -2).reshape(B, C, H * W)


def diagonal_scatter(tensor_flat, original_shape):
    # 把斜向元素拼接起来的一维向量还原为最初的矩阵形式
    B, C, H, W = original_shape
    shift = torch.arange(H, device=tensor_flat.device).unsqueeze(1)  # 创建一个列向量[H, 1]
    index = (shift + torch.arange(W, device=tensor_flat.device)) % W  # 利用广播创建索引矩阵[H, W]
    # 扩展索引以适应B和C维度
    expanded_index = index.unsqueeze(0).unsqueeze(0).expand(B, C, -1, -1)
    # 创建一个空的张量来存储反向散布的结果
    result_tensor = torch.zeros(B, C, H, W, device=tensor_flat.device, dtype=tensor_flat.dtype)
    # 将平铺的张量重新变形为[B, C, H, W]，考虑到需要使用transpose将H和W调换
    tensor_reshaped = tensor_flat.reshape(B, C, W, H).transpose(-1, -2)
    # 使用scatter_根据expanded_index将元素放回原位
    result_tensor.scatter_(3, expanded_index, tensor_reshaped)
    return result_tensor


def antidiagonal_scatter(tensor_flat, original_shape):
    # 把反斜向元素拼接起来的一维向量还原为最初的矩阵形式
    B, C, H, W = original_shape
    shift = torch.arange(H, device=tensor_flat.device).unsqueeze(1)  # 创建一个列向量[H, 1]
    index = (torch.arange(W, device=tensor_flat.device) - shift) % W  # 利用广播创建索引矩阵[H, W]
    expanded_index = index.unsqueeze(0).unsqueeze(0).expand(B, C, -1, -1)
    # 初始化一个与原始张量形状相同、元素全为0的张量
    result_tensor = torch.zeros(B, C, H, W, device=tensor_flat.device, dtype=tensor_flat.dtype)
    # 将平铺的张量重新变形为[B, C, W, H]，因为操作是沿最后一个维度收集的，需要调整形状并交换维度
    tensor_reshaped = tensor_flat.reshape(B, C, W, H).transpose(-1, -2)
    # 使用scatter_将元素根据索引放回原位
    result_tensor.scatter_(3, expanded_index, tensor_reshaped)
    return result_tensor


class CrossScan(torch.autograd.Function):
    # 将图像按照特定方向展平的地方，改变扫描方向可以在这里修改
    @staticmethod
    def forward(ctx, x: torch.Tensor):
        B, C, H, W = x.shape
        ctx.shape = (B, C, H, W)
        xs = x.new_empty((B, 8, C, H * W))
        # 添加横向和竖向的扫描
        xs[:, 0] = x.flatten(2, 3)
        xs[:, 1] = x.transpose(dim0=2, dim1=3).flatten(2, 3)
        xs[:, 2:4] = torch.flip(xs[:, 0:2], dims=[-1])

        # 提供斜向和反斜向的扫描
        xs[:, 4] = diagonal_gather(x)
        xs[:, 5] = antidiagonal_gather(x)
        xs[:, 6:8] = torch.flip(xs[:, 4:6], dims=[-1])

        return xs

    @staticmethod
    def backward(ctx, ys: torch.Tensor):
        # out: (b, k, d, l)
        B, C, H, W = ctx.shape
        L = H * W
        # 把横向和竖向的反向部分再反向回来，并和原来的横向和竖向相加
        y_rb = ys[:, 0:2] + ys[:, 2:4].flip(dims=[-1]).view(B, 2, -1, L)
        # 把竖向的部分转成横向，然后再相加,再转回最初是的矩阵形式
        y_rb = y_rb[:, 0] + y_rb[:, 1].view(B, -1, W, H).transpose(dim0=2, dim1=3).contiguous().view(B, -1, L)
        y_rb = y_rb.view(B, -1, H, W)

        # 把斜向和反斜向的反向部分再反向回来，并和原来的斜向和反斜向相加
        y_da = ys[:, 4:6] + ys[:, 6:8].flip(dims=[-1]).view(B, 2, -1, L)
        # 把斜向和反斜向的部分都转成原来的最初的矩阵形式，再相加
        y_da = diagonal_scatter(y_da[:, 0], (B, C, H, W)) + antidiagonal_scatter(y_da[:, 1], (B, C, H, W))

        y_res = y_rb + y_da
        return y_res


# 这里是mamba的具体内容，在这里增加扫描方向
def cross_selective_scan(
        x: torch.Tensor = None,
        x_proj_weight: torch.Tensor = None,
        x_proj_bias: torch.Tensor = None,
        dt_projs_weight: torch.Tensor = None,
        dt_projs_bias: torch.Tensor = None,
        A_logs: torch.Tensor = None,
        Ds: torch.Tensor = None,
        delta_softplus=True,
        # ==============================
        to_dtype=True,  # True: final out to dtype
        force_fp32=False,  # True: input fp32
        # ==============================
        nrows=-1,  # for SelectiveScanNRow; 0: auto; -1: disable;
        backnrows=-1,  # for SelectiveScanNRow; 0: auto; -1: disable;
        ssoflex=True,  # True: out fp32 in SSOflex; else, SSOflex is the same as SSCore
        # ==============================
        SelectiveScan=None,
        CrossScan=CrossScan):


    B, D, H, W = x.shape
    D, N = A_logs.shape
    K, D, R = dt_projs_weight.shape
    L = H * W

    if nrows == 0:
        if D % 4 == 0:
            nrows = 4
        elif D % 3 == 0:
            nrows = 3
        elif D % 2 == 0:
            nrows = 2
        else:
            nrows = 1

    if backnrows == 0:
        if D % 4 == 0:
            backnrows = 4
        elif D % 3 == 0:
            backnrows = 3
        elif D % 2 == 0:
            backnrows = 2
        else:
            backnrows = 1

    def selective_scan(u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=True):
        return SelectiveScan.apply(u, delta, A, B, C, D, delta_bias, delta_softplus, nrows, backnrows, ssoflex)

    xs = CrossScan.apply(x)

    x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs, x_proj_weight)
    if x_proj_bias is not None:
        x_dbl = x_dbl + x_proj_bias.view(1, K, -1, 1)
    dts, Bs, Cs = torch.split(x_dbl, [R, N, N], dim=2)
    dts = torch.einsum("b k r l, k d r -> b k d l", dts, dt_projs_weight)
    xs = xs.view(B, -1, L)
    dts = dts.contiguous().view(B, -1, L)
    As = -torch.exp(A_logs.to(torch.float))  # (k * c, d_state)
    Bs = Bs.contiguous()
    Cs = Cs.contiguous()
    Ds = Ds.to(torch.float)  # (K * c)
    delta_bias = dt_projs_bias.view(-1).to(torch.float)

    if force_fp32:
        xs = xs.to(torch.float)
        dts = dts.to(torch.float)
        Bs = Bs.to(torch.float)
        Cs = Cs.to(torch.float)

    # 将矩阵拆分成不同方向的序列，并进行扫描
    ys = selective_scan(
        xs, dts, As, Bs, Cs, Ds, delta_bias, delta_softplus
    ).view(B, K, -1, H, W)

    return ys

class OSSM(nn.Module):
    def __init__(
            self,
            # basic dims ===========
            d_model=96,
            d_state=16,
            ssm_ratio=2.0,
            dt_rank="auto",
            # dwconv ===============
            d_conv=3,  # < 2 means no conv
            conv_bias=True,
            # ======================
            dropout=0.0,
            bias=False,
            # dt init ==============
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            initialize="v0",
            forward_type="v2",
            **kwargs,
    ):
        super().__init__()
        d_inner = int(ssm_ratio * d_model)
        dt_rank = math.ceil(d_model / 16) if dt_rank == "auto" else dt_rank
        self.d_conv = d_conv

        self.norm1 = nn.LayerNorm(d_inner)
        self.norm2 = nn.LayerNorm(d_inner)


        # k_group 为扫描的方向
        k_group = 8 if forward_type not in ["debugscan_sharessm"] else 1
        # self.fuse = nn.Conv2d(
        #     in_channels=d_inner * k_group,
        #     out_channels=d_inner,
        #     kernel_size=1,
        #     stride=1)

        # in proj =======================================
        self.in_proj = nn.Linear(d_model, d_inner * 2, bias=bias)
        # self.in_proj = nn.Conv2d(
        #             in_channels=d_model,
        #             out_channels=d_inner * 2,
        #             kernel_size=1,
        #             stride=1)

        self.act = nn.SiLU()

        # conv =======================================
        self.conv2d = nn.Conv2d(
            in_channels=d_inner,
            out_channels=d_inner,
            groups=d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv) // 2,
        )

        # x proj ============================
        self.x_proj = [
            nn.Linear(d_inner, (dt_rank + d_state * 2), bias=False)
            for _ in range(k_group)
        ]
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))  # (K, N, inner)
        del self.x_proj

        # out proj =======================================
        self.out_proj = nn.Linear(d_inner, d_model, bias=bias)
        # self.out_proj = nn.Conv2d(
        #         in_channels=d_inner,
        #         out_channels=d_model,
        #         kernel_size=1,
        #         stride=1)

        self.dropout = nn.Dropout(dropout) if dropout > 0. else nn.Identity()

        if initialize in ["v0"]:
            # dt proj ============================
            self.dt_projs = [
                self.dt_init(dt_rank, d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor)
                for _ in range(k_group)
            ]
            self.dt_projs_weight = nn.Parameter(
                torch.stack([t.weight for t in self.dt_projs], dim=0))  # (K, inner, rank)
            self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0))  # (K, inner)
            del self.dt_projs

            # A, D =======================================
            self.A_logs = self.A_log_init(d_state, d_inner, copies=k_group, merge=True)  # (K * D, N)
            self.Ds = self.D_init(d_inner, copies=k_group, merge=True)  # (K * D)


    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4,
                **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        # dt_proj.bias._no_reinit = True

        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=-1, device=None, merge=True):
        # S4D real initialization
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        if copies > 0:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=-1, device=None, merge=True):
        # D "skip" parameter
        D = torch.ones(d_inner, device=device)
        if copies > 0:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True
        return D

    def forward(self, x, hw_shape):

        x = self.in_proj(x)
        x, z = x.chunk(2, dim=2)
        z = self.act(z)

        # x = x.permute(0, 3, 1, 2).contiguous()
        x = nlc_to_nchw(x, hw_shape)
        x = self.conv2d(x)  # (b, d, h, w)
        x = self.act(x)

        ys = cross_selective_scan(
            x, self.x_proj_weight, None, self.dt_projs_weight, self.dt_projs_bias,
            self.A_logs, self.Ds, delta_softplus=True,
            force_fp32=True,
            SelectiveScan=SelectiveScanCore,
        )

        B, K, D, H, W = ys.shape
        ys = ys.view(B, K, D, -1)

        # y_rb0 = ys[:, 0:2]
        # y_rb1 = ys[:, 2:4].flip(dims=[-1]).view(B, 2, D, -1)
        # y_rb0[:, 1].view(B, -1, H, W).transpose(dim0=2, dim1=3).contiguous().view(B, D, -1)
        # y_rb1[:, 1].view(B, -1, H, W).transpose(dim0=2, dim1=3).contiguous().view(B, D, -1)
        #
        # y_da0 = ys[:, 4:6]
        # y_da1 = ys[:, 6:8].flip(dims=[-1]).view(B, 2, D, -1)
        # diagonal_scatter(y_da0[:, 1], (B, D, H, W))
        # antidiagonal_scatter(y_da1[:, 1], (B, D, H, W))
        # y = torch.cat([y_rb0, y_rb1, y_da0, y_da1], dim=1).view(B, K*D, H, W)
        # y = self.fuse(y)

        y_rb = ys[:, 0:2] + ys[:, 2:4].flip(dims=[-1]).view(B, 2, D, -1)
        # 把竖向的部分转成横向，然后再相加,再转回最初是的矩阵形式
        y_rb = y_rb[:, 0] + y_rb[:, 1].view(B, -1, W, H).transpose(dim0=2, dim1=3).contiguous().view(B, D, -1)
        y_rb = y_rb.view(B, -1, H, W)
        # 把斜向和反斜向的反向部分再反向回来，并和原来的斜向和反斜向相加
        y_da = ys[:, 4:6] + ys[:, 6:8].flip(dims=[-1]).view(B, 2, D, -1)
        # 把斜向和反斜向的部分都转成原来的最初的矩阵形式，再相加
        y_da = diagonal_scatter(y_da[:, 0], (B, D, H, W)) + antidiagonal_scatter(y_da[:, 1], (B, D, H, W))

        y_res = y_rb + y_da
        y_res = nchw_to_nlc(y_res)

        y = z * y_res
        out = self.dropout(self.out_proj(y))
        return out


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.,channels_first=False):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        Linear = partial(nn.Conv2d, kernel_size=1, padding=0) if channels_first else nn.Linear
        self.fc1 = Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class OSSBlock(nn.Module):
    def __init__(
            self,
            hidden_dim: int = 0,
            drop_path: float = 0,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            # =============================
            ssm_d_state: int = 16,
            ssm_ratio=2.0,
            ssm_dt_rank = "auto",
            ssm_act_layer=nn.SiLU,
            ssm_conv: int = 3,
            ssm_conv_bias=True,
            ssm_drop_rate: float = 0,
            ssm_init="v0",
            forward_type="v2",
            # =============================
            mlp_ratio=4.0,
            mlp_act_layer=nn.GELU,
            mlp_drop_rate: float = 0.0,
            # =============================
            use_checkpoint: bool = False,
            post_norm: bool = False,
            **kwargs,
    ):
        super().__init__()
        self.ssm_branch = ssm_ratio > 0
        self.mlp_branch = mlp_ratio > 0
        self.use_checkpoint = use_checkpoint
        self.post_norm = post_norm

        if self.ssm_branch:
            self.norm = norm_layer(hidden_dim)
            self.op = OSSM(
                d_model=hidden_dim,
                d_state=ssm_d_state,
                ssm_ratio=ssm_ratio,
                dt_rank=ssm_dt_rank,
                act_layer=ssm_act_layer,
                d_conv=ssm_conv,
                conv_bias=ssm_conv_bias,
                # ==========================
                dropout=ssm_drop_rate,
                initialize=ssm_init,
                # ==========================
                forward_type=forward_type,
            )

        self.drop_path = DropPath(drop_path)

        if self.mlp_branch:
            self.norm2 = norm_layer(hidden_dim)
            mlp_hidden_dim = int(hidden_dim * mlp_ratio)
            self.mlp = Mlp(in_features=hidden_dim, hidden_features=mlp_hidden_dim, act_layer=mlp_act_layer,
                           drop=mlp_drop_rate, channels_first=False)

    def forward(self, input: torch.Tensor):
        if self.ssm_branch:
            if self.post_norm:
                x = input + self.drop_path(self.norm(self.op(input)))
            else:
                x = input + self.drop_path(self.op(self.norm(input)))
        if self.mlp_branch:
            if self.post_norm:
                x = x + self.drop_path(self.norm2(self.mlp(x)))  # FFN
            else:
                x = x + self.drop_path(self.mlp(self.norm2(x)))  # FFN
        return x
