import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
import warnings
from timm.models.layers import DropPath, Mlp, trunc_normal_
from fvcore.nn.distributed import differentiable_all_reduce
import fvcore.nn.weight_init as weight_init
import torch.distributed as dist
import math


class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x


class MLP(nn.Module):
    def __init__(self, input_dim = 256, output_dim = 256) -> None:
        super().__init__()
        self.proj = nn.Linear(input_dim , output_dim)
    def forward(self, x:torch.Tensor):
        x = x.flatten(2).transpose(1, 2)
        x = self.proj(x)
        return x.permute(0, 2, 1)


class CrossAttentionBlock(nn.Module):
    def __init__(self, dim, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        
        self.q = nn.Conv2d(dim, dim, 1)
        self.k = nn.Conv2d(dim, dim, 1)
        self.v = nn.Conv2d(dim, dim, 1)
        self.proj = nn.Conv2d(dim, dim, 1)
        
        self.norm_q = LayerNorm(dim)
        self.norm_kv = LayerNorm(dim)

    def forward(self, x1, x2):
        B, C, H, W = x1.shape
        
        x1_norm = self.norm_q(x1)
        x2_norm = self.norm_kv(x2)

        q = self.q(x1_norm).flatten(2).transpose(1, 2)
        k = self.k(x2_norm).flatten(2).transpose(1, 2)
        v = self.v(x2_norm).flatten(2).transpose(1, 2)
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        
        x = (attn @ v).transpose(1, 2).reshape(B, C, H, W)
        x = self.proj(x)
        return x + x1


class SelfAttentionMLPHead(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=4, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim)
        )

    def forward(self, x):
        B, C, H, W = x.shape
        x_flat = x.flatten(2).transpose(1, 2)
        
        # Self Attention
        x2 = self.norm1(x_flat)
        x2, _ = self.attn(x2, x2, x2)
        x_flat = x_flat + x2
        
        # MLP
        x2 = self.norm2(x_flat)
        x2 = self.mlp(x2)
        x_flat = x_flat + x2
        
        return x_flat.transpose(1, 2).reshape(B, C, H, W)


class ArtifactPredictionHead(nn.Module):
    def __init__(self, 
                feature_channels : list,
                embed_dim = 256,
                predict_channels : int = 1,
                norm : str = "BN"
                ) -> None:
        super().__init__()
        
        in_dim = feature_channels[0] 
        self.embed_dim = embed_dim

        self.ca_high = CrossAttentionBlock(in_dim)
        self.ca_low = CrossAttentionBlock(in_dim)

        self.fusion_conv = nn.Sequential(
            nn.Conv2d(in_dim * 2, embed_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.GELU(),
            nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(embed_dim)
        )

        self.sa_mlp = SelfAttentionMLPHead(embed_dim)

        self.linear_predict = nn.Conv2d(embed_dim, predict_channels, kernel_size=1)
        
    def forward(self, x):
        c1, c2, c3, c4 = x 

        tgt_size = c1.shape[2:] 
        c2_resized = F.interpolate(c2, size=tgt_size, mode='bilinear', align_corners=False)
        c3_resized = F.interpolate(c3, size=tgt_size, mode='bilinear', align_corners=False)
        c4_resized = F.interpolate(c4, size=tgt_size, mode='bilinear', align_corners=False)

        feat_high = self.ca_high(c1, c3_resized)
        feat_low = self.ca_low(c2_resized, c4_resized)
        
        fused = torch.cat([feat_high, feat_low], dim=1)
        
        x = self.fusion_conv(fused)
        x = self.sa_mlp(x)
        x = self.linear_predict(x)
        
        return x


def get_world_size() -> int:
    if not dist.is_available():
        return 1
    if not dist.is_initialized():
        return 1
    return dist.get_world_size()

TORCH_VERSION = tuple(int(x) for x in torch.__version__.split(".")[:2])

def get_rel_pos(q_size, k_size, rel_pos):
    max_rel_dist = int(2 * max(q_size, k_size) - 1)
    if rel_pos.shape[0] != max_rel_dist:
        rel_pos_resized = F.interpolate(
            rel_pos.reshape(1, rel_pos.shape[0], -1).permute(0, 2, 1),
            size=max_rel_dist,
            mode="linear",
        )
        rel_pos_resized = rel_pos_resized.reshape(-1, max_rel_dist).permute(1, 0)
    else:
        rel_pos_resized = rel_pos

    q_coords = torch.arange(q_size)[:, None] * max(k_size / q_size, 1.0)
    k_coords = torch.arange(k_size)[None, :] * max(q_size / k_size, 1.0)
    relative_coords = (q_coords - k_coords) + (k_size - 1) * max(q_size / k_size, 1.0)

    return rel_pos_resized[relative_coords.long()]


def add_decomposed_rel_pos(attn, q, rel_pos_h, rel_pos_w, q_size, k_size):
    q_h, q_w = q_size
    k_h, k_w = k_size
    Rh = get_rel_pos(q_h, k_h, rel_pos_h)
    Rw = get_rel_pos(q_w, k_w, rel_pos_w)

    B, _, dim = q.shape
    r_q = q.reshape(B, q_h, q_w, dim)
    rel_h = torch.einsum("bhwc,hkc->bhwk", r_q, Rh)
    rel_w = torch.einsum("bhwc,wkc->bhwk", r_q, Rw)

    attn = (
        attn.view(B, q_h, q_w, k_h, k_w) + rel_h[:, :, :, :, None] + rel_w[:, :, :, None, :]
    ).view(B, q_h * q_w, k_h * k_w)

    return attn


class Attention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=True,
        use_rel_pos=False,
        rel_pos_zero_init=True,
        input_size=None,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

        self.use_rel_pos = use_rel_pos
        if self.use_rel_pos:
            self.rel_pos_h = nn.Parameter(torch.zeros(2 * input_size[0] - 1, head_dim))
            self.rel_pos_w = nn.Parameter(torch.zeros(2 * input_size[1] - 1, head_dim))

            if not rel_pos_zero_init:
                trunc_normal_(self.rel_pos_h, std=0.02)
                trunc_normal_(self.rel_pos_w, std=0.02)

    def forward(self, x):
        B, H, W, _ = x.shape
        qkv = self.qkv(x).reshape(B, H * W, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.reshape(3, B * self.num_heads, H * W, -1).unbind(0)

        attn = (q * self.scale) @ k.transpose(-2, -1)

        if self.use_rel_pos:
            attn = add_decomposed_rel_pos(attn, q, self.rel_pos_h, self.rel_pos_w, (H, W), (H, W))

        attn = attn.softmax(dim=-1)
        x = (attn @ v).view(B, self.num_heads, H, W, -1).permute(0, 2, 3, 1, 4).reshape(B, H, W, -1)
        x = self.proj(x)

        return x

class FrozenBatchNorm2d(nn.Module):
    _version = 3

    def __init__(self, num_features, eps=1e-5):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.register_buffer("weight", torch.ones(num_features))
        self.register_buffer("bias", torch.zeros(num_features))
        self.register_buffer("running_mean", torch.zeros(num_features))
        self.register_buffer("running_var", torch.ones(num_features) - eps)

    def forward(self, x):
        if x.requires_grad:
            scale = self.weight * (self.running_var + self.eps).rsqrt()
            bias = self.bias - self.running_mean * scale
            scale = scale.reshape(1, -1, 1, 1)
            bias = bias.reshape(1, -1, 1, 1)
            out_dtype = x.dtype
            return x * scale.to(out_dtype) + bias.to(out_dtype)
        else:
            return F.batch_norm(
                x,
                self.running_mean,
                self.running_var,
                self.weight,
                self.bias,
                training=False,
                eps=self.eps,
            )

    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
    ):
        version = local_metadata.get("version", None)

        if version is None or version < 2:
            if prefix + "running_mean" not in state_dict:
                state_dict[prefix + "running_mean"] = torch.zeros_like(self.running_mean)
            if prefix + "running_var" not in state_dict:
                state_dict[prefix + "running_var"] = torch.ones_like(self.running_var)

        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
        )

    def __repr__(self):
        return "FrozenBatchNorm2d(num_features={}, eps={})".format(self.num_features, self.eps)

    @classmethod
    def convert_frozen_batchnorm(cls, module):
        bn_module = nn.modules.batchnorm
        bn_module = (bn_module.BatchNorm2d, bn_module.SyncBatchNorm)
        res = module
        if isinstance(module, bn_module):
            res = cls(module.num_features)
            if module.affine:
                res.weight.data = module.weight.data.clone().detach()
                res.bias.data = module.bias.data.clone().detach()
            res.running_mean.data = module.running_mean.data
            res.running_var.data = module.running_var.data
            res.eps = module.eps
        else:
            for name, child in module.named_children():
                new_child = cls.convert_frozen_batchnorm(child)
                if new_child is not child:
                    res.add_module(name, new_child)
        return res


class CNNBlockBase(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride

    def freeze(self):
        for p in self.parameters():
            p.requires_grad = False
        FrozenBatchNorm2d.convert_frozen_batchnorm(self)
        return self


class Conv2d(torch.nn.Conv2d):
    def __init__(self, *args, **kwargs):
        norm = kwargs.pop("norm", None)
        activation = kwargs.pop("activation", None)
        super().__init__(*args, **kwargs)

        self.norm = norm
        self.activation = activation

    def forward(self, x):
        if not torch.jit.is_scripting():
            with warnings.catch_warnings(record=True):
                if x.numel() == 0 and self.training:
                    assert not isinstance(
                        self.norm, torch.nn.SyncBatchNorm
                    ), "SyncBatchNorm does not support empty inputs!"

        x = F.conv2d(
            x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups
        )
        if self.norm is not None:
            x = self.norm(x)
        if self.activation is not None:
            x = self.activation(x)
        return x
    
BatchNorm2d = torch.nn.BatchNorm2d
  
class NaiveSyncBatchNorm(BatchNorm2d):
    def __init__(self, *args, stats_mode="", **kwargs):
        super().__init__(*args, **kwargs)
        assert stats_mode in ["", "N"]
        self._stats_mode = stats_mode

    def forward(self, input):
        if get_world_size() == 1 or not self.training:
            return super().forward(input)

        B, C = input.shape[0], input.shape[1]

        half_input = input.dtype == torch.float16
        if half_input:
            input = input.float()
        mean = torch.mean(input, dim=[0, 2, 3])
        meansqr = torch.mean(input * input, dim=[0, 2, 3])

        if self._stats_mode == "":
            assert B > 0, 'SyncBatchNorm(stats_mode="") does not support zero batch size.'
            vec = torch.cat([mean, meansqr], dim=0)
            vec = differentiable_all_reduce(vec) * (1.0 / dist.get_world_size())
            mean, meansqr = torch.split(vec, C)
            momentum = self.momentum
        else:
            if B == 0:
                vec = torch.zeros([2 * C + 1], device=mean.device, dtype=mean.dtype)
                vec = vec + input.sum()
            else:
                vec = torch.cat(
                    [mean, meansqr, torch.ones([1], device=mean.device, dtype=mean.dtype)], dim=0
                )
            vec = differentiable_all_reduce(vec * B)

            total_batch = vec[-1].detach()
            momentum = total_batch.clamp(max=1) * self.momentum
            mean, meansqr, _ = torch.split(vec / total_batch.clamp(min=1), C)

        var = meansqr - mean * mean
        invstd = torch.rsqrt(var + self.eps)
        scale = self.weight * invstd
        bias = self.bias - mean * scale
        scale = scale.reshape(1, -1, 1, 1)
        bias = bias.reshape(1, -1, 1, 1)

        self.running_mean += momentum * (mean.detach() - self.running_mean)
        self.running_var += momentum * (var.detach() - self.running_var)
        ret = input * scale + bias
        if half_input:
            ret = ret.half()
        return ret

def get_norm(norm, out_channels):
    if norm is None:
        return None
    if isinstance(norm, str):
        if len(norm) == 0:
            return None
        norm = {
            "BN": BatchNorm2d,
            "SyncBN": NaiveSyncBatchNorm if TORCH_VERSION <= (1, 5) else nn.SyncBatchNorm,
            "FrozenBN": FrozenBatchNorm2d,
            "GN": lambda channels: nn.GroupNorm(32, channels),
            "nnSyncBN": nn.SyncBatchNorm,
            "naiveSyncBN": NaiveSyncBatchNorm,
            "naiveSyncBN_N": lambda channels: NaiveSyncBatchNorm(channels, stats_mode="N"),
            "LN": lambda channels: LayerNorm(channels),
        }[norm]
    return norm(out_channels)


class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x

class ResBottleneckBlock(CNNBlockBase):
    def __init__(
        self,
        in_channels,
        out_channels,
        bottleneck_channels,
        norm="LN",
        act_layer=nn.GELU,
    ):
        super().__init__(in_channels, out_channels, 1)

        self.conv1 = Conv2d(in_channels, bottleneck_channels, 1, bias=False)
        self.norm1 = get_norm(norm, bottleneck_channels)
        self.act1 = act_layer()

        self.conv2 = Conv2d(
            bottleneck_channels,
            bottleneck_channels,
            3,
            padding=1,
            bias=False,
        )
        self.norm2 = get_norm(norm, bottleneck_channels)
        self.act2 = act_layer()

        self.conv3 = Conv2d(bottleneck_channels, out_channels, 1, bias=False)
        self.norm3 = get_norm(norm, out_channels)

        for layer in [self.conv1, self.conv2, self.conv3]:
            weight_init.c2_msra_fill(layer)
        for layer in [self.norm1, self.norm2]:
            layer.weight.data.fill_(1.0)
            layer.bias.data.zero_()
        self.norm3.weight.data.zero_()
        self.norm3.bias.data.zero_()

    def forward(self, x):
        out = x
        for layer in self.children():
            out = layer(out)

        out = x + out
        return out

def window_partition(x, window_size):
    B, H, W, C = x.shape

    pad_h = (window_size - H % window_size) % window_size
    pad_w = (window_size - W % window_size) % window_size
    if pad_h > 0 or pad_w > 0:
        x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
    Hp, Wp = H + pad_h, W + pad_w

    x = x.view(B, Hp // window_size, window_size, Wp // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows, (Hp, Wp)


def window_unpartition(windows, window_size, pad_hw, hw):
    Hp, Wp = pad_hw
    H, W = hw
    B = windows.shape[0] // (Hp * Wp // window_size // window_size)
    x = windows.view(B, Hp // window_size, Wp // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, Hp, Wp, -1)

    if Hp > H or Wp > W:
        x = x[:, :H, :W, :].contiguous()
    return x

class Block(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=True,
        drop_path=0.0,
        norm_layer=nn.LayerNorm,
        act_layer=nn.GELU,
        use_rel_pos=False,
        rel_pos_zero_init=True,
        window_size=0,
        use_residual_block=False,
        input_size=None,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            use_rel_pos=use_rel_pos,
            rel_pos_zero_init=rel_pos_zero_init,
            input_size=input_size if window_size == 0 else (window_size, window_size),
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer)

        self.window_size = window_size

        self.use_residual_block = use_residual_block
        if use_residual_block:
            self.residual = ResBottleneckBlock(
                in_channels=dim,
                out_channels=dim,
                bottleneck_channels=dim // 2,
                norm="LN",
                act_layer=act_layer,
            )

    def forward(self, x):
        shortcut = x
        x = self.norm1(x)
        if self.window_size > 0:
            H, W = x.shape[1], x.shape[2]
            x, pad_hw = window_partition(x, self.window_size)

        x = self.attn(x)
        if self.window_size > 0:
            x = window_unpartition(x, self.window_size, pad_hw, (H, W))

        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        if self.use_residual_block:
            x = self.residual(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        return x



class PatchEmbed(nn.Module):
    def __init__(
        self, kernel_size=(16, 16), stride=(16, 16), padding=(0, 0), in_chans=3, embed_dim=768
    ):
        super().__init__()

        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=kernel_size, stride=stride, padding=padding
        )

    def forward(self, x):
        x = self.proj(x)
        x = x.permute(0, 2, 3, 1)
        return x


def get_abs_pos(abs_pos, has_cls_token, hw):
    h, w = hw
    if has_cls_token:
        abs_pos = abs_pos[:, 1:]
    xy_num = abs_pos.shape[1]
    size = int(math.sqrt(xy_num))
    assert size * size == xy_num

    if size != h or size != w:
        new_abs_pos = F.interpolate(
            abs_pos.reshape(1, size, size, -1).permute(0, 3, 1, 2),
            size=(h, w),
            mode="bicubic",
            align_corners=False,
        )

        return new_abs_pos.permute(0, 2, 3, 1)
    else:
        return abs_pos.reshape(1, h, w, -1)

class ViT(nn.Module):
    def __init__(
        self,
        img_size=256,
        patch_size=16,
        in_chans=3,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        drop_path_rate=0.0,
        norm_layer=nn.LayerNorm,
        act_layer=nn.GELU,
        use_abs_pos=True,
        use_rel_pos=False,
        rel_pos_zero_init=True,
        window_size=0,
        window_block_indexes=(),
        residual_block_indexes=(),
        use_act_checkpoint=False,
        pretrain_img_size=224,
        pretrain_use_cls_token=True,
        out_feature="last_feat",
    ):
        super().__init__()
        self.pretrain_use_cls_token = pretrain_use_cls_token

        self.patch_embed = PatchEmbed(
            kernel_size=(patch_size, patch_size),
            stride=(patch_size, patch_size),
            in_chans=in_chans,
            embed_dim=embed_dim,
        )

        if use_abs_pos:
            num_patches = (pretrain_img_size // patch_size) * (pretrain_img_size // patch_size)
            num_positions = (num_patches + 1) if pretrain_use_cls_token else num_patches
            self.pos_embed = nn.Parameter(torch.zeros(1, num_positions, embed_dim))
        else:
            self.pos_embed = None

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]

        self.blocks = nn.ModuleList()
        for i in range(depth):
            block = Block(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop_path=dpr[i],
                norm_layer=norm_layer,
                act_layer=act_layer,
                use_rel_pos=use_rel_pos,
                rel_pos_zero_init=rel_pos_zero_init,
                window_size=window_size if i in window_block_indexes else 0,
                use_residual_block=i in residual_block_indexes,
                input_size=(img_size // patch_size, img_size // patch_size),
            )

            self.blocks.append(block)

        self._out_feature_channels = {out_feature: embed_dim}
        self._out_feature_strides = {out_feature: patch_size}
        self._out_features = [out_feature]

        self.output_size = (img_size // patch_size, img_size // patch_size)
        self.stride = patch_size
        self.channels = embed_dim
        
        if self.pos_embed is not None:
            trunc_normal_(self.pos_embed, std=0.02)

        self.apply(self._init_weights)

    def output_shape(self):
        return {
            '${.net.out_feature}':self.output_size,
            'stride': self.stride,
            'channels' : self.channels
            } 
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def add_position_embed(self, x):
        if self.pos_embed is not None:
            x = x + get_abs_pos(
                self.pos_embed, self.pretrain_use_cls_token, (x.shape[1], x.shape[2])
            )
        return x
    def mae_forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        return x.permute(0, 3, 1, 2)
    
    def forward(self, x):
        x = self.patch_embed(x)
        x = self.add_position_embed(x)
        for blk in self.blocks:
            x = blk(x)

        outputs = {self._out_features[0]: x.permute(0, 3, 1, 2)}
        return outputs


class LastLevelMaxPool(nn.Module):
    def __init__(self):
        super().__init__()
        self.num_levels = 1
        self.in_feature = "p5"

    def forward(self, x):
        return [F.max_pool2d(x, kernel_size=1, stride=2, padding=0)]


class BottomUpsample(nn.Module):
    """
    This module is used to upsample the input feature map from P2
    to generate a higher resolution feature map.
    """

    def __init__(self):
        super().__init__()
        self.num_levels = 1
        self.in_feature = "p2"

    def forward(self, x):
        return [F.interpolate(x, scale_factor=2, mode='nearest')]

def _assert_strides_are_log2_contiguous(strides):
    """
    Assert that each stride is 2x times its preceding stride, i.e. "contiguous in log2".
    """
    for i, stride in enumerate(strides[1:], 1):
        assert stride == 2 * strides[i - 1], "Strides {} {} are not log2 contiguous".format(
            stride, strides[i - 1]
        )


class ResidualConvUnit(nn.Module):
    def __init__(self, in_channels, out_channels, use_bias=True, norm_layer=nn.BatchNorm2d):
        super().__init__()

        self.proj = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=use_bias)
        self.proj_norm = norm_layer(out_channels)
        
        self.res_branch = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=use_bias),
            norm_layer(out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=use_bias),
            norm_layer(out_channels)
        )
        self.act = nn.GELU()

    def forward(self, x):
        x = self.proj_norm(self.proj(x))
        residual = self.res_branch(x)
        return self.act(x + residual)


class MultiScaleFeatureEnhancementBottleneck(nn.Module):
    def __init__(self,
        in_feature_shape,
        out_channels,
        scale_factors,
        input_stride = 16,
        top_block=None,
        norm="BN"
    ) -> None:
        super().__init__()
        
        _, dim, H, W = in_feature_shape
        self.scale_factors = scale_factors
        
        self.stages = nn.ModuleList()
        strides = [input_stride // s for s in scale_factors]
        
        if norm == "BN":
            norm_layer = nn.BatchNorm2d
        elif norm == "IN":
            norm_layer = partial(nn.InstanceNorm2d, track_running_stats=True)
        else:
            norm_layer = nn.BatchNorm2d

        for idx, scale in enumerate(scale_factors):
            if scale == 4.0:   # Upsample x4
                scale_op = nn.Sequential(
                    nn.ConvTranspose2d(dim, dim // 2, kernel_size=2, stride=2),
                    norm_layer(dim // 2),
                    nn.GELU(),
                    nn.ConvTranspose2d(dim // 2, dim, kernel_size=2, stride=2)
                )
            elif scale == 2.0: # Upsample x2
                scale_op = nn.ConvTranspose2d(dim, dim, kernel_size=2, stride=2)
            elif scale == 1.0: # Identity
                scale_op = nn.Identity()
            elif scale == 0.5: # Downsample
                scale_op = nn.MaxPool2d(kernel_size=2, stride=2)
            else:
                raise NotImplementedError(f"scale_factor={scale} is not supported.")

            res_block = ResidualConvUnit(
                in_channels=dim, 
                out_channels=out_channels, 
                norm_layer=norm_layer
            )
            
            self.stages.append(nn.Sequential(scale_op, res_block))

        self._out_feature_strides = {"p{}".format(int(math.log2(s))): s for s in strides}
        self._out_features = list(self._out_feature_strides.keys())

    def forward(self, x):
        features = x['last_feat']
        results = []

        for stage in self.stages:
            results.append(stage(features))

        return {f: res for f, res in zip(self._out_features, results)}


class hadetector_model(nn.Module):
    
    def __init__(
        self, 
        # Feature Extraction Backbone
        input_size = 256,
        patch_size = 16,
        embed_dim = 768,
        vit_pretrain_path = None,
        # Multi-scale Feature Enhancement Bottleneck
        fpn_channels = 256,
        fpn_scale_factors = (4.0, 2.0, 1.0, 0.5),
        # Artifacts Prediction Head
        mlp_embeding_dim = 256,
        predict_head_norm = "BN",
        # Edge loss
        edge_lambda = 20,
    ):
        super(hadetector_model, self).__init__()
        self.input_size = input_size
        self.patch_size = patch_size
        
        # Feature Extraction Backbone (FEB)
        self.encoder_net = ViT(  
            img_size = input_size,
            patch_size=16,
            embed_dim=embed_dim,
            depth=12,
            num_heads=12,
            drop_path_rate=0.1,
            window_size=14,
            mlp_ratio=4,
            qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            window_block_indexes=[0, 1, 3, 4, 6, 7, 9, 10],
            residual_block_indexes=[],
            use_rel_pos=True,
            out_feature="last_feat",
        )
        self.vit_pretrain_path = vit_pretrain_path
        
        # Multi-scale Feature Enhancement Bottleneck (MFEB)
        self.featurePyramid_net = MultiScaleFeatureEnhancementBottleneck(
            in_feature_shape= (1, embed_dim, 256, 256),
            out_channels= fpn_channels,
            scale_factors=fpn_scale_factors,
            input_stride=16,
            norm="BN",
            top_block=None,
        )

        # Artifact Prediction Head
        self.artifact_predict_head = ArtifactPredictionHead(
            feature_channels=[fpn_channels for i in range(4)], 
            embed_dim=mlp_embeding_dim,
            predict_channels=1,
            norm=predict_head_norm 
        )

        # Loss
        self.BCE_loss = nn.BCEWithLogitsLoss()
        self.edge_lambda = edge_lambda
        self.apply(self._init_weights)
        self._mae_init_weights()
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
    
    def _mae_init_weights(self):
        if self.vit_pretrain_path is not None:
            try:
                checkpoint = torch.load(self.vit_pretrain_path, map_location='cpu')
                state_dict = checkpoint['model'] if 'model' in checkpoint else checkpoint
                
                msg = self.encoder_net.load_state_dict(state_dict, strict=False)
                print(f'Load pretrained weights from \'{self.vit_pretrain_path}\'. Msg: {msg}')
            except Exception as e:
                print(f"Error loading pretrained weights: {e}")

    def forward(self, x:torch.Tensor, masks, edge_masks, shape= None):
        x = self.encoder_net(x)
        
        x = self.featurePyramid_net(x)
        
        feature_list = []
        for k, v in x.items():
            feature_list.append(v)
        
        logits = self.artifact_predict_head(feature_list)

        mask_pred = F.interpolate(logits, size = (self.input_size, self.input_size), mode='bilinear', align_corners=False)
        
        if self.training:
            mask_pred = torch.clamp(mask_pred, min=-10.0, max=10.0)

        predict_loss = self.BCE_loss(mask_pred, masks)
        
        edge_loss = F.binary_cross_entropy_with_logits(
            input = mask_pred,
            target= masks, 
            weight = edge_masks
            ) * self.edge_lambda 
        
        predict_loss += edge_loss
        mask_pred_sigmoid = torch.sigmoid(mask_pred)
        return predict_loss, mask_pred_sigmoid, edge_loss
    
    def __init__(
        self, 
        # Feature Extraction Backbone
        input_size = 256,
        patch_size = 16,
        embed_dim = 768,
        vit_pretrain_path = None,
        # Multi-scale Feature Enhancement Bottleneck
        fpn_channels = 256,
        fpn_scale_factors = (4.0, 2.0, 1.0, 0.5),
        # Artifacts Prediction Head
        mlp_embeding_dim = 256,
        predict_head_norm = "BN",
        # Edge loss
        edge_lambda = 20,
    ):
        super(hadetector_model, self).__init__()
        self.input_size = input_size
        self.patch_size = patch_size
        
        # Feature Extraction Backbone (FEB)
        self.encoder_net = ViT(  
            img_size = input_size,
            patch_size=16,
            embed_dim=embed_dim,
            depth=12,
            num_heads=12,
            drop_path_rate=0.1,
            window_size=14,
            mlp_ratio=4,
            qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            window_block_indexes=[0, 1, 3, 4, 6, 7, 9, 10],
            residual_block_indexes=[],
            use_rel_pos=True,
            out_feature="last_feat",
        )
        self.vit_pretrain_path = vit_pretrain_path
        
        # Multi-scale Feature Enhancement Bottleneck (MFEB)
        self.featurePyramid_net = MultiScaleFeatureEnhancementBottleneck(
            in_feature_shape= (1, embed_dim, 256, 256),
            out_channels= fpn_channels,
            scale_factors=fpn_scale_factors,
            input_stride=16,
            norm="BN",
            top_block=None,
        )

        # Artifacts Prediction Head
        self.predict_head = ArtifactPredictionHead(
            feature_channels=[fpn_channels for i in range(4)], 
            embed_dim=mlp_embeding_dim,
            predict_channels=1,
            norm=predict_head_norm 
        )

        # Loss
        self.BCE_loss = nn.BCEWithLogitsLoss()
        self.edge_lambda = edge_lambda
        
        self.apply(self._init_weights)
        self._mae_init_weights()
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
    
    def _mae_init_weights(self):
        if self.vit_pretrain_path != None:
            self.encoder_net.load_state_dict(
                torch.load(self.vit_pretrain_path, map_location='cpu')['model'], 
                strict=False
            )
            print('load pretrained weights from \'{}\'.'.format(self.vit_pretrain_path))

    def forward(self, x:torch.Tensor, masks, edge_masks, shape= None):
        # 1. Backbone
        x = self.encoder_net(x)
        
        # 2. FPN
        x = self.featurePyramid_net(x)
        
        feature_list = []
        for k, v in x.items():
            feature_list.append(v)
        
        # 3. Artifact Prediction Head
        logits = self.artifact_predict_head(feature_list)
        
        # 4. Upsample to original image size
        mask_pred = F.interpolate(logits, size = (self.input_size, self.input_size), mode='bilinear', align_corners=False)
        
        # 5. Loss Calculation
        predict_loss = self.BCE_loss(mask_pred, masks)
        edge_loss = F.binary_cross_entropy_with_logits(
            input = mask_pred,
            target= masks, 
            weight = edge_masks
            ) * self.edge_lambda 
        predict_loss += edge_loss
        
        # Sigmoid for output
        mask_pred = torch.sigmoid(mask_pred)
        
        return predict_loss, mask_pred, edge_loss