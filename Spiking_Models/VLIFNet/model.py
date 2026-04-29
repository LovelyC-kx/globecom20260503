"""
VLIFNet — ported from system1_VLIFNet/model.py

Changes vs. the original file (kept minimal on purpose):

1.  `from fsta_module import …`  →  `from .fsta_module import …`
    (package-relative import so the file can live under
    Spiking_Models/VLIFNet/)

2.  Introduce a module-level `_BACKEND` so the caller may pick 'torch' or
    'cupy'; all LIFNode instances honour it.  v1 defaults to 'torch'
    because cupy wheels sometimes fail to build on user machines.

3.  Add `build_vlifnet(...)` / `set_vlifnet_backend(...)` factory helpers.
    The original `model()` convenience wrapper is preserved for backwards
    compat with train.py in the upstream repo.

4.  `VLIFNet.forward` unchanged.  MultiSpike4 / mem_update unchanged.
    ThresholdDependentBatchNorm2d, MultiDimensionalAttention, LIFNode
    are imported from spikingjelly exactly as in the original.

Nothing about the network numerics is modified — MultiSpike4's 4-level
quantiser, the SRB dual-group frequency filter, GatedSkipFusion, FSTA
and FreMLPBlock all behave identically to upstream.
"""

from spikingjelly.activation_based.neuron import LIFNode
from spikingjelly.activation_based import functional, layer
import torch
import torch.nn as nn
import torch.nn.functional as F

from .fsta_module import FSTAModule, FreMLPBlock

# ---------------------------------------------------------------------------
# Module-level knobs  (caller may change these before instantiation)
# ---------------------------------------------------------------------------
_BACKEND = "torch"    # v1 default; set_vlifnet_backend('cupy') to switch
v_th = 0.15
alpha = 1 / (2 ** 0.5)
decay = 0.25          # MultiSpike4 / mem_update decay constant


def set_vlifnet_backend(backend: str) -> None:
    """Set backend used by all LIFNode layers instantiated *after* this call.

    Passing 'cupy' on a machine without a compatible cupy install will raise
    at forward time; CloudRemovalSNNTask in v1 catches this and falls back.
    """
    global _BACKEND
    assert backend in ("torch", "cupy"), f"backend must be 'torch' or 'cupy', got {backend}"
    _BACKEND = backend


# ---------------------------------------------------------------------------
# 4-level quantised spike (default VLIFNet neuron, UNCHANGED from upstream)
# ---------------------------------------------------------------------------
class MultiSpike4(nn.Module):
    class quant4(torch.autograd.Function):
        @staticmethod
        def forward(ctx, input):
            ctx.save_for_backward(input)
            # First quantise in the range 0-4, then divide by 4 to cap at 1.
            quantized = torch.round(torch.clamp(input, min=0, max=4))
            return quantized / 4.0

        @staticmethod
        def backward(ctx, grad_output):
            input, = ctx.saved_tensors
            grad_input = grad_output.clone()
            grad_input[input < 0] = 0
            grad_input[input > 4] = 0
            return grad_input / 4.0

    def forward(self, x):
        return self.quant4.apply(x)


class mem_update(nn.Module):
    def __init__(self):
        super(mem_update, self).__init__()
        self.qtrick = MultiSpike4()

    def forward(self, x):
        spike = torch.zeros_like(x[0]).to(x.device)
        output = torch.zeros_like(x)
        mem_old = 0
        time_window = x.shape[0]
        for i in range(time_window):
            if i >= 1:
                mem = (mem_old - spike.detach()) * decay + x[i]
            else:
                mem = x[i]
            spike = self.qtrick(mem)
            mem_old = mem.clone()
            output[i] = spike
        return output


# ---------------------------------------------------------------------------
# PixelShuffleLIFBlock
# ---------------------------------------------------------------------------
class PixelShuffleLIFBlock(nn.Module):
    def __init__(self, in_channels, downsample_factor=2):
        super().__init__()
        self.downsample_factor = downsample_factor
        self.r = downsample_factor

        self.pixel_unshuffle = nn.PixelUnshuffle(downsample_factor)
        self.lif_node = mem_update()

        self.channel_adjust = layer.Conv2d(
            in_channels * (downsample_factor ** 2),
            in_channels,
            kernel_size=1,
            bias=False,
            step_mode="m",
        )
        functional.set_step_mode(self.channel_adjust, step_mode="m")

    def forward(self, x):
        T, B, C, H, W = x.shape
        x_flat = x.reshape(T * B, C, H, W)
        x_downsampled = self.pixel_unshuffle(x_flat)
        x_reshaped = x_downsampled.reshape(T, B, C * self.r ** 2, H // self.r, W // self.r)
        x_patch_separated = x_reshaped.reshape(T, B, C, self.r ** 2, H // self.r, W // self.r)
        x_temporal = x_patch_separated.permute(0, 3, 1, 2, 4, 5).contiguous()
        x_temporal = x_temporal.reshape(T * self.r ** 2, B, C, H // self.r, W // self.r)
        x_lif_output = self.lif_node(x_temporal)
        x_reorganized = x_lif_output.reshape(T, self.r ** 2, B, C, H // self.r, W // self.r)
        x_reorganized = x_reorganized.permute(0, 2, 3, 1, 4, 5).contiguous()
        x_reorganized = x_reorganized.reshape(T, B, C * self.r ** 2, H // self.r, W // self.r)
        x_flat_reorganized = x_reorganized.reshape(T * B, C * self.r ** 2, H // self.r, W // self.r)
        x_upsampled = F.interpolate(x_flat_reorganized, size=(H, W), mode="bilinear", align_corners=False)
        x_restored = x_upsampled.reshape(T, B, C * self.r ** 2, H, W)
        x_final = self.channel_adjust(x_restored)
        return x_final


# ---------------------------------------------------------------------------
# TimeAttention (3D channel attention over time)
# ---------------------------------------------------------------------------
class TimeAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.max_pool = nn.AdaptiveMaxPool3d(1)
        self.sharedMLP = nn.Sequential(
            nn.Conv3d(in_planes, in_planes // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv3d(in_planes // ratio, in_planes, 1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = self.sharedMLP(self.avg_pool(x))
        maxout = self.sharedMLP(self.max_pool(x))
        return self.sigmoid(avgout + maxout)


class SUNet_Level1_Block(nn.Module):
    """
    SUNet module for Level 1.
    1. Spiking_Residual_Block
    2. PixelShuffleLIFBlock downsampling (extended time dim)
    3. TimeAttention + 3D Conv compress time dim
    4. 2 × [LIF-Conv2d-TDBN-MultiDimensionalAttention]
    5. Upsample back
    6. Skip connection + MultiDimensionalAttention
    7. FreMLPBlock
    """
    def __init__(self, dim):
        super().__init__()
        functional.set_step_mode(self, step_mode="m")

        self.downsample_factor = 2
        self.r = 2

        self.initial_residual = Spiking_Residual_Block(dim=dim)
        self.pixel_unshuffle = nn.PixelUnshuffle(self.downsample_factor)
        self.lif_node = mem_update()
        self.time_attention = TimeAttention(in_planes=16, ratio=4)  # T * r^2 = 16
        self.temporal_compress = nn.Sequential(
            nn.Conv3d(dim, dim, kernel_size=(4, 1, 1), stride=(4, 1, 1), padding=0, bias=False),
            nn.ReLU(inplace=True),
        )
        self.lif_1 = LIFNode(v_threshold=v_th, backend=_BACKEND, step_mode="m", decay_input=False)
        self.conv_1 = layer.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, bias=False, step_mode="m")
        self.bn_1 = layer.ThresholdDependentBatchNorm2d(num_features=dim, alpha=alpha, v_th=v_th, affine=True)
        self.attn_1 = layer.MultiDimensionalAttention(T=4, reduction_t=4, reduction_c=16, kernel_size=3, C=dim)

        self.lif_2 = LIFNode(v_threshold=v_th, backend=_BACKEND, step_mode="m", decay_input=False)
        self.conv_2 = layer.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, bias=False, step_mode="m")
        self.bn_2 = layer.ThresholdDependentBatchNorm2d(num_features=dim, alpha=alpha, v_th=v_th, affine=True)
        self.attn_2 = layer.MultiDimensionalAttention(T=4, reduction_t=4, reduction_c=16, kernel_size=3, C=dim)

        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.final_attn = layer.MultiDimensionalAttention(T=4, reduction_t=4, reduction_c=16, kernel_size=3, C=dim)
        self.fre_mlp = FreMLPBlock(channels=dim, expand=2)

    def forward(self, x):
        residual_out = self.initial_residual(x)
        skip_features = residual_out.clone()

        T, B, C, H, W = residual_out.shape
        x_flat = residual_out.reshape(T * B, C, H, W)
        x_downsampled = self.pixel_unshuffle(x_flat)
        x_reshaped = x_downsampled.reshape(T, B, C * self.r ** 2, H // self.r, W // self.r)
        x_patch_separated = x_reshaped.reshape(T, B, C, self.r ** 2, H // self.r, W // self.r)
        x_temporal = x_patch_separated.permute(0, 3, 1, 2, 4, 5).contiguous()
        downsampled = x_temporal.reshape(T * self.r ** 2, B, C, H // self.r, W // self.r)
        downsampled = self.lif_node(downsampled)

        T_extended, B, C, H_down, W_down = downsampled.shape
        downsampled_transposed = downsampled.transpose(0, 1)
        time_att = self.time_attention(downsampled_transposed)
        attended = downsampled_transposed * time_att
        attended = attended.transpose(0, 1)
        attended_3d = attended.permute(1, 2, 0, 3, 4)
        compressed = self.temporal_compress(attended_3d)
        out = compressed.permute(2, 0, 1, 3, 4)

        out = self.lif_1(out)
        out = self.conv_1(out)
        out = self.bn_1(out)
        out = self.attn_1(out)

        out = self.lif_2(out)
        out = self.conv_2(out)
        out = self.bn_2(out)
        out = self.attn_2(out)

        T, B, C, H, W = out.shape
        out_flat = out.reshape(T * B, C, H, W)
        upsampled_flat = self.upsample(out_flat)
        upsampled = upsampled_flat.reshape(T, B, C, upsampled_flat.shape[2], upsampled_flat.shape[3])

        combined = upsampled + skip_features
        final_out = self.final_attn(combined)
        final_out = self.fre_mlp(final_out)
        return final_out


class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=32, spike_mode="lif",
                 LayerNorm_type="WithBias", bias=False, T=4):
        super().__init__()
        functional.set_step_mode(self, step_mode="m")
        self.proj = layer.Conv2d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x):
        x = self.proj(x)
        return x


class Spiking_Residual_Block(nn.Module):
    def __init__(self, dim):
        super().__init__()
        functional.set_step_mode(self, step_mode="m")

        self.lif_1 = LIFNode(v_threshold=v_th, backend=_BACKEND, step_mode="m", decay_input=False)
        self.conv1 = layer.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, bias=False, step_mode="m")
        self.bn1 = layer.ThresholdDependentBatchNorm2d(num_features=dim, alpha=alpha, v_th=v_th, affine=True)

        self.high_freq_scale_1 = nn.Parameter(torch.ones(1))
        self.low_freq_scale_1 = nn.Parameter(torch.ones(1))

        self.lif_2 = PixelShuffleLIFBlock(in_channels=dim, downsample_factor=2)
        self.conv2 = layer.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, bias=False, step_mode="m")
        self.bn2 = layer.ThresholdDependentBatchNorm2d(num_features=dim, alpha=alpha, v_th=v_th * 0.2, affine=True)

        self.high_freq_scale_2 = nn.Parameter(torch.ones(1))
        self.low_freq_scale_2 = nn.Parameter(torch.ones(1))

        self.shortcut = nn.Sequential(
            layer.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, bias=False, step_mode="m"),
            layer.ThresholdDependentBatchNorm2d(num_features=dim, alpha=alpha, v_th=v_th, affine=True),
        )

        self.attn = layer.MultiDimensionalAttention(T=4, reduction_t=4, reduction_c=16, kernel_size=3, C=dim)
        self.fsta = FSTAModule(channels=dim, T=4)
        self.cross_scale_gate = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        # Group 1: temporal high-frequency filter (LIF)
        x_h_1 = self.lif_1(x)
        x_l_1 = x - x_h_1
        x_h_1_scaled = self.high_freq_scale_1 * x_h_1
        x_l_1_scaled = self.low_freq_scale_1 * x_l_1
        x_enhanced_1 = x * x_h_1
        combined_features_1 = x_h_1_scaled + x_l_1_scaled + x_enhanced_1

        out = self.conv1(combined_features_1)
        out = self.bn1(out)

        # Group 2: spatial high-frequency filter (PixelShuffle)
        x_h_2 = self.lif_2(out)
        x_l_2 = out - x_h_2
        x_h_2_scaled = self.high_freq_scale_2 * x_h_2
        x_l_2_scaled = self.low_freq_scale_2 * x_l_2
        x_enhanced_2 = out * x_h_2

        x_cross = torch.sigmoid(self.cross_scale_gate) * x_h_1 * x_h_2
        combined_features_2 = x_h_2_scaled + x_l_2_scaled + x_enhanced_2 + x_cross

        out = self.conv2(combined_features_2)
        out = self.bn2(out)

        shortcut = torch.clone(x)
        out = out + self.shortcut(shortcut)
        out = self.attn(out) + shortcut
        out = self.fsta(out)
        return out


class DownSampling(nn.Module):
    def __init__(self, dim):
        super().__init__()
        functional.set_step_mode(self, step_mode="m")
        self.lif = LIFNode(v_threshold=v_th, backend=_BACKEND, step_mode="m", decay_input=False)
        self.conv = layer.Conv2d(dim, dim * 2, kernel_size=3, stride=2, padding=1, step_mode="m", bias=False)
        self.bn = layer.ThresholdDependentBatchNorm2d(alpha=alpha, v_th=v_th, num_features=dim * 2, affine=True)

    def forward(self, x):
        x = self.lif(x)
        x = self.conv(x)
        x = self.bn(x)
        return x


class UpSampling(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.scale_factor = 2
        self.lif = LIFNode(v_threshold=v_th, backend=_BACKEND, step_mode="m", decay_input=False)
        self.conv = layer.Conv2d(dim, dim // 2, kernel_size=3, stride=1, padding=1, step_mode="m", bias=False)
        self.bn = layer.ThresholdDependentBatchNorm2d(alpha=alpha, v_th=v_th, num_features=dim // 2, affine=True)

    def forward(self, input):
        T, B, C, H, W = input.shape
        x_flat = input.reshape(T * B, C, H, W)
        x_up = F.interpolate(x_flat, scale_factor=self.scale_factor, mode="bilinear", align_corners=False)
        out = x_up.reshape(T, B, C, H * self.scale_factor, W * self.scale_factor)

        out = self.lif(out)
        out = self.conv(out)
        out = self.bn(out)
        return out


class GatedSkipFusion(nn.Module):
    """Gated skip-connection fusion (per upstream)."""
    def __init__(self, dim):
        super().__init__()
        functional.set_step_mode(self, step_mode="m")
        self.gate_conv = layer.Conv2d(dim * 2, dim, kernel_size=1, bias=True, step_mode="m")
        self.lif = LIFNode(v_threshold=v_th, backend=_BACKEND, step_mode="m", decay_input=False)
        self.bn = layer.ThresholdDependentBatchNorm2d(num_features=dim, alpha=alpha, v_th=v_th)

    def forward(self, dec, enc):
        combined = torch.cat([dec, enc], dim=2)
        gate = torch.sigmoid(self.gate_conv(combined))
        fused = gate * dec + (1.0 - gate) * enc
        fused = self.lif(fused)
        fused = self.bn(fused)
        return fused


# ---------------------------------------------------------------------------
# Full VLIFNet
# ---------------------------------------------------------------------------
class VLIFNet(nn.Module):
    def __init__(self, inp_channels=3, out_channels=3, dim=24,
                 en_num_blocks=(2, 2, 4, 4), de_num_blocks=(2, 2, 2, 2),
                 bias=False, T=4, use_refinement=False):
        super().__init__()

        functional.set_backend(self, backend=_BACKEND)
        functional.set_step_mode(self, step_mode="m")

        self.T = T
        self.use_refinement = use_refinement
        self.patch_embed = OverlapPatchEmbed(in_c=inp_channels, embed_dim=dim, T=T)
        self.encoder_level1 = SUNet_Level1_Block(dim=int(dim * 1))

        self.down1_2 = DownSampling(dim)
        self.encoder_level2 = nn.Sequential(*[
            Spiking_Residual_Block(dim=int(dim * 2 ** 1)) for _ in range(en_num_blocks[1])])

        self.down2_3 = DownSampling(int(dim * 2 ** 1))
        self.encoder_level3 = nn.Sequential(*[
            Spiking_Residual_Block(dim=int(dim * 2 ** 2)) for _ in range(en_num_blocks[2])])

        self.decoder_level3 = nn.Sequential(*[
            Spiking_Residual_Block(dim=int(dim * 2 ** 2)) for _ in range(de_num_blocks[2])])

        self.up3_2 = UpSampling(int(dim * 2 ** 2))
        self.skip_fusion_level2 = GatedSkipFusion(dim=int(dim * 2 ** 1))
        self.decoder_level2 = nn.Sequential(*[
            Spiking_Residual_Block(dim=int(dim * 2 ** 1)) for _ in range(de_num_blocks[1])])

        self.up2_1 = UpSampling(int(dim * 2 ** 1))
        self.skip_fusion_level1 = GatedSkipFusion(dim=int(dim * 2 ** 0))
        self.decoder_level1 = SUNet_Level1_Block(dim=int(dim * 2 ** 0))
        self.additional_sunet_level1 = SUNet_Level1_Block(dim=int(dim * 2 ** 0))

        # Auxiliary supervision heads (v3 feature; instantiated but unused in v1)
        self.aux_head_level2 = nn.Conv2d(int(dim * 2 ** 1), out_channels, kernel_size=1)
        self.aux_head_level3 = nn.Conv2d(int(dim * 2 ** 2), out_channels, kernel_size=1)

        if self.use_refinement:
            self.refinement_blocks = nn.Sequential(*[
                Spiking_Residual_Block(dim=int(dim * 2 ** 0)) for _ in range(4)
            ])

        self.output = nn.Sequential(
            nn.Conv2d(in_channels=int(dim * 2 ** 0), out_channels=out_channels, kernel_size=3, stride=1, padding=1)
        )

    def forward(self, inp_img, return_aux=False):
        short = inp_img.clone()
        if len(inp_img.shape) < 5:
            inp_img = inp_img.unsqueeze(0).repeat(self.T, 1, 1, 1, 1)

        inp_enc_level1 = self.patch_embed(inp_img)
        out_enc_level1 = self.encoder_level1(inp_enc_level1)

        inp_enc_level2 = self.down1_2(out_enc_level1)
        out_enc_level2 = self.encoder_level2(inp_enc_level2)

        inp_enc_level3 = self.down2_3(out_enc_level2)
        out_enc_level3 = self.encoder_level3(inp_enc_level3)

        out_dec_level3 = self.decoder_level3(out_enc_level3)

        inp_dec_level2 = self.up3_2(out_dec_level3)
        inp_dec_level2 = self.skip_fusion_level2(inp_dec_level2, out_enc_level2)
        out_dec_level2 = self.decoder_level2(inp_dec_level2)

        inp_dec_level1 = self.up2_1(out_dec_level2)
        inp_dec_level1 = self.skip_fusion_level1(inp_dec_level1, out_enc_level1)
        out_dec_level1 = self.decoder_level1(inp_dec_level1)
        out_dec_level1 = self.additional_sunet_level1(out_dec_level1)

        if self.use_refinement:
            out_dec_level1 = self.refinement_blocks(out_dec_level1)

        main_out = self.output(out_dec_level1.mean(0)) + short

        if return_aux:
            H, W = short.shape[-2], short.shape[-1]
            aux2 = self.aux_head_level2(
                F.interpolate(out_enc_level2.mean(0), size=(H, W), mode="bilinear", align_corners=False)
            ) + short
            aux3 = self.aux_head_level3(
                F.interpolate(out_enc_level3.mean(0), size=(H, W), mode="bilinear", align_corners=False)
            ) + short
            return main_out, aux2, aux3

        return main_out


# ---------------------------------------------------------------------------
# Factory helpers
# ---------------------------------------------------------------------------
def build_vlifnet(dim=24, en_num_blocks=(2, 2, 4, 4), de_num_blocks=(2, 2, 2, 2),
                  T=4, use_refinement=False,
                  inp_channels=3, out_channels=3,
                  backend="torch") -> VLIFNet:
    """Preferred constructor — handles backend switch + does NOT force .cuda()."""
    set_vlifnet_backend(backend)
    net = VLIFNet(
        inp_channels=inp_channels,
        out_channels=out_channels,
        dim=dim,
        en_num_blocks=list(en_num_blocks),
        de_num_blocks=list(de_num_blocks),
        T=T,
        use_refinement=use_refinement,
    )
    # Force children that were instantiated before set_backend() to honour the
    # chosen backend as well.  This is a no-op on 'torch'.
    functional.set_backend(net, backend=backend)
    functional.set_step_mode(net, step_mode="m")
    return net


def model(use_refinement=False):
    """Upstream convenience wrapper — returns a cuda'd dim=48 net.  Kept for
    compatibility; v1 prefers build_vlifnet(dim=24, backend='torch')."""
    return VLIFNet(
        dim=48, en_num_blocks=[4, 4, 8, 8], de_num_blocks=[2, 2, 2, 2],
        T=4, use_refinement=use_refinement,
    ).cuda()
