"""
VLIFNet
=======
Spiking U-Net for image restoration, ported from
https://github.com/LovelyC-kx/qun_bme/tree/main/system1_VLIFNet

Exports:
    VLIFNet          : the full U-Net
    build_vlifnet    : factory (dim / blocks / T / backend)
    MultiSpike4      : 4-level quantised spike (default neuron)
    NoisySpikeWrapper: drop-in surrogate swap for v2 ablation (NOT USED IN v1)

Usage (v1, matches VLIFNet original training defaults):

    from Spiking_Models.VLIFNet import build_vlifnet
    model = build_vlifnet(dim=24,
                          en_num_blocks=[2, 2, 4, 4],
                          de_num_blocks=[2, 2, 2, 2],
                          T=4,
                          backend='torch')   # 'cupy' optional on supported CUDA

Note
----
`backend='torch'` is the v1 default because cupy wheels sometimes fail
to build; accuracy is identical, speed is ~20–30% slower on V100.
"""

from .model import (
    VLIFNet,
    build_vlifnet,
    MultiSpike4,
    mem_update,
    Spiking_Residual_Block,
    SUNet_Level1_Block,
    GatedSkipFusion,
    DownSampling,
    UpSampling,
    OverlapPatchEmbed,
    TimeAttention,
    PixelShuffleLIFBlock,
    set_vlifnet_backend,
)

__all__ = [
    "VLIFNet",
    "build_vlifnet",
    "MultiSpike4",
    "mem_update",
    "Spiking_Residual_Block",
    "SUNet_Level1_Block",
    "GatedSkipFusion",
    "DownSampling",
    "UpSampling",
    "OverlapPatchEmbed",
    "TimeAttention",
    "PixelShuffleLIFBlock",
    "set_vlifnet_backend",
]
