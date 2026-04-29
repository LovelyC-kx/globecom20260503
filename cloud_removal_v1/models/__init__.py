"""VLIFNet model package — vendored from system1_VLIFNet."""

from .vlifnet import (
    VLIFNet,
    build_vlifnet,
    set_vlifnet_backend,
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
)

__all__ = [
    "VLIFNet",
    "build_vlifnet",
    "set_vlifnet_backend",
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
]
