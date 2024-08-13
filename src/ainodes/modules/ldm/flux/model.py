#Original code can be found on: https://github.com/black-forest-labs/flux

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from..common_dit import pad_to_patch_size
from .layers import (
    DoubleStreamBlock,
    EmbedND,
    LastLayer,
    MLPEmbedder,
    SingleStreamBlock,
    timestep_embedding,
)

#from einops import rearrange, repeat

@dataclass
class FluxParams:
    in_channels: int
    vec_in_dim: int
    context_in_dim: int
    hidden_size: int
    mlp_ratio: float
    num_heads: int
    depth: int
    depth_single_blocks: int
    axes_dim: list
    theta: int
    qkv_bias: bool
    guidance_embed: bool


class Flux(nn.Module):
    """
    Transformer model for flow matching on sequences.
    """

    def __init__(self, image_model=None, dtype=None, device=None, operations=None, **kwargs):
        super().__init__()
        self.dtype = dtype
        params = FluxParams(**kwargs)
        self.params = params
        self.in_channels = params.in_channels * 2 * 2
        self.out_channels = self.in_channels
        if params.hidden_size % params.num_heads != 0:
            raise ValueError(
                f"Hidden size {params.hidden_size} must be divisible by num_heads {params.num_heads}"
            )
        pe_dim = params.hidden_size // params.num_heads
        if sum(params.axes_dim) != pe_dim:
            raise ValueError(f"Got {params.axes_dim} but expected positional dim {pe_dim}")
        self.hidden_size = params.hidden_size
        self.num_heads = params.num_heads
        self.pe_embedder = EmbedND(dim=pe_dim, theta=params.theta, axes_dim=params.axes_dim)
        self.img_in = operations.Linear(self.in_channels, self.hidden_size, bias=True, dtype=dtype, device=device)
        self.time_in = MLPEmbedder(in_dim=256, hidden_dim=self.hidden_size, dtype=dtype, device=device, operations=operations)
        self.vector_in = MLPEmbedder(params.vec_in_dim, self.hidden_size, dtype=dtype, device=device, operations=operations)
        self.guidance_in = (
            MLPEmbedder(in_dim=256, hidden_dim=self.hidden_size, dtype=dtype, device=device, operations=operations) if params.guidance_embed else nn.Identity()
        )
        self.txt_in = operations.Linear(params.context_in_dim, self.hidden_size, dtype=dtype, device=device)

        self.double_blocks = nn.ModuleList(
            [
                DoubleStreamBlock(
                    self.hidden_size,
                    self.num_heads,
                    mlp_ratio=params.mlp_ratio,
                    qkv_bias=params.qkv_bias,
                    dtype=dtype, device=device, operations=operations
                )
                for _ in range(params.depth)
            ]
        )

        self.single_blocks = nn.ModuleList(
            [
                SingleStreamBlock(self.hidden_size, self.num_heads, mlp_ratio=params.mlp_ratio, dtype=dtype, device=device, operations=operations)
                for _ in range(params.depth_single_blocks)
            ]
        )

        self.final_layer = LastLayer(self.hidden_size, 1, self.out_channels, dtype=dtype, device=device, operations=operations)

    def forward_orig(
        self,
        img: Tensor,
        img_ids: Tensor,
        txt: Tensor,
        txt_ids: Tensor,
        timesteps: Tensor,
        y: Tensor,
        guidance: Tensor = None,
    ) -> Tensor:
        if img.ndim != 3 or txt.ndim != 3:
            raise ValueError("Input img and txt tensors must have 3 dimensions.")

        # running on sequences img
        img = self.img_in(img)
        vec = self.time_in(timestep_embedding(timesteps, 256).to(img.dtype))
        if self.params.guidance_embed:
            if guidance is None:
                raise ValueError("Didn't get guidance strength for guidance distilled model.")
            vec = vec + self.guidance_in(timestep_embedding(guidance, 256).to(img.dtype))

        vec = vec + self.vector_in(y)
        txt = self.txt_in(txt)

        ids = torch.cat((txt_ids, img_ids), dim=1)
        pe = self.pe_embedder(ids)

        for block in self.double_blocks:
            img, txt = block(img=img, txt=txt, vec=vec, pe=pe)

        img = torch.cat((txt, img), 1)
        for block in self.single_blocks:
            img = block(img, vec=vec, pe=pe)
        img = img[:, txt.shape[1] :, ...]

        img = self.final_layer(img, vec)  # (N, T, patch_size ** 2 * out_channels)
        return img
    def forward(self, x, timestep, context, y, guidance, *args, **kwargs):
        # print(x.shape, timestep.shape, context.shape, y.shape, guidance.shape)

        patch_size = 2

        bs, c, h, w = x.shape

        # Calculate padding to ensure dimensions are multiples of patch_size
        pad_h = (patch_size - h % patch_size) % patch_size
        pad_w = (patch_size - w % patch_size) % patch_size

        # Pad the input if necessary
        if pad_h > 0 or pad_w > 0:
            x = torch.nn.functional.pad(x, (0, pad_w, 0, pad_h))

        # Unfold x into patches
        x_patched = x.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)

        b, c, h_p, w_p, ph, pw = x_patched.size()

        # Reshape and permute x_patched to get img
        img = x_patched.permute(0, 2, 3, 1, 4, 5).reshape(b, h_p * w_p, -1)

        # Generate img_ids
        if isinstance(h_p, torch.Tensor):
            h_len, w_len = h_p.to(x.device), w_p.to(x.device)
        else:
            h_len, w_len = h_p, w_p
        h_ids = torch.arange(h_len, device=x.device, dtype=x.dtype).repeat_interleave(w_len)
        w_ids = torch.arange(w_len, device=x.device, dtype=x.dtype).repeat(h_len)
        img_ids = torch.stack((torch.zeros_like(h_ids), h_ids, w_ids), dim=-1).unsqueeze(0).repeat(bs, 1, 1)

        # Generate txt_ids (zeros as placeholder, if needed)
        txt_ids = torch.zeros((bs, context.shape[1], 3), device=x.device, dtype=x.dtype)

        # Forward through the original function
        out = self.forward_orig(img, img_ids, context, txt_ids, timestep, y, guidance)

        # Reshape the output back to the original format and remove padding
        out = out.view(bs, h_len, w_len, c, ph, pw).permute(0, 3, 1, 4, 2, 5).reshape(bs, c, h_len * ph, w_len * pw)
        return out[:, :, :h, :w]

    # def forward(self, x, timestep, context, y, guidance, **kwargs):
    #     bs, c, h, w = x.shape
    #     patch_size = 2
    #     pad_h = (patch_size - h % 2) % patch_size
    #     pad_w = (patch_size - w % 2) % patch_size
    #     x = pad_to_patch_size(x, (patch_size, patch_size))
    #     # Reshape and permute x to get patches
    #     x_patched = x.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
    #     b, c, h_p, w_p, ph, pw = x_patched.size()
    #     img = x_patched.permute(0, 2, 3, 1, 4, 5).contiguous().view(b, h_p * w_p, c * ph * pw)
    #     # Generate img_ids
    #     h_len = (h + pad_h) // patch_size
    #     w_len = (w + pad_w) // patch_size
    #     img_ids = torch.zeros((bs, h_len * w_len, 3), device=x.device, dtype=x.dtype)
    #     h_ids = torch.linspace(0, h_len - 1, steps=h_len, device=x.device, dtype=x.dtype)
    #     w_ids = torch.linspace(0, w_len - 1, steps=w_len, device=x.device, dtype=x.dtype)
    #     img_ids[:, :, 1] = h_ids.repeat_interleave(w_len)
    #     img_ids[:, :, 2] = w_ids.repeat(h_len)
    #     # Generate txt_ids
    #     txt_ids = torch.zeros((bs, context.shape[1], 3), device=x.device, dtype=x.dtype)
    #     # Forward through the original function
    #     out = self.forward_orig(img, img_ids, context, txt_ids, timestep, y, guidance)
    #     # Reshape the output back to the original format
    #     out = out.view(bs, h_len, w_len, c, ph, pw).permute(0, 3, 1, 4, 2, 5).contiguous()
    #     out = out.view(bs, c, h_len * ph, w_len * pw)
    #     # Remove the padding if it was added
    #     return out[:, :, :h, :w]
