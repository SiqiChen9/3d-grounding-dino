"""
Simplified 3D Swin Transformer backbone for CT volumes.
Implements 3D window attention and hierarchical feature extraction.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List
import numpy as np


class PatchEmbed3D(nn.Module):
    """3D patch embedding layer."""
    
    def __init__(
        self,
        patch_size: Tuple[int, int, int] = (4, 4, 4),
        in_channels: int = 1,
        embed_dim: int = 96
    ):
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        
        self.proj = nn.Conv3d(
            in_channels, embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )
        self.norm = nn.LayerNorm(embed_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, D, H, W)
        Returns:
            (B, D', H', W', embed_dim)
        """
        x = self.proj(x)  # (B, embed_dim, D', H', W')
        x = x.permute(0, 2, 3, 4, 1)  # (B, D', H', W', embed_dim)
        x = self.norm(x)
        return x


def window_partition(x: torch.Tensor, window_size: Tuple[int, int, int]) -> torch.Tensor:
    """
    Splits input into small non-overlapping 3D windows.
    Args:
        x: (B, D, H, W, C)
        window_size: (Wd, Wh, Ww)
    Returns:
        windows: (num_windows*B, Wd*Wh*Ww, C)
    """
    B, D, H, W, C = x.shape
    wd, wh, ww = window_size
    x = x.view(B, D // wd, wd, H // wh, wh, W // ww, ww, C)
    windows = x.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous().view(-1, wd * wh * ww, C)
    return windows


def window_reverse(windows: torch.Tensor, window_size: Tuple[int, int, int], B: int, D: int, H: int, W: int) -> torch.Tensor:
    """
    Reconstructs original tensor from windows.
    Args:
        windows: (num_windows*B, Wd*Wh*Ww, C)
        window_size: (Wd, Wh, Ww)
        B: Batch size of image
        D, H, W: Spatial dimensions of image
    Returns:
        x: (B, D, H, W, C)
    """
    wd, wh, ww = window_size
    x = windows.view(B, D // wd, H // wh, W // ww, wd, wh, ww, -1)
    x = x.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous().view(B, D, H, W, -1)
    return x


class WindowAttention3D(nn.Module):
    """
    3D window-based multi-head self-attention.
    """
    
    def __init__(
        self,
        dim: int,
        window_size: Tuple[int, int, int] = (7, 7, 7),
        num_heads: int = 8,
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0
    ):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        # Define relative position bias table
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1) * (2 * window_size[2] - 1), num_heads)
        )

        # Get pair-wise relative position index for each token inside the window
        coords_d = torch.arange(self.window_size[0])
        coords_h = torch.arange(self.window_size[1])
        coords_w = torch.arange(self.window_size[2])
        # indexing='ij' ensures (D, H, W) order
        coords = torch.stack(torch.meshgrid([coords_d, coords_h, coords_w], indexing='ij')) 
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        
        # Shift to start from 0
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 2] += self.window_size[2] - 1
        
        relative_coords[:, :, 0] *= (2 * self.window_size[1] - 1) * (2 * self.window_size[2] - 1)
        relative_coords[:, :, 1] *= (2 * self.window_size[2] - 1)
        
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)

        nn.init.trunc_normal_(self.relative_position_bias_table, std=.02)
    
    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            x: (B*num_windows, N, C)
            mask: (num_windows, N, N)
        Returns:
            (B*num_windows, N, C)
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        # Add relative position bias
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1] * self.window_size[2],
            self.window_size[0] * self.window_size[1] * self.window_size[2], -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = F.softmax(attn, dim=-1)
        else:
            attn = F.softmax(attn, dim=-1)

        attn = self.attn_drop(attn)
        
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        
        return x


class SwinTransformerBlock3D(nn.Module):
    """Swin Transformer block with window attention and MLP."""
    
    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size: Tuple[int, int, int] = (7, 7, 7),
        shift_size: Tuple[int, int, int] = (0, 0, 0),
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio

        # Validate shift size (D, H, W)
        assert 0 <= self.shift_size[0] < self.window_size[0], "shift_size must in 0-window_size"
        assert 0 <= self.shift_size[1] < self.window_size[1], "shift_size must in 0-window_size"
        assert 0 <= self.shift_size[2] < self.window_size[2], "shift_size must in 0-window_size"

        # Dynamic caching for attention mask (performance optimization)
        self.attn_mask = None
        self.last_input_shape = None

        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention3D(
            dim, window_size, num_heads,
            qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop
        )
        
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(drop)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, D, H, W, C)
        Returns:
            (B, D, H, W, C)
        """
        B, D, H, W, C = x.shape
        shortcut = x
        x = self.norm1(x)

        # Pad samples to be multiples of window size
        pad_d = (self.window_size[0] - D % self.window_size[0]) % self.window_size[0]
        pad_h = (self.window_size[1] - H % self.window_size[1]) % self.window_size[1]
        pad_w = (self.window_size[2] - W % self.window_size[2]) % self.window_size[2]
        x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h, 0, pad_d))
        _, Dp, Hp, Wp, _ = x.shape

        # Calculate attention mask for SW-MSA (with dynamic caching)
        if any(s > 0 for s in self.shift_size):
            current_shape = (Dp, Hp, Wp)
            
            # Check if we can reuse cached mask
            if self.attn_mask is not None and self.last_input_shape == current_shape:
                # Reuse cached mask (fast path)
                attn_mask = self.attn_mask
            else:
                # Compute new mask (slow path - only runs when input shape changes)
                img_mask = torch.zeros((1, Dp, Hp, Wp, 1), device=x.device)
                d_slices = (slice(0, -self.window_size[0]),
                            slice(-self.window_size[0], -self.shift_size[0]),
                            slice(-self.shift_size[0], None))
                h_slices = (slice(0, -self.window_size[1]),
                            slice(-self.window_size[1], -self.shift_size[1]),
                            slice(-self.shift_size[1], None))
                w_slices = (slice(0, -self.window_size[2]),
                            slice(-self.window_size[2], -self.shift_size[2]),
                            slice(-self.shift_size[2], None))
                cnt = 0
                for d in d_slices:
                    for h in h_slices:
                        for w in w_slices:
                            img_mask[:, d, h, w, :] = cnt
                            cnt += 1

                mask_windows = window_partition(img_mask, self.window_size)
                mask_windows = mask_windows.view(-1, self.window_size[0] * self.window_size[1] * self.window_size[2])
                attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
                attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
                
                # Cache the computed mask and shape
                self.attn_mask = attn_mask
                self.last_input_shape = current_shape
        else:
            attn_mask = None

        # Cyclic shift
        if any(s > 0 for s in self.shift_size):
            shifted_x = torch.roll(x, shifts=(-self.shift_size[0], -self.shift_size[1], -self.shift_size[2]), dims=(1, 2, 3))
        else:
            shifted_x = x

        # Partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # (nW*B, window_size, C)
        x_windows = x_windows.view(-1, self.window_size[0] * self.window_size[1] * self.window_size[2], C)  # (nW*B, N, C)

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=attn_mask)  # (nW*B, N, C)

        # Merge windows
        attn_windows = attn_windows.view(-1, self.window_size[0], self.window_size[1], self.window_size[2], C)
        shifted_x = window_reverse(attn_windows, self.window_size, B, Dp, Hp, Wp)  # (B, Dp, Hp, Wp, C)

        # Reverse cyclic shift
        if any(s > 0 for s in self.shift_size):
            x = torch.roll(shifted_x, shifts=(self.shift_size[0], self.shift_size[1], self.shift_size[2]), dims=(1, 2, 3))
        else:
            x = shifted_x

        # Remove padding
        if pad_d > 0 or pad_h > 0 or pad_w > 0:
            x = x[:, :D, :H, :W, :].contiguous()

        # FFN
        x = shortcut + x
        x = x + self.mlp(self.norm2(x))
        
        return x


class PatchMerging3D(nn.Module):
    """Downsample by merging patches (2x2x2)."""
    
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(8 * dim, 2 * dim, bias=False)
        self.norm = nn.LayerNorm(8 * dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, D, H, W, C)
        Returns:
            (B, D/2, H/2, W/2, 2*C)
        """
        B, D, H, W, C = x.shape
        
        # Pad if needed
        pad_d = (2 - D % 2) % 2
        pad_h = (2 - H % 2) % 2
        pad_w = (2 - W % 2) % 2
        
        if pad_d > 0 or pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h, 0, pad_d))
            D, H, W = D + pad_d, H + pad_h, W + pad_w
        
        # Merge 2x2x2 patches
        x0 = x[:, 0::2, 0::2, 0::2, :]  # (B, D/2, H/2, W/2, C)
        x1 = x[:, 1::2, 0::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, 0::2, :]
        x3 = x[:, 1::2, 1::2, 0::2, :]
        x4 = x[:, 0::2, 0::2, 1::2, :]
        x5 = x[:, 1::2, 0::2, 1::2, :]
        x6 = x[:, 0::2, 1::2, 1::2, :]
        x7 = x[:, 1::2, 1::2, 1::2, :]
        
        x = torch.cat([x0, x1, x2, x3, x4, x5, x6, x7], dim=-1)  # (B, D/2, H/2, W/2, 8*C)
        x = self.norm(x)
        x = self.reduction(x)  # (B, D/2, H/2, W/2, 2*C)
        
        return x


class SwinTransformer3D(nn.Module):
    """
    Simplified 3D Swin Transformer backbone.
    Outputs multi-scale features for detection.
    """
    
    def __init__(
        self,
        in_channels: int = 1,
        patch_size: Tuple[int, int, int] = (4, 4, 4),
        embed_dim: int = 96,
        depths: List[int] = [2, 2, 6, 2],
        num_heads: List[int] = [3, 6, 12, 24],
        window_size: Tuple[int, int, int] = (7, 7, 7),
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        out_channels: int = None,
        out_indices: Tuple[int, ...] = None # If None, only the last layer is returned
    ):
        super().__init__()
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.mlp_ratio = mlp_ratio
        # Default to the last stage if out_indices is not provided
        self.out_indices = out_indices if out_indices is not None else (self.num_layers - 1,)

        # Patch embedding
        self.patch_embed = PatchEmbed3D(patch_size, in_channels, embed_dim)

        # Build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = nn.ModuleList([
                SwinTransformerBlock3D(
                    dim=int(embed_dim * 2 ** i_layer),
                    num_heads=num_heads[i_layer],
                    window_size=window_size,
                    shift_size=(0, 0, 0) if (i % 2 == 0) else (window_size[0] // 2, window_size[1] // 2, window_size[2] // 2),
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate
                )
                for i in range(depths[i_layer])
            ])
            self.layers.append(layer)

        # Patch merging layers
        self.downsample_layers = nn.ModuleList([
            PatchMerging3D(dim=int(embed_dim * 2 ** i))
            if i < self.num_layers - 1 else nn.Identity()
            for i in range(self.num_layers)
        ])

        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))

        # Output projection layer (e.g., 768 -> 256)
        if out_channels is not None and out_channels != self.num_features:
            self.output_proj = nn.Linear(self.num_features, out_channels)
        else:
            self.output_proj = None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, D, H, W)
        Returns:
            (B, D', H', W', C') - final stage features
            OR
            Tuple of features if out_indices has multiple values
        """
        # Patch embedding
        x = self.patch_embed(x)  # (B, D', H', W', C)

        outs = []
        # Apply Swin blocks
        for i in range(self.num_layers):
            # Apply blocks in this layer
            for block in self.layers[i]:
                x = block(x)

            # Record output if in out_indices
            if i in self.out_indices:
                outs.append(x)

            # Downsample
            x = self.downsample_layers[i](x)

        # Compatibility mode: if only requesting the last layer (default behavior)
        if self.out_indices == (self.num_layers - 1,):
            out = outs[0]
            # Apply output projection if specified
            if self.output_proj is not None:
                out = self.output_proj(out)  # (B, D', H', W', out_channels)
            return out
        
        return tuple(outs)
