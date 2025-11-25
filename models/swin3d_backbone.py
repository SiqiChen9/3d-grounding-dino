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
            (B, D', H', W', C)
        """
        x = self.proj(x)  # (B, C, D', H', W')
        x = x.permute(0, 2, 3, 4, 1)  # (B, D', H', W', C)
        x = self.norm(x)
        return x


class WindowAttention3D(nn.Module):
    """3D window-based multi-head self-attention."""
    
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
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, D*H*W, C)
        Returns:
            (B, D*H*W, C)
        """
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, num_heads, N, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Scaled dot-product attention
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)
        
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
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
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.mlp_ratio = mlp_ratio
        
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
        
        # Reshape for attention
        x_flat = x.view(B, D * H * W, C)
        
        # Window attention with residual
        x_flat = x_flat + self.attn(self.norm1(x_flat))
        
        # MLP with residual
        x_flat = x_flat + self.mlp(self.norm2(x_flat))
        
        # Reshape back
        x = x_flat.view(B, D, H, W, C)
        
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
        attn_drop_rate: float = 0.0
    ):
        super().__init__()
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.mlp_ratio = mlp_ratio
        
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
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate
                )
                for _ in range(depths[i_layer])
            ])
            self.layers.append(layer)
        
        # Patch merging layers
        self.downsample_layers = nn.ModuleList([
            PatchMerging3D(dim=int(embed_dim * 2 ** i))
            if i < self.num_layers - 1 else nn.Identity()
            for i in range(self.num_layers)
        ])
        
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, D, H, W)
        Returns:
            (B, D', H', W', C') - final stage features
        """
        # Patch embedding
        x = self.patch_embed(x)  # (B, D', H', W', C)
        
        # Apply Swin blocks
        for i in range(self.num_layers):
            # Apply blocks in this layer
            for block in self.layers[i]:
                x = block(x)
            
            # Downsample
            x = self.downsample_layers[i](x)
        
        return x
