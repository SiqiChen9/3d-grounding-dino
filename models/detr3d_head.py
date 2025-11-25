"""
3D DETR detection head with transformer encoder-decoder.
Predicts 3D bounding boxes and class labels.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import math


class PositionalEncoding3D(nn.Module):
    """Learnable 3D positional embeddings."""
    
    def __init__(self, d_model: int, max_len: int = 1000):
        super().__init__()
        self.d_model = d_model
        self.pos_embed = nn.Parameter(torch.randn(1, max_len, d_model))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, N, C)
        Returns:
            (B, N, C) - with positional encoding added
        """
        B, N, C = x.shape
        return x + self.pos_embed[:, :N, :]


class TransformerEncoderLayer(nn.Module):
    """Transformer encoder layer with self-attention and FFN."""
    
    def __init__(
        self,
        d_model: int = 256,
        num_heads: int = 8,
        dim_feedforward: int = 2048,
        dropout: float = 0.1
    ):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout)
        
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
    
    def forward(self, src: torch.Tensor) -> torch.Tensor:
        """
        Args:
            src: (N, B, C)
        Returns:
            (N, B, C)
        """
        # Self-attention
        src2 = self.self_attn(src, src, src)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        
        # FFN
        src2 = self.linear2(self.dropout(F.relu(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        
        return src


class TransformerDecoderLayer(nn.Module):
    """Transformer decoder layer with cross-attention."""
    
    def __init__(
        self,
        d_model: int = 256,
        num_heads: int = 8,
        dim_feedforward: int = 2048,
        dropout: float = 0.1
    ):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout)
        self.cross_attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout)
        
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
    
    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            tgt: (M, B, C) - queries
            memory: (N, B, C) - encoder output
        Returns:
            (M, B, C)
        """
        # Self-attention on queries
        tgt2 = self.self_attn(tgt, tgt, tgt)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        
        # Cross-attention to encoder features
        tgt2 = self.cross_attn(tgt, memory, memory)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        
        # FFN
        tgt2 = self.linear2(self.dropout(F.relu(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        
        return tgt


class DETR3DHead(nn.Module):
    """
    3D DETR detection head.
    Transformer encoder-decoder with object queries for 3D box prediction.
    """
    
    def __init__(
        self,
        hidden_dim: int = 256,
        num_queries: int = 100,
        num_classes: int = 5,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
        num_heads: int = 8,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        backbone_dim: int = 768,  # Output dim from Swin3D
        use_grounding_fusion: bool = False  # NEW: Enable grounding fusion
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_queries = num_queries
        self.num_classes = num_classes
        self.use_grounding_fusion = use_grounding_fusion
        
        # Project backbone features to transformer dimension
        self.input_proj = nn.Conv3d(backbone_dim, hidden_dim, kernel_size=1)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding3D(hidden_dim)
        
        # Transformer encoder
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(hidden_dim, num_heads, dim_feedforward, dropout)
            for _ in range(num_encoder_layers)
        ])
        
        # Learnable object queries
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        
        # Grounding fusion layer (optional)
        if use_grounding_fusion:
            self.grounding_fusion = nn.MultiheadAttention(
                hidden_dim, num_heads, dropout=dropout
            )
            self.fusion_norm = nn.LayerNorm(hidden_dim)
            self.fusion_dropout = nn.Dropout(dropout)
        
        # Transformer decoder
        self.decoder_layers = nn.ModuleList([
            TransformerDecoderLayer(hidden_dim, num_heads, dim_feedforward, dropout)
            for _ in range(num_decoder_layers)
        ])
        
        # Prediction heads
        self.class_embed = nn.Linear(hidden_dim, num_classes + 1)  # +1 for background
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 6, 3)  # 6D boxes: (cx,cy,cz,w,h,d)
    
    def forward(
        self, 
        features: torch.Tensor,
        class_tokens: torch.Tensor = None  # NEW: (B, num_classes, hidden_dim)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            features: (B, C, D, H, W) from backbone
            class_tokens: Optional (B, num_classes, hidden_dim) from grounding module
        
        Returns:
            pred_logits: (B, num_queries, num_classes+1)
            pred_boxes: (B, num_queries, 6)
        """
        B = features.shape[0]
        
        # Project features
        features = self.input_proj(features)  # (B, hidden_dim, D, H, W)
        
        # Flatten spatial dimensions
        D, H, W = features.shape[2:]
        features_flat = features.flatten(2).permute(2, 0, 1)  # (D*H*W, B, hidden_dim)
        
        # Add positional encoding
        features_flat = features_flat.permute(1, 0, 2)  # (B, D*H*W, hidden_dim)
        features_flat = self.pos_encoding(features_flat)
        features_flat = features_flat.permute(1, 0, 2)  # (D*H*W, B, hidden_dim)
        
        # Encoder
        memory = features_flat
        for layer in self.encoder_layers:
            memory = layer(memory)
        
        # Decoder with queries
        query_embed = self.query_embed.weight.unsqueeze(1).repeat(1, B, 1)  # (num_queries, B, hidden_dim)
        
        # FIXED: Initialize with query_embed instead of zeros
        tgt = query_embed.clone()
        
        # NEW: Fuse with class tokens before decoder (if available)
        if self.use_grounding_fusion and class_tokens is not None:
            # Reshape class_tokens: (B, num_classes, hidden_dim) -> (num_classes, B, hidden_dim)
            class_tokens_t = class_tokens.permute(1, 0, 2)
            
            # Cross-attention: queries attend to class tokens
            tgt2, _ = self.grounding_fusion(tgt, class_tokens_t, class_tokens_t)
            tgt = tgt + self.fusion_dropout(tgt2)
            tgt = tgt.permute(1, 0, 2)  # (B, num_queries, hidden_dim)
            tgt = self.fusion_norm(tgt)
            tgt = tgt.permute(1, 0, 2)  # (num_queries, B, hidden_dim)
        
        for layer in self.decoder_layers:
            tgt = layer(tgt, memory)
        
        # Predictions
        tgt = tgt.permute(1, 0, 2)  # (B, num_queries, hidden_dim)
        pred_logits = self.class_embed(tgt)  # (B, num_queries, num_classes+1)
        pred_boxes = self.bbox_embed(tgt).sigmoid()  # (B, num_queries, 6), normalized to [0,1]
        
        return pred_logits, pred_boxes


class MLP(nn.Module):
    """Simple MLP for box regression."""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int
    ):
        super().__init__()
        layers = []
        for i in range(num_layers):
            if i == 0:
                layers.append(nn.Linear(input_dim, hidden_dim))
            elif i == num_layers - 1:
                layers.append(nn.Linear(hidden_dim, output_dim))
            else:
                layers.append(nn.Linear(hidden_dim, hidden_dim))
            
            if i < num_layers - 1:
                layers.append(nn.ReLU(inplace=True))
        
        self.layers = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)
