"""
Cross-Modality Decoder.
Corresponds to "3. A Decoder Layer" in the architecture diagram.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
import math


class PositionalEncoding3D(nn.Module):
    """Learnable 3D positional embeddings for spatial features."""
    
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


class CrossModalityDecoderLayer(nn.Module):
    """
    Cross-Modality Decoder Layer as shown in the architecture diagram.
    
    Architecture:
    ┌─────────────────────────────────────────────────────┐
    │ 1. Self-Attention                                   │
    │    - Query self-attention on object queries         │
    │    - Input: queries (Q, K, V from queries)          │
    ├─────────────────────────────────────────────────────┤
    │ 2. Image Cross-Attention                            │
    │    - Queries attend to image features               │
    │    - Q from queries, K/V from image features        │
    ├─────────────────────────────────────────────────────┤
    │ 3. Text Cross-Attention                             │
    │    - Queries attend to text features                │
    │    - Q from queries, K/V from text features         │
    ├─────────────────────────────────────────────────────┤
    │ 4. Feed-Forward Network (FFN)                       │
    │    - MLP with ReLU activation                       │
    └─────────────────────────────────────────────────────┘
    
    Each operation has residual connection + layer normalization.
    """
    
    def __init__(
        self,
        d_model: int = 256,
        num_heads: int = 8,
        dim_feedforward: int = 2048,
        dropout: float = 0.1
    ):
        super().__init__()
        
        # Self-attention on queries
        self.self_attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout)
        
        # Image cross-attention (Q from queries, K/V from image)
        self.image_cross_attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout)
        
        # Text cross-attention (Q from queries, K/V from text)
        self.text_cross_attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout)
        
        # Feed-forward network
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model)  # After self-attention
        self.norm2 = nn.LayerNorm(d_model)  # After text cross-attention
        self.norm3 = nn.LayerNorm(d_model)  # After image cross-attention
        self.norm4 = nn.LayerNorm(d_model)  # After FFN
        
        # Dropout layers
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.dropout4 = nn.Dropout(dropout)
    
    def forward(
        self,
        queries: torch.Tensor,
        text_features: torch.Tensor,
        image_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass through one decoder layer.
        
        Args:
            queries: (num_queries, B, hidden_dim) - object queries
            text_features: (num_classes, B, hidden_dim) - enhanced text features
            image_features: (N, B, hidden_dim) - enhanced image features
        
        Returns:
            updated_queries: (num_queries, B, hidden_dim)
        """
        # 1. Self-attention on queries
        queries2 = self.self_attn(queries, queries, queries)[0]
        queries = queries + self.dropout1(queries2)
        queries = self.norm1(queries)
        
        # 2. Image cross-attention (queries attend to image)
        queries2 = self.image_cross_attn(
            query=queries,
            key=image_features,
            value=image_features
        )[0]
        queries = queries + self.dropout2(queries2)
        queries = self.norm2(queries)
        
        # 3. Text cross-attention (queries attend to text)
        queries2 = self.text_cross_attn(
            query=queries,
            key=text_features,
            value=text_features
        )[0]
        queries = queries + self.dropout3(queries2)
        queries = self.norm3(queries)
        
        # 4. Feed-forward network
        queries2 = self.linear2(self.dropout(F.relu(self.linear1(queries))))
        queries = queries + self.dropout4(queries2)
        queries = self.norm4(queries)
        
        return queries


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


class CrossModalityDecoder(nn.Module):
    """
    Cross-Modality Decoder for 3D object detection.
    
    This decoder receives enhanced features from FeatureEnhancer and performs
    cross-attention between queries and both text/image modalities.
    
    Components:
        1. Stack of CrossModalityDecoderLayer
        2. Classification head
        3. Bounding box regression head
    
    Note: Image preprocessing (projection, encoding) is handled by FeatureEnhancer,
          so this decoder works directly with enhanced features.
    """
    
    def __init__(
        self,
        hidden_dim: int = 256,
        num_queries: int = 100,
        num_classes: int = 5,
        num_decoder_layers: int = 6,
        num_heads: int = 8,
        dim_feedforward: int = 2048,
        dropout: float = 0.1
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_queries = num_queries
        self.num_classes = num_classes
        
        # Cross-modality decoder layers
        self.decoder_layers = nn.ModuleList([
            CrossModalityDecoderLayer(hidden_dim, num_heads, dim_feedforward, dropout)
            for _ in range(num_decoder_layers)
        ])
        
        # Prediction heads
        self.class_embed = nn.Linear(hidden_dim, num_classes + 1)  # +1 for background
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 6, 3)  # 6D boxes: (cx,cy,cz,w,h,d)
    
    def forward(
        self,
        image_features: torch.Tensor,
        text_features: torch.Tensor,
        queries: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the cross-modality decoder.
        
        Args:
            image_features: (N, B, hidden_dim) - enhanced image features (flattened spatial)
            text_features: (B, num_classes, hidden_dim) - enhanced text features
            queries: (num_queries, B, hidden_dim) - object queries from query selection
        
        Returns:
            pred_logits: (B, num_queries, num_classes+1) - classification scores
            pred_boxes: (B, num_queries, 6) - bounding box predictions (normalized)
        """
        # Prepare text features for cross-attention
        # Convert from (B, num_classes, hidden_dim) to (num_classes, B, hidden_dim)
        text_features_t = text_features.permute(1, 0, 2)
        
        # Decode with cross-modality attention
        # image_features is already in shape (N, B, hidden_dim) from FeatureEnhancer
        tgt = queries  # (num_queries, B, hidden_dim)
        for layer in self.decoder_layers:
            tgt = layer(tgt, text_features_t, image_features)
        
        # Generate predictions
        tgt = tgt.permute(1, 0, 2)  # (B, num_queries, hidden_dim)
        pred_logits = self.class_embed(tgt)  # (B, num_queries, num_classes+1)
        pred_boxes = self.bbox_embed(tgt).sigmoid()  # (B, num_queries, 6), normalized to [0,1]
        
        return pred_logits, pred_boxes
