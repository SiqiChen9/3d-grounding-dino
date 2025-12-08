"""
Feature Enhancer Layer - Full Implementation.
Corresponds to "2. A Feature Enhancer Layer" in the architecture diagram.

Implements bidirectional cross-attention between text and image modalities:
  1. Self-Attention (for both modalities)
  2. Image-to-Text Cross-Attention
  3. Text-to-Image Cross-Attention
  4. Feed-Forward Networks (FFN)
  5. Residual connections + Layer normalization
"""
import torch
import torch.nn as nn
from typing import Tuple


class FeatureEnhancerLayer(nn.Module):
    """
    Single Feature Enhancer block with self-attention, bidirectional cross-attention,
    and feed-forward networks.
    
    Implements the following pipeline:
    1. Self-Attention for image features
    2. Self-Attention for text features
    3. Image-to-Text Cross-Attention (Image queries, Text keys/values)
    4. Text-to-Image Cross-Attention (Text queries, Image keys/values)
    5. FFN for image features
    6. FFN for text features
    
    With residual connections and layer normalization after each sub-layer.
    """
    
    def __init__(
        self,
        hidden_dim: int = 256,
        num_heads: int = 8,
        dropout: float = 0.1
    ):
        """
        Args:
            hidden_dim: Hidden dimension for features
            num_heads: Number of attention heads
            dropout: Dropout rate
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.dropout_rate = dropout
        
        # Image self-attention
        self.image_self_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=False  # (N, B, C)
        )
        self.image_self_attn_norm = nn.LayerNorm(hidden_dim)
        
        # Text self-attention
        self.text_self_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=False  # (num_classes, B, C)
        )
        self.text_self_attn_norm = nn.LayerNorm(hidden_dim)
        
        # Image-to-Text Cross-Attention (Image queries, Text keys/values)
        self.image_to_text_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=False
        )
        self.image_to_text_attn_norm = nn.LayerNorm(hidden_dim)
        
        # Text-to-Image Cross-Attention (Text queries, Image keys/values)
        self.text_to_image_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=False
        )
        self.text_to_image_attn_norm = nn.LayerNorm(hidden_dim)
        
        # Feed-Forward Networks
        ffn_hidden_dim = hidden_dim * 4
        self.image_ffn = nn.Sequential(
            nn.Linear(hidden_dim, ffn_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_hidden_dim, hidden_dim)
        )
        self.image_ffn_norm = nn.LayerNorm(hidden_dim)
        
        self.text_ffn = nn.Sequential(
            nn.Linear(hidden_dim, ffn_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_hidden_dim, hidden_dim)
        )
        self.text_ffn_norm = nn.LayerNorm(hidden_dim)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        text_features: torch.Tensor,
        image_features: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with residual connections.
        
        Args:
            text_features: (B, num_classes, hidden_dim)
            image_features: (N, B, hidden_dim)
        
        Returns:
            enhanced_text_features: (B, num_classes, hidden_dim)
            enhanced_image_features: (N, B, hidden_dim)
        """
        # Transpose for MultiheadAttention (expects (seq_len, batch, dim))
        # text_features: (B, num_classes, C) -> (num_classes, B, C)
        text_features_t = text_features.transpose(0, 1)
        # image_features: already (N, B, C)
        
        # ============ Self-Attention ============
        # Image self-attention
        image_attn_out, _ = self.image_self_attn(
            image_features, image_features, image_features
        )
        image_features = image_features + self.dropout(image_attn_out)
        image_features = self.image_self_attn_norm(image_features)
        
        # Text self-attention
        text_attn_out, _ = self.text_self_attn(
            text_features_t, text_features_t, text_features_t
        )
        text_features_t = text_features_t + self.dropout(text_attn_out)
        text_features_t = self.text_self_attn_norm(text_features_t)
        
        # ============ Cross-Attention ============
        # Image-to-Text Cross-Attention (Image queries, Text keys/values)
        image_to_text_out, _ = self.image_to_text_attn(
            query=image_features,
            key=text_features_t,
            value=text_features_t
        )
        image_features = image_features + self.dropout(image_to_text_out)
        image_features = self.image_to_text_attn_norm(image_features)
        
        # Text-to-Image Cross-Attention (Text queries, Image keys/values)
        text_to_image_out, _ = self.text_to_image_attn(
            query=text_features_t,
            key=image_features,
            value=image_features
        )
        text_features_t = text_features_t + self.dropout(text_to_image_out)
        text_features_t = self.text_to_image_attn_norm(text_features_t)
        
        # ============ Feed-Forward Networks ============
        # Image FFN
        image_ffn_out = self.image_ffn(image_features)
        image_features = image_features + self.dropout(image_ffn_out)
        image_features = self.image_ffn_norm(image_features)
        
        # Text FFN
        text_ffn_out = self.text_ffn(text_features_t)
        text_features_t = text_features_t + self.dropout(text_ffn_out)
        text_features_t = self.text_ffn_norm(text_features_t)
        
        # Transpose back
        text_features = text_features_t.transpose(0, 1)
        
        return text_features, image_features


class FeatureEnhancer(nn.Module):
    """
    Feature Enhancer - Stacks multiple FeatureEnhancerLayers.
    
    This module enhances text and image features through bidirectional cross-attention.
    It applies self-attention to both modalities, enables image-to-text and text-to-image
    cross-attention, and uses feed-forward networks for feature refinement.
    
    Architecture:
    ┌─────────────────────────────────────────────────────┐
    │ 1. Self-Attention (for both modalities)             │
    │    - Separate self-attention for text and image     │
    ├─────────────────────────────────────────────────────┤
    │ 2. Image-to-Text Cross-Attention                    │
    │    - Query: Image Features                          │
    │    - Key/Value: Text Features                       │
    │    - Output: Updated Image Features                 │
    ├─────────────────────────────────────────────────────┤
    │ 3. Text-to-Image Cross-Attention                    │
    │    - Query: Text Features                           │
    │    - Key/Value: Image Features                      │
    │    - Output: Updated Text Features                  │
    ├─────────────────────────────────────────────────────┤
    │ 4. Feed-Forward Networks                            │
    │    - FFN for text features                          │
    │    - FFN for image features                         │
    └─────────────────────────────────────────────────────┘
    """
    
    def __init__(
        self,
        hidden_dim: int = 256,
        num_heads: int = 8,
        dropout: float = 0.1,
        num_layers: int = 1
    ):
        """
        Args:
            hidden_dim: Hidden dimension for features (default: 256)
            num_heads: Number of attention heads (default: 8)
            dropout: Dropout rate (default: 0.1)
            num_layers: Number of enhancer layers to stack (default: 1)
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.num_layers = num_layers
        
        # Stack multiple enhancer layers
        self.layers = nn.ModuleList([
            FeatureEnhancerLayer(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                dropout=dropout
            )
            for _ in range(num_layers)
        ])
    
    def forward(
        self,
        text_features: torch.Tensor,
        image_features: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Enhance text and image features through bidirectional cross-attention.
        
        Args:
            text_features: (B, num_classes, hidden_dim) - pseudo text embeddings
            image_features: (N, B, hidden_dim) - flattened spatial image features
                where N = spatial_tokens (e.g., 32 for 64x64x64 volume with 16x downsampling)
                B = batch size
                hidden_dim = feature dimension (256)
        
        Returns:
            enhanced_text_features: (B, num_classes, hidden_dim)
            enhanced_image_features: (N, B, hidden_dim)
        """
        # Apply stacked enhancer layers
        for layer in self.layers:
            text_features, image_features = layer(text_features, image_features)
        
        return text_features, image_features
