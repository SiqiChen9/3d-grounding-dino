"""
Complete 3D Grounding-DETR model.
Integrates all components following the architecture diagram.
"""
import torch
import torch.nn as nn
from typing import Dict

from .swin3d_backbone import SwinTransformer3D
from .text_feature_generator import PseudoTextFeatureGenerator
from .feature_enhancer import FeatureEnhancer
from .query_selection import LanguageGuidedQuerySelection
from .cross_modality_decoder import CrossModalityDecoder


class GroundingDETR3D(nn.Module):
    """
    3D Grounding-DETR for CT volume detection.
    
    Architecture (following diagram):
    ┌─────────────────────────────────────────────────────┐
    │ 1. Model Overall                                    │
    │    ┌─────────────┐         ┌──────────────────┐     │
    │    │ Image       │         │ Pseudo Text      │     │
    │    │ Backbone    │         │ Feature Gen      │     │
    │    │ (Swin3D)    │         │                  │     │
    │    └──────┬──────┘         └────────┬─────────┘     │
    │           │                         │               │
    │           │ Vanilla Features        │               │
    │           ▼                         ▼               │
    │    ┌──────────────────────────────────────────┐     │
    │    │ 2. Feature Enhancer (TODO)               │     │
    │    │    - Bidirectional cross-attention       │     │
    │    └──────┬───────────────────────┬───────────┘     │
    │           │                       │                 │
    │           │ Enhanced Features     │                 │
    │           ▼                       │                 │
    │    ┌──────────────┐               │                 │
    │    │ Language-    │◀──────────────┘                 │
    │    │ guided Query │                                 │
    │    │ Selection    │                                 │
    │    └──────┬───────┘                                 │
    │           │ Selected Queries                        │
    │           ▼                                         │
    │    ┌─────────────────────────────────────────┐      │
    │    │ 3. Cross-Modality Decoder               │      │
    │    │    - Self-Attention                     │      │
    │    │    - Text Cross-Attention               │      │
    │    │    - Image Cross-Attention              │      │
    │    │    - FFN                                │      │
    │    └──────┬──────────────────────────────────┘      │
    │           │                                         │
    │           ▼                                         │
    │    ┌─────────────────┐                              │
    │    │ Prediction Heads│                              │
    │    │ - Class         │                              │
    │    │ - BBox          │                              │
    │    └─────────────────┘                              │
    └─────────────────────────────────────────────────────┘
    """
    
    def __init__(
        self,
        num_classes: int = 5,
        num_queries: int = 100,
        hidden_dim: int = 256,
        # Backbone parameters
        backbone_embed_dim: int = 96,
        backbone_depths: list = [2, 2, 6, 2],
        backbone_num_heads: list = [3, 6, 12, 24],
        backbone_window_size: tuple = (7, 7, 7),
        backbone_out_indices: tuple = None,  # None = auto multi-scale (last 2 stages)
        # Decoder parameters
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
        num_heads: int = 8,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        # Text feature parameters
        trainable_pseudo_features: bool = True
    ):
        super().__init__()
        self.num_classes = num_classes
        self.num_queries = num_queries
        self.hidden_dim = hidden_dim
        
        # Determine backbone output indices
        # Default: last 2 stages for multi-scale (e.g., (2, 3) for 4-stage backbone)
        num_stages = len(backbone_depths)
        if backbone_out_indices is None:
            if num_stages >= 2:
                backbone_out_indices = tuple(range(num_stages - 2, num_stages))
            else:
                backbone_out_indices = (num_stages - 1,)
        self.backbone_out_indices = backbone_out_indices
        self.multi_scale = len(backbone_out_indices) > 1
        
        # ═══════════════════════════════════════════════════════
        # Component 1: Image Backbone (Swin3D) with multi-scale output
        # ═══════════════════════════════════════════════════════
        self.image_backbone = SwinTransformer3D(
            in_channels=1,
            patch_size=(4, 4, 4),
            embed_dim=backbone_embed_dim,
            depths=backbone_depths,
            num_heads=backbone_num_heads,
            window_size=backbone_window_size,
            mlp_ratio=4.0,
            qkv_bias=True,
            drop_rate=dropout,
            attn_drop_rate=dropout,
            out_channels=hidden_dim,  # Each scale projected to hidden_dim
            out_indices=backbone_out_indices
        )
        
        # ═══════════════════════════════════════════════════════
        # Component 2: Pseudo Text Feature Generator
        # ═══════════════════════════════════════════════════════
        self.text_feature_generator = PseudoTextFeatureGenerator(
            num_classes=num_classes,
            hidden_dim=hidden_dim,
            trainable_pseudo_features=trainable_pseudo_features
        )
        
        # ═══════════════════════════════════════════════════════
        # Component 3: Feature Enhancer (Placeholder)
        # ═══════════════════════════════════════════════════════
        self.feature_enhancer = FeatureEnhancer(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            num_layers=1
        )
        
        # ═══════════════════════════════════════════════════════
        # Component 4: Language-guided Query Selection
        # ═══════════════════════════════════════════════════════
        self.query_selection = LanguageGuidedQuerySelection(
            num_queries=num_queries,
            hidden_dim=hidden_dim,
            image_feature_dim=hidden_dim,
            text_feature_dim=hidden_dim
        )
        
        # ═══════════════════════════════════════════════════════
        # Component 5: Cross-Modality Decoder
        # ═══════════════════════════════════════════════════════
        self.decoder = CrossModalityDecoder(
            hidden_dim=hidden_dim,
            num_queries=num_queries,
            num_classes=num_classes,
            num_decoder_layers=num_decoder_layers,
            num_heads=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout
        )
    
    def forward(self, volumes: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the complete architecture.
        
        Args:
            volumes: (B, 1, D, H, W) - CT volumes
        
        Returns:
            dict with:
                - 'pred_logits': (B, num_queries, num_classes+1)
                - 'pred_boxes': (B, num_queries, 6)
                - 'vanilla_text_features': (B, num_classes, hidden_dim)
                - 'vanilla_image_features': (N_total, B, hidden_dim) multi-scale
        """
        B = volumes.shape[0]
        
        # ═══════════════════════════════════════════════════════
        # Step 1: Extract vanilla features from both modalities
        # ═══════════════════════════════════════════════════════
        
        # Image features from backbone
        backbone_out = self.image_backbone(volumes)
        
        if self.multi_scale:
            # Multi-scale output: backbone returns (N_total, B, hidden_dim)
            image_features_flat = backbone_out  # already (N_total, B, C)
        else:
            # Legacy single-scale: backbone returns (B, D', H', W', C)
            vanilla_image_features = backbone_out
            image_features_permuted = vanilla_image_features.permute(0, 4, 1, 2, 3)
            image_features_flat = image_features_permuted.flatten(2).permute(2, 0, 1)
        
        # Text features from pseudo generator
        vanilla_text_features = self.text_feature_generator(B)  # (B, num_classes, hidden_dim)
        
        # ═══════════════════════════════════════════════════════
        # Step 2: Enhance features through bidirectional cross-attention
        # (Currently a placeholder - just passes through)
        # ═══════════════════════════════════════════════════════
        
        # image_features_flat: (N_total, B, hidden_dim)
        enhanced_text_features, enhanced_image_features = self.feature_enhancer(
            vanilla_text_features,
            image_features_flat
        )
        
        # ═══════════════════════════════════════════════════════
        # Step 3: Language-guided query selection
        # ═══════════════════════════════════════════════════════
        
        # Convert image features from (N, B, D) to (B, N, D) for query selection
        image_features_batch_first = enhanced_image_features.permute(1, 0, 2)
        
        selected_queries = self.query_selection(
            image_features_batch_first,
            enhanced_text_features,
            B
        )  # (num_queries, B, hidden_dim)
        
        # ═══════════════════════════════════════════════════════
        # Step 4: Cross-modality decoding
        # ═══════════════════════════════════════════════════════
        
        pred_logits, pred_boxes = self.decoder(
            enhanced_image_features,  # (N, B, hidden_dim)
            enhanced_text_features,   # (B, num_classes, hidden_dim)
            selected_queries          # (num_queries, B, hidden_dim)
        )
        
        # ═══════════════════════════════════════════════════════
        # Step 5: Return predictions and intermediate features
        # ═══════════════════════════════════════════════════════
        
        outputs = {
            'pred_logits': pred_logits,
            'pred_boxes': pred_boxes,
            'vanilla_text_features': vanilla_text_features,
            'vanilla_image_features': image_features_flat  # (N_total, B, hidden_dim)
        }
        
        return outputs
    
    def get_num_params(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def build_model(config: dict) -> GroundingDETR3D:
    """
    Build model from configuration dict.
    
    Args:
        config: Configuration dictionary with model parameters
    
    Returns:
        Initialized GroundingDETR3D model
    """
    model_config = config.get('model', {})
    
    model = GroundingDETR3D(
        num_classes=model_config.get('num_classes', 5),
        num_queries=model_config.get('num_queries', 100),
        hidden_dim=model_config.get('hidden_dim', 256),
        backbone_embed_dim=model_config.get('backbone_embed_dim', 96),
        backbone_depths=model_config.get('backbone_depths', [2, 2, 6, 2]),
        backbone_num_heads=model_config.get('backbone_num_heads', [3, 6, 12, 24]),
        backbone_window_size=tuple(model_config.get('backbone_window_size', [7, 7, 7])),
        backbone_out_indices=tuple(model_config.get('backbone_out_indices', [])) or None,
        num_encoder_layers=model_config.get('num_encoder_layers', 6),
        num_decoder_layers=model_config.get('num_decoder_layers', 6),
        num_heads=model_config.get('num_heads', 8),
        dim_feedforward=model_config.get('dim_feedforward', 2048),
        dropout=model_config.get('dropout', 0.1),
        trainable_pseudo_features=model_config.get('trainable_pseudo_features', True)
    )
    
    print(f"Model built with {model.get_num_params():,} parameters")
    
    return model
