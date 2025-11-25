"""
Complete 3D Grounding-DETR model.
Integrates Swin3D backbone, grounding module, and DETR head.
"""
import torch
import torch.nn as nn
from typing import Tuple, Dict

from .swin3d_backbone import SwinTransformer3D
from .detr3d_head import DETR3DHead
from .grounding_module import GroundingModule


class GroundingDETR3D(nn.Module):
    """
    3D Grounding-DETR for CT volume detection.
    
    Architecture:
        1. Swin3D backbone extracts features
        2. Grounding module generates class tokens
        3. DETR head predicts boxes and classes
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
        # DETR parameters
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
        num_heads: int = 8,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        # Grounding parameters
        use_grounding: bool = True
    ):
        super().__init__()
        self.num_classes = num_classes
        self.num_queries = num_queries
        self.hidden_dim = hidden_dim
        self.use_grounding = use_grounding
        
        # Compute backbone output dimension
        backbone_dim = int(backbone_embed_dim * 2 ** (len(backbone_depths) - 1))
        
        # Backbone: Swin3D
        self.backbone = SwinTransformer3D(
            in_channels=1,
            patch_size=(4, 4, 4),
            embed_dim=backbone_embed_dim,
            depths=backbone_depths,
            num_heads=backbone_num_heads,
            window_size=(7, 7, 7),
            mlp_ratio=4.0,
            qkv_bias=True,
            drop_rate=dropout,
            attn_drop_rate=dropout
        )
        
        # Grounding module (optional)
        if use_grounding:
            self.grounding_module = GroundingModule(
                num_classes=num_classes,
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                dropout=dropout,
                use_fusion=True
            )
        else:
            self.grounding_module = None
        
        # Detection head: DETR3D
        self.detr_head = DETR3DHead(
            hidden_dim=hidden_dim,
            num_queries=num_queries,
            num_classes=num_classes,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            num_heads=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            backbone_dim=backbone_dim,
            use_grounding_fusion=use_grounding  # Enable fusion if grounding is enabled
        )
    
    def forward(self, volumes: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            volumes: (B, 1, D, H, W) - CT volumes
        
        Returns:
            dict with:
                - 'pred_logits': (B, num_queries, num_classes+1)
                - 'pred_boxes': (B, num_queries, 6)
                - 'class_tokens': (B, num_classes, hidden_dim) if grounding enabled
        """
        B = volumes.shape[0]
        
        # Extract features with backbone
        features = self.backbone(volumes)  # (B, D', H', W', C)
        
        # Reshape features for DETR head
        features = features.permute(0, 4, 1, 2, 3)  # (B, C, D', H', W')
        
        # Generate class tokens if grounding is enabled
        class_tokens = None
        if self.grounding_module is not None:
            class_tokens = self.grounding_module(B)  # (B, num_classes, hidden_dim)
        
        # Get predictions (pass class_tokens for fusion)
        pred_logits, pred_boxes = self.detr_head(features, class_tokens)
        
        # Build outputs
        outputs = {
            'pred_logits': pred_logits,
            'pred_boxes': pred_boxes
        }
        
        if class_tokens is not None:
            outputs['class_tokens'] = class_tokens
        
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
        num_encoder_layers=model_config.get('num_encoder_layers', 6),
        num_decoder_layers=model_config.get('num_decoder_layers', 6),
        num_heads=model_config.get('num_heads', 8),
        dim_feedforward=model_config.get('dim_feedforward', 2048),
        dropout=model_config.get('dropout', 0.1),
        use_grounding=model_config.get('use_grounding', True)
    )
    
    print(f"Model built with {model.get_num_params():,} parameters")
    
    return model
