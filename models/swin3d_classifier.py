"""
Swin3D Classifier for backbone pretraining.
Wraps SwinTransformer3D with a classification head for multi-label
organ injury classification on the full RSNA dataset.
"""
import torch
import torch.nn as nn
from typing import Dict

from .swin3d_backbone import SwinTransformer3D


class Swin3DClassifier(nn.Module):
    """
    Swin3D backbone + classification head for pretraining.
    
    Architecture:
        SwinTransformer3D → AdaptiveAvgPool3D → Dropout → FC → Sigmoid
    
    After pretraining, the backbone weights can be loaded into
    GroundingDETR3D.image_backbone via load_pretrained_backbone().
    """
    
    def __init__(
        self,
        num_labels: int = 14,
        in_channels: int = 1,
        patch_size: tuple = (4, 4, 4),
        embed_dim: int = 96,
        depths: list = [2, 2, 6, 2],
        num_heads: list = [3, 6, 12, 24],
        window_size: tuple = (7, 7, 7),
        mlp_ratio: float = 4.0,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        classifier_dropout: float = 0.3,
    ):
        super().__init__()
        self.num_labels = num_labels
        
        # Backbone (same architecture as in GroundingDETR3D)
        self.backbone = SwinTransformer3D(
            in_channels=in_channels,
            patch_size=patch_size,
            embed_dim=embed_dim,
            depths=depths,
            num_heads=num_heads,
            window_size=window_size,
            mlp_ratio=mlp_ratio,
            qkv_bias=True,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            out_channels=None,  # No projection, use raw backbone features
            out_indices=(len(depths) - 1,),
        )
        
        # Feature dimension from backbone last stage
        backbone_dim = int(embed_dim * 2 ** (len(depths) - 1))
        
        # Classification head
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(classifier_dropout),
            nn.Linear(backbone_dim, backbone_dim // 2),
            nn.GELU(),
            nn.Dropout(classifier_dropout * 0.5),
            nn.Linear(backbone_dim // 2, num_labels),
        )
        
        self._init_classifier()
    
    def _init_classifier(self):
        """Initialize classifier head weights."""
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: (B, 1, D, H, W) - CT volumes
        
        Returns:
            dict with:
                - 'logits': (B, num_labels) - raw logits (before sigmoid)
                - 'features': (B, backbone_dim) - pooled backbone features
        """
        # Backbone features: (B, D', H', W', C)
        features = self.backbone(x)
        
        # Pool: (B, D', H', W', C) -> (B, C, D', H', W') -> (B, C, 1, 1, 1) -> (B, C)
        features_permuted = features.permute(0, 4, 1, 2, 3)
        pooled = self.pool(features_permuted).flatten(1)  # (B, C)
        
        # Classify
        logits = self.classifier(pooled)  # (B, num_labels)
        
        return {
            'logits': logits,
            'features': pooled,
        }
    
    def get_backbone_state_dict(self) -> dict:
        """
        Extract backbone weights for transfer to GroundingDETR3D.
        
        Returns:
            State dict with only backbone parameters.
        """
        return self.backbone.state_dict()
    
    def get_num_params(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_backbone_num_params(self) -> int:
        """Count backbone parameters (for reference)."""
        return sum(p.numel() for p in self.backbone.parameters() if p.requires_grad)


def build_pretrain_model(config: dict) -> Swin3DClassifier:
    """
    Build pretrain classifier from configuration dict.
    
    Args:
        config: Configuration dictionary with model parameters
    
    Returns:
        Initialized Swin3DClassifier model
    """
    model_cfg = config.get('model', {})
    
    model = Swin3DClassifier(
        num_labels=model_cfg.get('num_labels', 14),
        in_channels=1,
        patch_size=tuple(model_cfg.get('patch_size', [4, 4, 4])),
        embed_dim=model_cfg.get('backbone_embed_dim', 96),
        depths=model_cfg.get('backbone_depths', [2, 2, 6, 2]),
        num_heads=model_cfg.get('backbone_num_heads', [3, 6, 12, 24]),
        window_size=tuple(model_cfg.get('backbone_window_size', [7, 7, 7])),
        mlp_ratio=model_cfg.get('mlp_ratio', 4.0),
        drop_rate=model_cfg.get('dropout', 0.0),
        attn_drop_rate=model_cfg.get('attn_drop_rate', 0.0),
        classifier_dropout=model_cfg.get('classifier_dropout', 0.3),
    )
    
    total_params = model.get_num_params()
    backbone_params = model.get_backbone_num_params()
    print(f"Pretrain model built: {total_params:,} total params "
          f"({backbone_params:,} backbone + {total_params - backbone_params:,} classifier)")
    
    return model
