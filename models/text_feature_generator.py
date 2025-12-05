"""
Pseudo Text Feature Generator.
Replaces the Text Backbone in the architecture diagram with learnable embeddings.
"""
import torch
import torch.nn as nn
from typing import Optional


class PseudoTextFeatureGenerator(nn.Module):
    """
    Generates pseudo text features using learnable embeddings.
    
    This replaces the Text Backbone in the original Grounding-DINO architecture.
    Instead of processing real text inputs (like "cat, person, mouse..."), 
    it uses learnable embeddings for a fixed set of classes.
    
    Architecture mapping:
        - Input: None (generates features based on batch size)
        - Output: vanilla_text_features (B, num_classes, hidden_dim)
    """
    
    def __init__(
        self,
        num_classes: int = 5,
        hidden_dim: int = 256,
        trainable_pseudo_features: bool = True
    ):
        """
        Args:
            num_classes: Number of object classes
            hidden_dim: Dimension of text features
            trainable_pseudo_features: If True, pseudo features are trainable.
                                      If False, they are frozen after initialization.
        """
        super().__init__()
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        self.trainable_pseudo_features = trainable_pseudo_features
        
        # Learnable class embeddings (pseudo text features)
        self.class_embeddings = nn.Parameter(
            torch.randn(num_classes, hidden_dim),
            requires_grad=trainable_pseudo_features
        )
        
        # Optional projection layer to add capacity
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Initialize embeddings
        self._init_embeddings()
    
    def _init_embeddings(self):
        """Initialize embeddings with Xavier uniform."""
        nn.init.xavier_uniform_(self.class_embeddings)
    
    def forward(self, batch_size: int) -> torch.Tensor:
        """
        Generate pseudo text features for a batch.
        
        Args:
            batch_size: Number of samples in the batch
        
        Returns:
            vanilla_text_features: (B, num_classes, hidden_dim)
                Pseudo text features representing each class.
        """
        # Get class embeddings
        text_features = self.class_embeddings  # (num_classes, hidden_dim)
        
        # Apply projection
        text_features = self.projection(text_features)  # (num_classes, hidden_dim)
        
        # Expand for batch: (num_classes, hidden_dim) -> (B, num_classes, hidden_dim)
        text_features = text_features.unsqueeze(0).expand(batch_size, -1, -1)
        
        return text_features
    
    def freeze_embeddings(self):
        """Freeze the pseudo text embeddings (make them non-trainable)."""
        self.class_embeddings.requires_grad = False
        self.trainable_pseudo_features = False
    
    def unfreeze_embeddings(self):
        """Unfreeze the pseudo text embeddings (make them trainable)."""
        self.class_embeddings.requires_grad = True
        self.trainable_pseudo_features = True
    
    def get_num_params(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
