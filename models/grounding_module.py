"""
Simplified grounding module with pseudo-class tokens.
Provides category conditioning for detection queries.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class PseudoClassTokenEncoder(nn.Module):
    """
    Learnable pseudo-class token encoder.
    Generates class-specific embeddings for grounding-style conditioning.
    """
    
    def __init__(
        self,
        num_classes: int = 5,
        hidden_dim: int = 256
    ):
        super().__init__()
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        
        # Learnable class token embeddings
        self.class_tokens = nn.Embedding(num_classes, hidden_dim)
        
        # Optional: add a small MLP for transformation
        self.token_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
    
    def forward(self, batch_size: int) -> torch.Tensor:
        """
        Generate class tokens for a batch.
        
        Args:
            batch_size: Number of samples in batch
        
        Returns:
            (B, num_classes, hidden_dim) - class token embeddings
        """
        # Get all class tokens
        class_indices = torch.arange(
            self.num_classes,
            device=self.class_tokens.weight.device
        )
        tokens = self.class_tokens(class_indices)  # (num_classes, hidden_dim)
        
        # Project tokens
        tokens = self.token_proj(tokens)
        
        # Expand for batch
        tokens = tokens.unsqueeze(0).expand(batch_size, -1, -1)  # (B, num_classes, hidden_dim)
        
        return tokens


class CrossModalFusion(nn.Module):
    """
    Cross-attention between object queries and class tokens.
    Allows queries to be conditioned on target classes.
    """
    
    def __init__(
        self,
        hidden_dim: int = 256,
        num_heads: int = 8,
        dropout: float = 0.1
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        self.cross_attn = nn.MultiheadAttention(
            hidden_dim,
            num_heads,
            dropout=dropout
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        queries: torch.Tensor,
        class_tokens: torch.Tensor
    ) -> torch.Tensor:
        """
        Fuse object queries with class tokens via cross-attention.
        
        Args:
            queries: (num_queries, B, hidden_dim)
            class_tokens: (num_classes, B, hidden_dim)
        
        Returns:
            (num_queries, B, hidden_dim) - fused queries
        """
        # Cross-attention: queries attend to class tokens
        fused_queries, _ = self.cross_attn(
            queries,
            class_tokens,
            class_tokens
        )
        
        # Residual connection
        queries = queries + self.dropout(fused_queries)
        queries = self.norm(queries)
        
        return queries


class GroundingModule(nn.Module):
    """
    Complete grounding module combining class tokens and fusion.
    """
    
    def __init__(
        self,
        num_classes: int = 5,
        hidden_dim: int = 256,
        num_heads: int = 8,
        dropout: float = 0.1,
        use_fusion: bool = True
    ):
        super().__init__()
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        self.use_fusion = use_fusion
        
        # Class token encoder
        self.class_encoder = PseudoClassTokenEncoder(num_classes, hidden_dim)
        
        # Cross-modal fusion (optional)
        if use_fusion:
            self.fusion = CrossModalFusion(hidden_dim, num_heads, dropout)
        else:
            self.fusion = None
    
    def forward(
        self,
        batch_size: int,
        queries: torch.Tensor = None
    ) -> tuple:
        """
        Generate class tokens and optionally fuse with queries.
        
        Args:
            batch_size: Number of samples
            queries: Optional (num_queries, B, hidden_dim) - object queries
        
        Returns:
            If queries provided and fusion enabled:
                (fused_queries, class_tokens)
            Otherwise:
                class_tokens
        """
        # Generate class tokens
        class_tokens = self.class_encoder(batch_size)  # (B, num_classes, hidden_dim)
        
        # Optionally fuse with queries
        if queries is not None and self.fusion is not None:
            # Reshape for cross-attention
            class_tokens_t = class_tokens.permute(1, 0, 2)  # (num_classes, B, hidden_dim)
            
            # Fuse
            fused_queries = self.fusion(queries, class_tokens_t)
            
            return fused_queries, class_tokens
        else:
            return class_tokens
