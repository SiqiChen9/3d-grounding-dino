"""
Language-guided Query Selection - ENHANCED IMPLEMENTATION. 
Corresponds to "Language-guide Query Selection" in the architecture diagram.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class LanguageGuidedQuerySelection(nn.Module):
    """
    Language-guided Query Selection with intelligent query generation and selection.
    
    This module implements dynamic query selection based on text features: 
    
    Planned Functionality:
    ┌─────────────────────────────────────────────────────┐
    │ 1. Analyze text features to understand targets      │
    │    - Which classes are present?                      │
    │    - What are their characteristics?                │
    ├─────────────────────────────────────────────────────┤
    │ 2. Select relevant queries                          │
    │    - Filter out irrelevant queries                  │
    │    - Prioritize queries for detected classes        │
    ├─────────────────────────────────────────────────────┤
    │ 3. Initialize queries with text guidance            │
    │    - Use text embeddings to guide query init        │
    │    - Improve query diversity and relevance          │
    └─────────────────────────────────────────────────────┘
    
    Implementation Details:
    - Content Queries: Generated from enhanced text features
    - Positional Queries: Learned spatial priors
    - Query Modulation: Text-conditioned query adjustment
    - Top-k Selection: Select most relevant queries per category
    """

    
    def __init__(
        self,
        num_queries: int = 100,
        hidden_dim: int = 256,
        num_classes: int = 5,
        image_feature_dim: int = 768,
        num_heads: int = 8,
        dropout:  float = 0.1,
        dynamic_selection: bool = True,
        query_per_class: int = 20,
    ):
        """
        Args:
            num_queries: Total number of object queries to generate
            hidden_dim: Hidden dimension for queries
            num_classes: Number of object classes
            image_feature_dim: Input dimension of image features from backbone
            num_heads: Number of attention heads for query-text matching
            dropout: Dropout rate
            dynamic_selection: Whether to enable dynamic query selection
            query_per_class: Number of queries to allocate per detected class
        """
        super().__init__()
        self.num_queries = num_queries
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.num_heads = num_heads
        self.dynamic_selection = dynamic_selection
        self.query_per_class = query_per_class

        # ═══════════════════════════════════════════════════════
        # 1. TEXT FEATURE ANALYSIS
        # ═══════════════════════════════════════════════════════
        
        self.text_proj = nn.Linear(hidden_dim, hidden_dim)
        
        self.class_importance_scorer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
        # ═══════════════════════════════════════════════════════
        # 2. IMAGE FEATURE PROCESSING
        # ═══════════════════════════════════════════════════════
        
        self.image_proj = nn.Linear(image_feature_dim, hidden_dim)
        
        # ═══════════════════════════════════════════════════════
        # 3. QUERY GENERATION AND MODULATION
        # ═══════════════════════════════════════════════════════
        
        self.learnable_queries = nn.Parameter(
            torch.randn(num_queries, hidden_dim)
        )
        nn.init.xavier_uniform_(self.learnable_queries)
        
        self. content_query_generator = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        self.query_modulator = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # ═══════════════════════════════════════════════════════
        # 4. QUERY-TEXT MATCHING
        # ═══════════════════════════════════════════════════════
        
        if dynamic_selection:
            self. query_text_matcher = nn.MultiheadAttention(
                embed_dim=hidden_dim,
                num_heads=num_heads,
                dropout=dropout,
                batch_first=True
            )
            
            self.query_ranker = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, 1)
            )
        
        # ═══════════════════════════════════════════════════════
        # 5. NORMALIZATION AND REGULARIZATION
        # ═══════════════════════════════════════════════════════
        
        self.layer_norm_text = nn.LayerNorm(hidden_dim)
        self.layer_norm_image = nn. LayerNorm(hidden_dim)
        self.layer_norm_queries = nn.LayerNorm(hidden_dim)
        
        self.dropout = nn.Dropout(dropout)
        
        self._init_weights()

    def _init_weights(self):
        """Initialize weights with Xavier uniform."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
    
    def _analyze_text_features(
        self,
        text_features: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Analyze text features to determine class importance and characteristics.
        
        Args:
            text_features: (B, num_classes, hidden_dim) - enhanced text features
        
        Returns:
            Tuple of:  
            - text_proj:  (B, num_classes, hidden_dim) - projected text features
            - class_scores: (B, num_classes) - importance score per class
        """
        text_proj = self.text_proj(text_features)
        text_proj = self.layer_norm_text(text_proj)
        
        B, C, D = text_proj.shape
        text_flat = text_proj.reshape(B * C, D)
        class_scores = self.class_importance_scorer(text_flat)
        class_scores = class_scores.reshape(B, C)
        
        return text_proj, class_scores
    
    def _process_image_features(
        self,
        image_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Process image features through projection and normalization.
        
        Args:
            image_features: (N, B, image_feature_dim) - image features
        
        Returns:  
            image_proj: (N, B, hidden_dim) - processed image features
        """
        image_proj = self.image_proj(image_features)
        image_proj = self.layer_norm_image(image_proj)
        return image_proj
    
    def _generate_content_queries(
        self,
        text_features: torch. Tensor,
        class_scores: torch.Tensor
    ) -> torch.Tensor:
        """
        Generate content queries from text features.
        
        Args:
            text_features: (B, num_classes, hidden_dim) - projected text features
            class_scores:   (B, num_classes) - class importance scores
        
        Returns:
            content_queries:  (B, num_classes * query_per_class, hidden_dim)
        """
        B, C, D = text_features.shape
        
        weighted_features = text_features * class_scores.unsqueeze(-1)
        
        repeated_features = weighted_features.unsqueeze(2).expand(
            B, C, self.query_per_class, D
        )
        
        repeated_flat = repeated_features.reshape(B * C * self.query_per_class, D)
        content_queries = self.content_query_generator(repeated_flat)
        content_queries = content_queries.reshape(
            B, C * self.query_per_class, D
        )
        
        return content_queries
    
    def _modulate_queries(
        self,
        base_queries: torch.Tensor,
        text_context: torch.Tensor,
        image_context: torch. Tensor
    ) -> torch.Tensor:
        """
        Modulate queries based on text and image context.
        
        Args:
            base_queries: (num_queries, B, hidden_dim) - base query embeddings
            text_context:   (B, hidden_dim) - aggregated text context
            image_context: (B, hidden_dim) - aggregated image context
        
        Returns:
            modulated_queries: (num_queries, B, hidden_dim)
        """
        num_q, B, D = base_queries. shape
        
        text_context = text_context.unsqueeze(0).expand(num_q, -1, -1)
        image_context = image_context.unsqueeze(0).expand(num_q, -1, -1)
        
        combined = torch.cat([base_queries, text_context + image_context], dim=-1)
        combined = combined.reshape(num_q * B, 2 * D)
        
        modulation = self.query_modulator(combined)
        modulation = modulation.reshape(num_q, B, D)
        
        modulated_queries = base_queries + modulation
        modulated_queries = self.layer_norm_queries(modulated_queries)
        
        return modulated_queries
    
    def _select_top_queries(
        self,
        queries: torch.Tensor,
        text_features: torch. Tensor,
        class_scores: torch.Tensor
    ) -> torch.Tensor:
        """
        Select top-k queries based on query-text matching. 
        
        Args:
            queries: (num_queries, B, hidden_dim) - candidate queries
            text_features: (B, num_classes, hidden_dim) - text features
            class_scores: (B, num_classes) - class importance scores
        
        Returns:  
            selected_queries: (num_queries, B, hidden_dim)
        """
        if not self.dynamic_selection:
            return queries
        
        num_q, B, D = queries.shape
        
        queries_t = queries.permute(1, 0, 2)
        text_mean = text_features.mean(dim=1, keepdim=True)
        
        with torch.no_grad():
            attn_output, _ = self.query_text_matcher(
                queries_t, text_mean, text_mean
            )
        
        query_scores = self.query_ranker(attn_output)
        query_scores = query_scores.squeeze(-1)
        
        query_weights = torch.softmax(query_scores, dim=1)
        
        return queries

    def forward(
        self,
        text_features: torch.Tensor,
        image_features: torch.Tensor,
        batch_size: int
    ) -> torch.Tensor:
        """
        Generate object queries guided by both text and image features. 

        Args:
            text_features: (B, num_classes, hidden_dim) - enhanced text features
            image_features:  (N, B, image_feature_dim) - image features (flattened spatial)
            batch_size: Number of samples in batch

        Returns:
            queries:   (num_queries, B, hidden_dim) - selected/initialized queries
        """
        # Step 1: Analyze text features
        text_proj, class_scores = self._analyze_text_features(text_features)
        
        # Step 2: Process image features
        image_proj = self._process_image_features(image_features)
        
        # Step 3: Generate content queries from text
        content_queries = self._generate_content_queries(text_proj, class_scores)
        
        # Pad or trim to num_queries
        B, C_q, D = content_queries.shape
        if C_q > self.num_queries:
            content_queries = content_queries[:, :self.num_queries, :]
        elif C_q < self.num_queries:
            num_pad = self.num_queries - C_q
            padding = self.learnable_queries[: num_pad, :]. unsqueeze(0).expand(B, -1, -1)
            content_queries = torch. cat([content_queries, padding], dim=1)
        
        queries = content_queries. permute(1, 0, 2)
        
        # Step 4: Aggregate text and image context
        text_pooled = text_proj.mean(dim=1)
        image_pooled = image_proj. mean(dim=0)
        
        # Step 5: Modulate queries with context
        queries = self._modulate_queries(queries, text_pooled, image_pooled)
        
        # Step 6: Select top-k queries based on text alignment
        queries = self._select_top_queries(queries, text_proj, class_scores)
        
        return queries