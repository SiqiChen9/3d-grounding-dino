"""
Language-guided Query Selection - Based on Research Paper Implementation.

Core Algorithm:
1. Compute image-text similarity using dot product
2. For each image position, find max similarity across all text tokens
3. Select top-K image positions with highest similarity scores
4. Initialize queries with mixed position and learnable content

Mathematical Formulation:
    S[b,i,j] = image_feat[b,i] · text_feat[b,j]
    score[b,i] = max_j(S[b,i,j])
    topk_idx[b] = argsort(score[b])[-K:]
    
    query[b,k] = selected_image_feat[b,k] + content_query[k]
"""
import torch
import torch.nn as nn
from typing import Tuple, Optional


class LanguageGuidedQuerySelection(nn.Module):
    """
    Language-guided Query Selection module following research paper algorithm.
    
    This module implements intelligent query selection based on image-text similarity.
    Instead of using fixed random queries like standard DETR, it dynamically selects
    the most relevant image features based on their similarity to text descriptions.
    
    Key Features:
    - Image-text similarity matching using dot product
    - Top-K selection of most relevant image features
    - Mixed query initialization (positional + learnable content)
    - Efficient einsum-based computation
    """

    def __init__(
        self,
        num_queries: int = 900,
        hidden_dim: int = 256,
        image_feature_dim: int = 256,
        text_feature_dim: int = 256,
    ):
        """
        Initialize the Language-Guided Query Selection module.
        
        Args:
            num_queries (int): Number of queries to select.
                              Default: 900 (can be adjusted based on computation budget)
            
            hidden_dim (int): Feature dimension used internally.
                             Default: 256
            
            image_feature_dim (int): Dimension of input image features from backbone.
                                    Default: 256
            
            text_feature_dim (int): Dimension of input text features from text encoder.
                                   Default: 256
        
        Example:
            >>> model = LanguageGuidedQuerySelection(
            ...     num_queries=900,
            ...     hidden_dim=256,
            ...     image_feature_dim=256,
            ...     text_feature_dim=256
            ... )
        """
        super().__init__()
        
        self.num_queries = num_queries
        self.hidden_dim = hidden_dim

        # ====================================================================
        # Feature Projection Layers
        # ====================================================================
        # Project image features to hidden dimension if needed
        # This ensures image and text features are in the same feature space
        if image_feature_dim != hidden_dim:
            self.image_proj = nn.Linear(image_feature_dim, hidden_dim)
        else:
            # If dimensions match, use identity (no-op)
            self.image_proj = nn.Identity()
        
        # Project text features to hidden dimension if needed
        if text_feature_dim != hidden_dim:
            self.text_proj = nn.Linear(text_feature_dim, hidden_dim)
        else:
            # If dimensions match, use identity (no-op)
            self.text_proj = nn.Identity()

        # ====================================================================
        # Learnable Query Content Parameters
        # ====================================================================
        # These are learnable parameters that will be optimized during training
        # They represent the "content" or "semantic" part of queries
        # Different from positional part which comes from selected image features
        #
        # Shape: (num_queries, hidden_dim)
        # Each row is a learnable vector for one query slot
        self.content_queries = nn.Parameter(
            torch.randn(num_queries, hidden_dim)
        )
        # Initialize with Xavier uniform for better gradient flow
        nn.init.xavier_uniform_(self.content_queries)

        # Initialize all weight matrices
        self._init_weights()

    def _init_weights(self):
        """
        Initialize all linear layer weights using Xavier uniform initialization.
        
        Xavier initialization ensures gradients flow stably during training,
        especially important for deep networks. Biases are initialized to zero.
        """
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def forward(
        self,
        image_features: torch.Tensor,
        text_features: torch.Tensor,
        batch_size: int
    ) -> torch.Tensor:
        """
        Forward pass: Generate language-guided object queries.
        
        This method implements the core algorithm:
        1. Project features to unified space
        2. Compute image-text similarity matrix
        3. Find max similarity for each image position
        4. Select top-K positions with highest similarity
        5. Mix selected image features with learnable content
        6. Format output for DETR decoder
        
        Args:
            image_features (torch.Tensor): Image feature tokens from backbone.
                Shape: (B, num_img_tokens, image_feature_dim)
                
                Where:
                - B: Batch size (number of images)
                - num_img_tokens: Number of image feature tokens
                                 (e.g., 196 for 14x14 ViT, or 10000+ for fine-grained)
                - image_feature_dim: Feature dimension (e.g., 768 from ViT, or 256 after projection)
                
                Example shape: (2, 10000, 256) for batch of 2 images
            
            text_features (torch.Tensor): Text feature tokens from text encoder.
                Shape: (B, num_text_tokens, text_feature_dim)
                
                Where:
                - B: Batch size (same as image_features)
                - num_text_tokens: Number of text tokens (usually < 256)
                - text_feature_dim: Feature dimension (e.g., 256)
                
                Example shape: (2, 50, 256) for batch of 2 text prompts with ~50 tokens
            
            batch_size (int): Batch size for compatibility. Should match B in tensors.
        
        Returns:
            torch.Tensor: Generated queries ready for DETR decoder.
                Shape: (num_queries, B, hidden_dim)
                
                Where:
                - num_queries: 900 (or configured value)
                - B: Batch size
                - hidden_dim: 256
                
                Format: (sequence_length, batch_size, feature_dim)
                This is the standard format expected by transformer decoders.
        
        Algorithm Flow:
        
            Step 1: Project Features to Unified Space
            ==========================================
            Input image_features: (B, N_img, image_feature_dim)
            Input text_features:  (B, N_text, text_feature_dim)
            
            image_feat = linear_projection(image_features)  # (B, N_img, hidden_dim=256)
            text_feat = linear_projection(text_features)    # (B, N_text, hidden_dim=256)
            
            
            Step 2: Compute Image-Text Similarity Matrix
            ===============================================
            Using efficient einsum operation:
            
            einsum("bic,btc->bit", image_feat, text_feat)
            
            Breakdown:
            - b: batch dimension (B)
            - i: image token index (N_img)
            - t: text token index (N_text)
            - c: channel/feature dimension (hidden_dim)
            
            Operation: image[b,i,:] · text[b,t,:]^T
            Result shape: (B, N_img, N_text)
            
            Each element logits[b,i,t] represents how similar image token i is to text token t.
            
            
            Step 3: Maximum Similarity Aggregation
            ========================================
            For each image position, find its strongest match across all text tokens:
            
            logits_per_img = max(logits, dim=-1)  # (B, N_img)
            
            Meaning: For each image feature, what is its highest similarity to any text token?
            
            
            Step 4: Top-K Selection
            =======================
            Select the num_queries image positions with highest similarity scores:
            
            topk_idx = topk(logits_per_img, k=num_queries)  # (B, num_queries)
            
            These indices tell us which image features are most relevant to the text.
            
            
            Step 5: Extract Selected Features
            ==================================
            Use the indices to gather the actual feature vectors:
            
            selected_features = gather(image_feat, topk_idx)  # (B, num_queries, hidden_dim)
            
            Now we have the actual feature vectors of the top-K similar image positions.
            
            
            Step 6: Mixed Query Initialization
            ===================================
            Combine positional and content information:
            
            position_part = selected_features           # (B, num_queries, hidden_dim)
            content_part = self.content_queries         # (num_queries, hidden_dim)
            
            mixed_queries = position_part + content_part  # (B, num_queries, hidden_dim)
            
            - Position part: Comes from real image features (spatial information)
            - Content part: Learnable parameters (semantic information)
            - Mixed: Both spatial and semantic information combined
            
            
            Step 7: Convert to Decoder Format
            ==================================
            Transpose from (B, num_queries, D) to (num_queries, B, D):
            
            queries = mixed_queries.permute(1, 0, 2)  # (num_queries, B, hidden_dim)
            
            This is the standard transformer decoder format where sequence comes first.
        
        Example:
            >>> model = LanguageGuidedQuerySelection(num_queries=900)
            >>> 
            >>> batch_size = 2
            >>> image_features = torch.randn(batch_size, 10000, 256)
            >>> text_features = torch.randn(batch_size, 50, 256)
            >>> 
            >>> queries = model(image_features, text_features, batch_size)
            >>> print(queries.shape)  # torch.Size([900, 2, 256])
            >>> 
            >>> # queries can now be fed to a DETR decoder
        """
        
        # ====================================================================
        # STEP 1: Project Features to Unified Space
        # ====================================================================
        # Ensure both image and text features are in the same feature space
        image_feat = self.image_proj(image_features)  # (B, num_img_tokens, hidden_dim)
        text_feat = self.text_proj(text_features)      # (B, num_text_tokens, hidden_dim)
        
        B, num_img_tokens, D = image_feat.shape

        # ====================================================================
        # STEP 2: Compute Image-Text Similarity Matrix
        # ====================================================================
        # Use einsum for efficient dot product computation
        # logits[b,i,j] = image_feat[b,i] · text_feat[b,j]
        #
        # Notation explanation:
        #   b = batch index (size B)
        #   i = image token index (size num_img_tokens)
        #   t = text token index (size num_text_tokens)
        #   c = channel/feature dimension (size hidden_dim)
        #
        # Operation: Compute dot product between each image-text pair
        logits = torch.einsum("bic,btc->bit", image_feat, text_feat)
        # Output shape: (B, num_img_tokens, num_text_tokens)
        
        # Example:
        # B=2, num_img_tokens=10000, num_text_tokens=50, D=256
        # Output: (2, 10000, 50)
        # Each element is a similarity score between one image feature and one text token

        # ====================================================================
        # STEP 3: Maximum Similarity Aggregation
        # ====================================================================
        # For each image token, find its maximum similarity across all text tokens
        # This removes the text dimension by taking max
        # Result: "How well does each image feature match the text?"
        #
        # max(dim=-1) returns tuple of (max_values, max_indices)
        # We take [0] to get just the values
        logits_per_img_feature = logits.max(dim=-1)[0]  # (B, num_img_tokens)
        
        # Example with above shapes:
        # Input: (2, 10000, 50)
        # Output: (2, 10000)
        # Each of 10000 image features has one score

        # ====================================================================
        # STEP 4: Top-K Selection
        # ====================================================================
        # Select the top num_queries indices with highest similarity scores
        #
        # torch.topk returns tuple of (values, indices)
        # We only need indices [1]
        # Handle edge case where num_queries > num_img_tokens
        topk_values, topk_idx = torch.topk(
            logits_per_img_feature,
            min(self.num_queries, num_img_tokens),  # Safe K value
            dim=1,
            largest=True  # Select largest values (most similar)
        )
        # Output shape: (B, num_queries)
        # Each value is the index of a selected image feature

        # ====================================================================
        # STEP 5: Extract Selected Features Using Indices
        # ====================================================================
        # Now we have indices, use them to gather actual feature vectors
        # We need to expand indices for proper gather operation
        # 
        # topk_idx shape: (B, num_queries) - but we need (B, num_queries, D)
        # Expand to match the feature dimension
        topk_idx_expanded = topk_idx.unsqueeze(-1).expand(-1, -1, D)
        # Shape: (B, num_queries, hidden_dim)
        
        # Use torch.gather to select features at the specified indices
        selected_features = torch.gather(
            image_feat,           # Source: all image features (B, num_img_tokens, D)
            dim=1,                # Gather along image_token dimension
            index=topk_idx_expanded
        )
        # Output shape: (B, num_queries, hidden_dim)

        # ====================================================================
        # STEP 6: Mix Position and Content Information
        # ====================================================================
        # Mixed Query Strategy:
        # - Positional Part: Selected image features (where)
        # - Content Part: Learnable query parameters (what)
        # - Mixed: Both parts combined
        
        # Positional part comes from actual image features
        # This gives spatial information about relevant image regions
        position_part = selected_features  # (B, num_queries, hidden_dim)
        
        # Content part is a learnable parameter
        # This is NOT extracted from features, but learned independently
        # This gives semantic information independent of input
        content_part = self.content_queries.unsqueeze(0).expand(
            batch_size, -1, -1
        )  # (B, num_queries, hidden_dim)
        # Expand from (num_queries, D) to (B, num_queries, D)
        
        # Mix both parts by element-wise addition
        # This combines spatial location (position) with semantic meaning (content)
        mixed_queries = position_part + content_part  # (B, num_queries, hidden_dim)

        # ====================================================================
        # STEP 7: Convert to Decoder Format
        # ====================================================================
        # Standard transformer decoder expects: (sequence_length, batch_size, feature_dim)
        # We currently have: (batch_size, sequence_length, feature_dim)
        # So we need to transpose
        #
        # permute(1, 0, 2) means:
        #   dimension 0 (batch) -> dimension 1
        #   dimension 1 (sequence) -> dimension 0
        #   dimension 2 (features) -> dimension 2
        queries = mixed_queries.permute(1, 0, 2)  # (num_queries, batch_size, hidden_dim)
        
        return queries