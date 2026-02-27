"""
Language-guided Query Selection based on Grounding DINO.

Algorithm:
    S[b,i,j] = image_feat[b,i] · text_feat[b,j]  (similarity matrix)
    score[b,i] = max_j(S[b,i,j])                  (per-position score)
    topk_idx = argsort(score)[-K:]                 (select top-K positions)
    query[b,k] = selected_feat[b,k] + content_query[k]  (mixed init)
"""
import torch
import torch.nn as nn


class LanguageGuidedQuerySelection(nn.Module):
    """
    Select object queries by image-text similarity matching.
    
    Instead of fixed learnable queries (standard DETR), dynamically selects
    the top-K image features most similar to text descriptions, then mixes
    them with learnable content queries.
    """

    def __init__(
        self,
        num_queries: int = 900,
        hidden_dim: int = 256,
        image_feature_dim: int = 256,
        text_feature_dim: int = 256,
    ):
        """
        Args:
            num_queries: Number of queries to select (K).
            hidden_dim: Internal feature dimension.
            image_feature_dim: Dimension of input image features.
            text_feature_dim: Dimension of input text features.
        """
        super().__init__()
        
        self.num_queries = num_queries
        self.hidden_dim = hidden_dim

        # Feature projection layers (identity if dims already match)
        # self.image_proj = (nn.Linear(image_feature_dim, hidden_dim)
        #                    if image_feature_dim != hidden_dim else nn.Identity())
        if image_feature_dim != hidden_dim:
            self.image_proj = nn.Linear(image_feature_dim, hidden_dim)
        else:
            self.image_proj = nn.Identity()
        # self.text_proj = (nn.Linear(text_feature_dim, hidden_dim)
        #                   if text_feature_dim != hidden_dim else nn.Identity())
        if text_feature_dim != hidden_dim:
            self.text_proj = nn.Linear(text_feature_dim, hidden_dim)
        else:
            self.text_proj = nn.Identity()

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
        Generate language-guided object queries.
        
        Args:
            image_features: (B, num_img_tokens, image_feature_dim)
            text_features:  (B, num_text_tokens, text_feature_dim)
            batch_size: Batch size (must match B).
        
        Returns:
            queries: (num_queries, B, hidden_dim) — ready for DETR decoder.
        """
        # Step 1: Project to unified space
        image_feat = self.image_proj(image_features)  # (B, N_img, D)
        text_feat = self.text_proj(text_features)      # (B, N_txt, D)
        B, num_img_tokens, D = image_feat.shape

        # Step 2: Image-text similarity matrix via dot product
        logits = torch.einsum("bic,btc->bit", image_feat, text_feat)  # (B, N_img, N_txt)

        # Step 3: Per-position score = max similarity across text tokens
        logits_per_img_feature = logits.max(dim=-1)[0]  # (B, N_img)

        # Step 4: Top-K selection (clamp K to available tokens)
        actual_k = min(self.num_queries, num_img_tokens)
        # _, topk_idx = torch.topk(logits_per_img_feature, actual_k, dim=1)  # (B, actual_k)
        topk_values, topk_idx = torch.topk(
            logits_per_img_feature,
            min(self.num_queries, num_img_tokens),  # Safe K value
            dim=1,
            largest=True  # Select largest values (most similar)
        )

        # Step 5: Gather selected image features
        topk_idx_expanded = topk_idx.unsqueeze(-1).expand(-1, -1, D)
        # selected_features = torch.gather(image_feat, dim=1, index=topk_idx_expanded)  # (B, actual_k, D)
        selected_features = torch.gather(
            image_feat,           # Source: all image features (B, num_img_tokens, D)
            dim=1,                # Gather along image_token dimension
            index=topk_idx_expanded
        )  # (B, actual_k, D)

        # Step 6: Pad if needed (when image tokens < num_queries)
        if actual_k < self.num_queries:
            pad = torch.zeros(B, self.num_queries - actual_k, D,
                              device=selected_features.device,
                              dtype=selected_features.dtype)
            selected_features = torch.cat([selected_features, pad], dim=1)  # (B, num_queries, D)

        # Step 7: Transpose to decoder format (seq, batch, dim)
        queries = selected_features.permute(1, 0, 2)  # (num_queries, B, D)
        
        return queries