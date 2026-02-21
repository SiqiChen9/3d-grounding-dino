"""
Unit tests for models/query_selection.py

Tests:
- LanguageGuidedQuerySelection: output shape, query count, gradient flow
"""
import pytest
import torch
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from models.query_selection import LanguageGuidedQuerySelection


class TestLanguageGuidedQuerySelection:
    """Tests for LanguageGuidedQuerySelection module."""
    
    def test_output_shape(self, device, batch_size, num_classes, hidden_dim, num_queries):
        """Test that output shape is (num_queries, B, hidden_dim)."""
        image_feature_dim = 768
        
        model = LanguageGuidedQuerySelection(
            num_queries=num_queries,
            hidden_dim=hidden_dim,
            image_feature_dim=image_feature_dim,
            text_feature_dim=hidden_dim
        ).to(device)
        
        N = 200  # Must be >= num_queries for full selection
        image_features = torch.randn(batch_size, N, image_feature_dim, device=device)
        text_features = torch.randn(batch_size, num_classes, hidden_dim, device=device)
        
        queries = model(image_features, text_features, batch_size)
        
        assert queries.shape == (num_queries, batch_size, hidden_dim)
    
    def test_different_num_queries(self, device, batch_size, num_classes, hidden_dim):
        """Test with different number of queries."""
        image_feature_dim = 256
        
        for nq in [50, 100, 200]:
            model = LanguageGuidedQuerySelection(
                num_queries=nq,
                hidden_dim=hidden_dim,
                image_feature_dim=image_feature_dim,
                text_feature_dim=hidden_dim
            ).to(device)
            
            image_features = torch.randn(batch_size, 300, image_feature_dim, device=device)
            text_features = torch.randn(batch_size, num_classes, hidden_dim, device=device)
            
            queries = model(image_features, text_features, batch_size)
            
            assert queries.shape[0] == nq
            assert queries.shape[1] == batch_size
            assert queries.shape[2] == hidden_dim
    
    def test_different_batch_sizes(self, device, num_classes, hidden_dim, num_queries):
        """Test with different batch sizes."""
        image_feature_dim = 256
        
        model = LanguageGuidedQuerySelection(
            num_queries=num_queries,
            hidden_dim=hidden_dim,
            image_feature_dim=image_feature_dim,
            text_feature_dim=hidden_dim
        ).to(device)
        
        for bs in [1, 2, 4, 8]:
            image_features = torch.randn(bs, 200, image_feature_dim, device=device)
            text_features = torch.randn(bs, num_classes, hidden_dim, device=device)
            
            queries = model(image_features, text_features, bs)
            
            assert queries.shape == (num_queries, bs, hidden_dim)
    
    def test_different_image_feature_dims(self, device, batch_size, num_classes, hidden_dim, num_queries):
        """Test with different image feature dimensions."""
        for ifd in [256, 512, 768]:
            model = LanguageGuidedQuerySelection(
                num_queries=num_queries,
                hidden_dim=hidden_dim,
                image_feature_dim=ifd,
                text_feature_dim=hidden_dim
            ).to(device)
            
            image_features = torch.randn(batch_size, 200, ifd, device=device)
            text_features = torch.randn(batch_size, num_classes, hidden_dim, device=device)
            
            queries = model(image_features, text_features, batch_size)
            
            assert queries.shape == (num_queries, batch_size, hidden_dim)
    
    def test_gradient_flow(self, device, batch_size, num_classes, hidden_dim, num_queries):
        """Test gradient flow through the module."""
        image_feature_dim = 256
        
        model = LanguageGuidedQuerySelection(
            num_queries=num_queries,
            hidden_dim=hidden_dim,
            image_feature_dim=image_feature_dim,
            text_feature_dim=hidden_dim
        ).to(device)
        
        image_features = torch.randn(
            batch_size, 200, image_feature_dim,
            device=device, requires_grad=True
        )
        text_features = torch.randn(
            batch_size, num_classes, hidden_dim,
            device=device, requires_grad=True
        )
        
        queries = model(image_features, text_features, batch_size)
        loss = queries.sum()
        loss.backward()
        
        # Image inputs should have gradients (text grads may be None due to topk)
        assert image_features.grad is not None
    
    def test_queries_from_text_and_image(self, device, batch_size, num_classes, hidden_dim, num_queries):
        """Test that queries change with different text and image features."""
        image_feature_dim = 256
        
        model = LanguageGuidedQuerySelection(
            num_queries=num_queries,
            hidden_dim=hidden_dim,
            image_feature_dim=image_feature_dim,
            text_feature_dim=hidden_dim
        ).to(device)
        model.eval()
        
        image_features = torch.randn(batch_size, 200, image_feature_dim, device=device)
        text_features = torch.randn(batch_size, num_classes, hidden_dim, device=device)
        
        with torch.no_grad():
            queries1 = model(image_features, text_features, batch_size)
            
            # Different text features should give different queries
            text_features2 = torch.randn(batch_size, num_classes, hidden_dim, device=device)
            queries2 = model(image_features, text_features2, batch_size)
            
            # Different image features should give different queries
            image_features2 = torch.randn(batch_size, 200, image_feature_dim, device=device)
            queries3 = model(image_features2, text_features, batch_size)
        
        assert not torch.allclose(queries1, queries2)
        assert not torch.allclose(queries1, queries3)
    
    def test_output_dtype(self, device, batch_size, num_classes, hidden_dim, num_queries):
        """Test output dtype is float32."""
        model = LanguageGuidedQuerySelection(
            num_queries=num_queries,
            hidden_dim=hidden_dim,
            image_feature_dim=256,
            text_feature_dim=hidden_dim
        ).to(device)
        
        image_features = torch.randn(batch_size, 200, 256, device=device)
        text_features = torch.randn(batch_size, num_classes, hidden_dim, device=device)
        
        queries = model(image_features, text_features, batch_size)
        
        assert queries.dtype == torch.float32
    
    def test_num_queries_exceeds_img_tokens(self, device, batch_size, num_classes, hidden_dim):
        """Test edge case where num_queries > num_img_tokens."""
        num_queries = 200
        model = LanguageGuidedQuerySelection(
            num_queries=num_queries,
            hidden_dim=hidden_dim,
            image_feature_dim=hidden_dim,
            text_feature_dim=hidden_dim
        ).to(device)
        
        # Only 50 image tokens but 200 queries requested
        image_features = torch.randn(batch_size, 50, hidden_dim, device=device)
        text_features = torch.randn(batch_size, num_classes, hidden_dim, device=device)
        
        queries = model(image_features, text_features, batch_size)
        
        # Should still return num_queries (padding with pure content queries)
        assert queries.shape == (num_queries, batch_size, hidden_dim)
