"""
Unit tests for models/feature_enhancer.py

Tests:
- FeatureEnhancer: input/output shapes, feature enhancement
"""
import pytest
import torch
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from models.feature_enhancer import FeatureEnhancer


class TestFeatureEnhancer:
    """Tests for FeatureEnhancer module."""
    
    def test_output_shapes(self, device, batch_size, num_classes, hidden_dim):
        """Test that output shapes match expected dimensions."""
        model = FeatureEnhancer(
            hidden_dim=hidden_dim,
            num_heads=8,
            num_layers=1
        ).to(device)
        
        N = 64  # Flattened spatial dimension
        text_features = torch.randn(batch_size, num_classes, hidden_dim, device=device)
        image_features = torch.randn(N, batch_size, hidden_dim, device=device)
        
        enhanced_text, enhanced_image = model(text_features, image_features)
        
        # Text features should maintain shape
        assert enhanced_text.shape == text_features.shape
        # Image features should maintain shape
        assert enhanced_image.shape == image_features.shape
    
    def test_different_spatial_sizes(self, device, batch_size, num_classes, hidden_dim):
        """Test with different spatial sizes."""
        model = FeatureEnhancer(hidden_dim=hidden_dim).to(device)
        
        text_features = torch.randn(batch_size, num_classes, hidden_dim, device=device)
        
        for N in [32, 64, 128, 256]:
            image_features = torch.randn(N, batch_size, hidden_dim, device=device)
            enhanced_text, enhanced_image = model(text_features, image_features)
            
            assert enhanced_image.shape == (N, batch_size, hidden_dim)
    
    def test_text_features_unchanged(self, device, batch_size, num_classes, hidden_dim):
        """Test that text features pass through unchanged (current implementation)."""
        model = FeatureEnhancer(hidden_dim=hidden_dim).to(device)
        model.eval()
        
        text_features = torch.randn(batch_size, num_classes, hidden_dim, device=device)
        image_features = torch.randn(64, batch_size, hidden_dim, device=device)
        
        with torch.no_grad():
            enhanced_text, _ = model(text_features, image_features)
        
        # Current implementation: text features are unchanged
        assert torch.allclose(enhanced_text, text_features)
    
    def test_image_features_enhanced(self, device, batch_size, num_classes, hidden_dim):
        """Test that image features are enhanced (not identical to input)."""
        model = FeatureEnhancer(hidden_dim=hidden_dim, num_layers=1).to(device)
        model.eval()
        
        text_features = torch.randn(batch_size, num_classes, hidden_dim, device=device)
        image_features = torch.randn(64, batch_size, hidden_dim, device=device)
        
        with torch.no_grad():
            _, enhanced_image = model(text_features, image_features)
        
        # Enhanced image features should be different from input
        assert not torch.allclose(enhanced_image, image_features)
    
    def test_gradient_flow(self, device, batch_size, num_classes, hidden_dim):
        """Test gradient flow through the module."""
        model = FeatureEnhancer(hidden_dim=hidden_dim).to(device)
        
        text_features = torch.randn(
            batch_size, num_classes, hidden_dim, 
            device=device, requires_grad=True
        )
        image_features = torch.randn(
            64, batch_size, hidden_dim, 
            device=device, requires_grad=True
        )
        
        enhanced_text, enhanced_image = model(text_features, image_features)
        loss = enhanced_text.sum() + enhanced_image.sum()
        loss.backward()
        
        # Image features should have gradients
        assert image_features.grad is not None
    
    def test_num_layers(self, device, batch_size, hidden_dim):
        """Test with different number of layers."""
        for num_layers in [1, 2, 3]:
            model = FeatureEnhancer(
                hidden_dim=hidden_dim,
                num_layers=num_layers
            ).to(device)
            
            text_features = torch.randn(batch_size, 5, hidden_dim, device=device)
            image_features = torch.randn(64, batch_size, hidden_dim, device=device)
            
            enhanced_text, enhanced_image = model(text_features, image_features)
            
            assert enhanced_text.shape == text_features.shape
            assert enhanced_image.shape == image_features.shape
    
    def test_deterministic_eval(self, device, batch_size, hidden_dim):
        """Test deterministic output in eval mode."""
        model = FeatureEnhancer(hidden_dim=hidden_dim).to(device)
        model.eval()
        
        text_features = torch.randn(batch_size, 5, hidden_dim, device=device)
        image_features = torch.randn(64, batch_size, hidden_dim, device=device)
        
        with torch.no_grad():
            out1_text, out1_image = model(text_features, image_features)
            out2_text, out2_image = model(text_features, image_features)
        
        assert torch.allclose(out1_text, out2_text)
        assert torch.allclose(out1_image, out2_image)
