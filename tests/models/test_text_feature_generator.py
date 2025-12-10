"""
Unit tests for models/text_feature_generator.py

Tests:
- PseudoTextFeatureGenerator: output shape, freeze/unfreeze, parameter count
"""
import pytest
import torch
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from models.text_feature_generator import PseudoTextFeatureGenerator


class TestPseudoTextFeatureGenerator:
    """Tests for PseudoTextFeatureGenerator module."""
    
    def test_output_shape(self, device, batch_size, num_classes, hidden_dim):
        """Test that output shape is (B, num_classes, hidden_dim)."""
        model = PseudoTextFeatureGenerator(
            num_classes=num_classes,
            hidden_dim=hidden_dim
        ).to(device)
        
        output = model(batch_size)
        
        assert output.shape == (batch_size, num_classes, hidden_dim)
    
    def test_different_batch_sizes(self, device, num_classes, hidden_dim):
        """Test with different batch sizes."""
        model = PseudoTextFeatureGenerator(
            num_classes=num_classes,
            hidden_dim=hidden_dim
        ).to(device)
        
        for bs in [1, 2, 4, 8]:
            output = model(bs)
            assert output.shape == (bs, num_classes, hidden_dim)
    
    def test_different_num_classes(self, device, batch_size, hidden_dim):
        """Test with different number of classes."""
        for nc in [3, 5, 10, 20]:
            model = PseudoTextFeatureGenerator(
                num_classes=nc,
                hidden_dim=hidden_dim
            ).to(device)
            
            output = model(batch_size)
            assert output.shape == (batch_size, nc, hidden_dim)
    
    def test_freeze_embeddings(self, device):
        """Test freezing embeddings."""
        model = PseudoTextFeatureGenerator(
            num_classes=5,
            hidden_dim=256,
            trainable_pseudo_features=True
        ).to(device)
        
        # Initially trainable
        assert model.class_embeddings.requires_grad
        
        # Freeze
        model.freeze_embeddings()
        assert not model.class_embeddings.requires_grad
        assert not model.trainable_pseudo_features
    
    def test_unfreeze_embeddings(self, device):
        """Test unfreezing embeddings."""
        model = PseudoTextFeatureGenerator(
            num_classes=5,
            hidden_dim=256,
            trainable_pseudo_features=False
        ).to(device)
        
        # Initially frozen
        assert not model.class_embeddings.requires_grad
        
        # Unfreeze
        model.unfreeze_embeddings()
        assert model.class_embeddings.requires_grad
        assert model.trainable_pseudo_features
    
    def test_get_num_params(self, device):
        """Test parameter counting."""
        model = PseudoTextFeatureGenerator(
            num_classes=5,
            hidden_dim=256,
            trainable_pseudo_features=True
        ).to(device)
        
        num_params = model.get_num_params()
        
        # Should be positive
        assert num_params > 0
        
        # Freeze and check again
        model.freeze_embeddings()
        num_params_frozen = model.get_num_params()
        
        # Fewer trainable params when embeddings frozen
        assert num_params_frozen < num_params
    
    def test_gradient_flow_trainable(self, device):
        """Test gradient flow when embeddings are trainable."""
        model = PseudoTextFeatureGenerator(
            num_classes=5,
            hidden_dim=128,
            trainable_pseudo_features=True
        ).to(device)
        
        output = model(batch_size=2)
        loss = output.sum()
        loss.backward()
        
        # Embeddings should have gradients
        assert model.class_embeddings.grad is not None
    
    def test_gradient_flow_frozen(self, device):
        """Test gradient flow when embeddings are frozen."""
        model = PseudoTextFeatureGenerator(
            num_classes=5,
            hidden_dim=128,
            trainable_pseudo_features=False
        ).to(device)
        
        output = model(batch_size=2)
        loss = output.sum()
        loss.backward()
        
        # Embeddings should not have gradients (frozen)
        assert model.class_embeddings.grad is None
    
    def test_batch_consistency(self, device):
        """Test that features are consistent across batch dimension."""
        model = PseudoTextFeatureGenerator(
            num_classes=5,
            hidden_dim=128
        ).to(device)
        model.eval()
        
        with torch.no_grad():
            output = model(batch_size=4)
        
        # All samples in batch should have identical features
        # (since they come from the same learned embeddings)
        for i in range(1, 4):
            assert torch.allclose(output[0], output[i])
    
    def test_output_dtype(self, device):
        """Test that output dtype is float32."""
        model = PseudoTextFeatureGenerator(num_classes=5, hidden_dim=128).to(device)
        output = model(batch_size=2)
        
        assert output.dtype == torch.float32
