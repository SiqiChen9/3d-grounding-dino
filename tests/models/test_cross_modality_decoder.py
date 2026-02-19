"""
Unit tests for models/cross_modality_decoder.py

Tests:
- PositionalEncoding3D: output shape, positional encoding
- CrossModalityDecoderLayer: output shape, cross-attention
- MLP: output shape, layer count
- CrossModalityDecoder: output shape, classification and regression heads
"""
import pytest
import torch
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from models.cross_modality_decoder import (
    PositionalEncoding3D,
    CrossModalityDecoderLayer,
    MLP,
    CrossModalityDecoder
)


class TestPositionalEncoding3D:
    """Tests for PositionalEncoding3D module."""
    
    def test_output_shape(self, device, batch_size, hidden_dim):
        """Test that output shape matches input shape."""
        model = PositionalEncoding3D(d_model=hidden_dim).to(device)
        
        seq_len = 64
        x = torch.randn(batch_size, seq_len, hidden_dim, device=device)
        output = model(x)
        
        assert output.shape == x.shape
    
    def test_positional_encoding_added(self, device, batch_size, hidden_dim):
        """Test that positional encoding is added to input."""
        model = PositionalEncoding3D(d_model=hidden_dim).to(device)
        
        x = torch.zeros(batch_size, 32, hidden_dim, device=device)
        output = model(x)
        
        # Output should not be all zeros (positional encoding added)
        assert not torch.allclose(output, x)
    
    def test_different_sequence_lengths(self, device, batch_size, hidden_dim):
        """Test with different sequence lengths."""
        model = PositionalEncoding3D(d_model=hidden_dim, max_len=1000).to(device)
        
        for seq_len in [32, 64, 128, 256]:
            x = torch.randn(batch_size, seq_len, hidden_dim, device=device)
            output = model(x)
            assert output.shape == x.shape
    
    def test_gradient_flow(self, device, hidden_dim):
        """Test gradient flow."""
        model = PositionalEncoding3D(d_model=hidden_dim).to(device)
        
        x = torch.randn(1, 32, hidden_dim, device=device, requires_grad=True)
        output = model(x)
        loss = output.sum()
        loss.backward()
        
        assert x.grad is not None


class TestCrossModalityDecoderLayer:
    """Tests for CrossModalityDecoderLayer module."""
    
    def test_output_shape(self, device, batch_size, num_classes, hidden_dim, num_queries):
        """Test that output shape is (num_queries, B, hidden_dim)."""
        model = CrossModalityDecoderLayer(
            d_model=hidden_dim,
            num_heads=8
        ).to(device)
        
        N = 64
        queries = torch.randn(num_queries, batch_size, hidden_dim, device=device)
        text_features = torch.randn(num_classes, batch_size, hidden_dim, device=device)
        image_features = torch.randn(N, batch_size, hidden_dim, device=device)
        
        output = model(queries, text_features, image_features)
        
        assert output.shape == queries.shape
    
    def test_queries_updated(self, device, batch_size, num_classes, hidden_dim, num_queries):
        """Test that queries are updated (not identical to input)."""
        model = CrossModalityDecoderLayer(d_model=hidden_dim, num_heads=8).to(device)
        model.eval()
        
        queries = torch.randn(num_queries, batch_size, hidden_dim, device=device)
        text_features = torch.randn(num_classes, batch_size, hidden_dim, device=device)
        image_features = torch.randn(64, batch_size, hidden_dim, device=device)
        
        with torch.no_grad():
            output = model(queries, text_features, image_features)
        
        # Output should be different from input
        assert not torch.allclose(output, queries)
    
    def test_gradient_flow(self, device, batch_size, num_classes, hidden_dim, num_queries):
        """Test gradient flow through all inputs."""
        model = CrossModalityDecoderLayer(d_model=hidden_dim, num_heads=8).to(device)
        
        queries = torch.randn(num_queries, batch_size, hidden_dim, device=device, requires_grad=True)
        text_features = torch.randn(num_classes, batch_size, hidden_dim, device=device, requires_grad=True)
        image_features = torch.randn(64, batch_size, hidden_dim, device=device, requires_grad=True)
        
        output = model(queries, text_features, image_features)
        loss = output.sum()
        loss.backward()
        
        assert queries.grad is not None
        assert text_features.grad is not None
        assert image_features.grad is not None


class TestMLP:
    """Tests for MLP module."""
    
    def test_output_shape(self, device, hidden_dim):
        """Test output shape matches expected output_dim."""
        input_dim = 256
        output_dim = 6  # For box regression
        
        model = MLP(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            num_layers=3
        ).to(device)
        
        x = torch.randn(2, 100, input_dim, device=device)
        output = model(x)
        
        assert output.shape == (2, 100, output_dim)
    
    def test_different_layer_counts(self, device):
        """Test with different number of layers (>=2 for proper output_dim)."""
        input_dim = 128
        output_dim = 6  # Explicitly set for box regression
        
        # Note: MLP with num_layers=1 outputs hidden_dim, not output_dim
        # So we test with num_layers >= 2
        for num_layers in [2, 3, 4]:
            model = MLP(
                input_dim=input_dim,
                hidden_dim=256,
                output_dim=output_dim,
                num_layers=num_layers
            ).to(device)
            
            x = torch.randn(1, 50, input_dim, device=device)
            output = model(x)
            
            assert output.shape == (1, 50, output_dim)
    
    def test_gradient_flow(self, device):
        """Test gradient flow."""
        model = MLP(input_dim=128, hidden_dim=256, output_dim=6, num_layers=3).to(device)
        
        x = torch.randn(1, 50, 128, device=device, requires_grad=True)
        output = model(x)
        loss = output.sum()
        loss.backward()
        
        assert x.grad is not None


class TestCrossModalityDecoder:
    """Tests for CrossModalityDecoder module."""
    
    def test_output_shapes(self, device, batch_size, num_classes, hidden_dim, num_queries):
        """Test pred_logits and pred_boxes shapes."""
        model = CrossModalityDecoder(
            hidden_dim=hidden_dim,
            num_queries=num_queries,
            num_classes=num_classes,
            num_decoder_layers=2,
            num_heads=8
        ).to(device)
        
        N = 64
        image_features = torch.randn(N, batch_size, hidden_dim, device=device)
        text_features = torch.randn(batch_size, num_classes, hidden_dim, device=device)
        queries = torch.randn(num_queries, batch_size, hidden_dim, device=device)
        
        pred_logits, pred_boxes = model(image_features, text_features, queries)
        
        # pred_logits: (B, num_queries, num_classes + 1)
        assert pred_logits.shape == (batch_size, num_queries, num_classes + 1)
        # pred_boxes: (B, num_queries, 6)
        assert pred_boxes.shape == (batch_size, num_queries, 6)
    
    def test_pred_boxes_normalized(self, device, batch_size, num_classes, hidden_dim, num_queries):
        """Test that predicted boxes are in [0, 1] range (sigmoid applied)."""
        model = CrossModalityDecoder(
            hidden_dim=hidden_dim,
            num_queries=num_queries,
            num_classes=num_classes,
            num_decoder_layers=1
        ).to(device)
        model.eval()
        
        image_features = torch.randn(64, batch_size, hidden_dim, device=device)
        text_features = torch.randn(batch_size, num_classes, hidden_dim, device=device)
        queries = torch.randn(num_queries, batch_size, hidden_dim, device=device)
        
        with torch.no_grad():
            _, pred_boxes = model(image_features, text_features, queries)
        
        # Check boxes are in [0, 1]
        assert (pred_boxes >= 0).all()
        assert (pred_boxes <= 1).all()
    
    def test_gradient_flow(self, device, batch_size, num_classes, hidden_dim, num_queries):
        """Test gradient flow through decoder."""
        model = CrossModalityDecoder(
            hidden_dim=hidden_dim,
            num_queries=num_queries,
            num_classes=num_classes,
            num_decoder_layers=1
        ).to(device)
        
        image_features = torch.randn(64, batch_size, hidden_dim, device=device, requires_grad=True)
        text_features = torch.randn(batch_size, num_classes, hidden_dim, device=device, requires_grad=True)
        queries = torch.randn(num_queries, batch_size, hidden_dim, device=device, requires_grad=True)
        
        pred_logits, pred_boxes = model(image_features, text_features, queries)
        loss = pred_logits.sum() + pred_boxes.sum()
        loss.backward()
        
        assert image_features.grad is not None
        assert text_features.grad is not None
        assert queries.grad is not None
    
    def test_different_decoder_layers(self, device, batch_size, num_classes, hidden_dim, num_queries):
        """Test with different number of decoder layers."""
        for num_layers in [1, 2, 4, 6]:
            model = CrossModalityDecoder(
                hidden_dim=hidden_dim,
                num_queries=num_queries,
                num_classes=num_classes,
                num_decoder_layers=num_layers
            ).to(device)
            
            image_features = torch.randn(64, batch_size, hidden_dim, device=device)
            text_features = torch.randn(batch_size, num_classes, hidden_dim, device=device)
            queries = torch.randn(num_queries, batch_size, hidden_dim, device=device)
            
            pred_logits, pred_boxes = model(image_features, text_features, queries)
            
            assert pred_logits.shape == (batch_size, num_queries, num_classes + 1)
            assert pred_boxes.shape == (batch_size, num_queries, 6)
