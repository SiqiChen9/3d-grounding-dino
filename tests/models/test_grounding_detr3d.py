"""
Unit tests for models/grounding_detr3d.py

Tests:
- GroundingDETR3D: forward pass output shapes, pred_logits and pred_boxes format
- build_model: model construction from config
"""
import pytest
import torch
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from models.grounding_detr3d import GroundingDETR3D, build_model


class TestGroundingDETR3D:
    """Tests for GroundingDETR3D module."""
    
    @pytest.fixture
    def small_model(self, device, num_classes, num_queries, hidden_dim):
        """Create a small model for testing (faster)."""
        return GroundingDETR3D(
            num_classes=num_classes,
            num_queries=num_queries,
            hidden_dim=hidden_dim,
            backbone_embed_dim=48,
            backbone_depths=[1, 1],
            backbone_num_heads=[2, 4],
            num_encoder_layers=1,
            num_decoder_layers=1,
            num_heads=4,
            dim_feedforward=256
        ).to(device)
    
    def test_forward_output_keys(self, small_model, device, batch_size, small_volume_size):
        """Test that forward returns expected keys."""
        D, H, W = small_volume_size
        x = torch.randn(batch_size, 1, D, H, W, device=device)
        
        output = small_model(x)
        
        assert 'pred_logits' in output
        assert 'pred_boxes' in output
        assert 'vanilla_text_features' in output
        assert 'vanilla_image_features' in output
    
    def test_pred_logits_shape(self, small_model, device, batch_size, small_volume_size, num_classes, num_queries):
        """Test pred_logits shape is (B, num_queries, num_classes+1)."""
        D, H, W = small_volume_size
        x = torch.randn(batch_size, 1, D, H, W, device=device)
        
        output = small_model(x)
        
        assert output['pred_logits'].shape == (batch_size, num_queries, num_classes + 1)
    
    def test_pred_boxes_shape(self, small_model, device, batch_size, small_volume_size, num_queries):
        """Test pred_boxes shape is (B, num_queries, 6)."""
        D, H, W = small_volume_size
        x = torch.randn(batch_size, 1, D, H, W, device=device)
        
        output = small_model(x)
        
        assert output['pred_boxes'].shape == (batch_size, num_queries, 6)
    
    def test_pred_boxes_normalized(self, small_model, device, batch_size, small_volume_size):
        """Test predicted boxes are in [0, 1] range."""
        D, H, W = small_volume_size
        x = torch.randn(batch_size, 1, D, H, W, device=device)
        
        small_model.eval()
        with torch.no_grad():
            output = small_model(x)
        
        pred_boxes = output['pred_boxes']
        assert (pred_boxes >= 0).all()
        assert (pred_boxes <= 1).all()
    
    def test_gradient_flow(self, small_model, device, batch_size, small_volume_size):
        """Test gradient flow through the model."""
        D, H, W = small_volume_size
        x = torch.randn(batch_size, 1, D, H, W, device=device, requires_grad=True)
        
        output = small_model(x)
        loss = output['pred_logits'].sum() + output['pred_boxes'].sum()
        loss.backward()
        
        assert x.grad is not None
        
        # Check model parameters have gradients
        grad_count = 0
        for param in small_model.parameters():
            if param.requires_grad and param.grad is not None:
                grad_count += 1
        assert grad_count > 0
    
    def test_eval_mode_deterministic(self, small_model, device, batch_size, small_volume_size):
        """Test eval mode produces deterministic output."""
        D, H, W = small_volume_size
        x = torch.randn(batch_size, 1, D, H, W, device=device)
        
        small_model.eval()
        with torch.no_grad():
            output1 = small_model(x)
            output2 = small_model(x)
        
        assert torch.allclose(output1['pred_logits'], output2['pred_logits'])
        assert torch.allclose(output1['pred_boxes'], output2['pred_boxes'])
    
    def test_different_batch_sizes(self, small_model, device, small_volume_size, num_queries):
        """Test with different batch sizes."""
        D, H, W = small_volume_size
        
        small_model.eval()
        for bs in [1, 2, 4]:
            x = torch.randn(bs, 1, D, H, W, device=device)
            with torch.no_grad():
                output = small_model(x)
            
            assert output['pred_logits'].shape[0] == bs
            assert output['pred_boxes'].shape[0] == bs
    
    def test_get_num_params(self, small_model):
        """Test parameter counting."""
        num_params = small_model.get_num_params()
        
        assert num_params > 0
        assert isinstance(num_params, int)

    def test_multi_scale_increases_tokens(self, device, num_classes, num_queries, hidden_dim, small_volume_size):
        """Test that multi-scale backbone produces more image tokens than single-scale."""
        D, H, W = small_volume_size

        # Multi-scale model: out_indices=(0, 1) — default for 2-stage backbone
        multi_model = GroundingDETR3D(
            num_classes=num_classes,
            num_queries=num_queries,
            hidden_dim=hidden_dim,
            backbone_embed_dim=48,
            backbone_depths=[1, 1],
            backbone_num_heads=[2, 4],
            backbone_out_indices=(0, 1),
            num_encoder_layers=1,
            num_decoder_layers=1,
            num_heads=4,
            dim_feedforward=256,
        ).to(device)

        multi_model.eval()
        x = torch.randn(1, 1, D, H, W, device=device)
        with torch.no_grad():
            output = multi_model(x)

        # vanilla_image_features should be (N_total, B, hidden_dim) with N_total > single-scale
        img_feat = output['vanilla_image_features']
        assert img_feat.dim() == 3
        # For 16×32×32 input with 2 stages: stage0=4×8×8=256, stage1=2×4×4=32 → 288
        assert img_feat.shape[0] == 288
        assert img_feat.shape[2] == hidden_dim

    def test_explicit_backbone_out_indices(self, device, num_classes, hidden_dim, small_volume_size):
        """Test explicit backbone_out_indices=(0,) for single-stage legacy mode."""
        D, H, W = small_volume_size
        num_queries = 50

        model = GroundingDETR3D(
            num_classes=num_classes,
            num_queries=num_queries,
            hidden_dim=hidden_dim,
            backbone_embed_dim=48,
            backbone_depths=[1, 1, 1],
            backbone_num_heads=[2, 4, 8],
            backbone_out_indices=(2,),  # single last stage → legacy path
            num_encoder_layers=1,
            num_decoder_layers=1,
            num_heads=4,
            dim_feedforward=256,
        ).to(device)

        model.eval()
        x = torch.randn(1, 1, D, H, W, device=device)
        with torch.no_grad():
            output = model(x)

        assert output['pred_logits'].shape == (1, num_queries, num_classes + 1)
        assert output['pred_boxes'].shape == (1, num_queries, 6)


class TestBuildModel:
    """Tests for build_model function."""
    
    def test_build_from_config(self, model_config, device):
        """Test building model from configuration dict."""
        model = build_model(model_config).to(device)
        
        assert isinstance(model, GroundingDETR3D)
    
    def test_built_model_forward(self, model_config, device):
        """Test forward pass of built model."""
        model = build_model(model_config).to(device)
        model.eval()
        
        x = torch.randn(1, 1, 16, 32, 32, device=device)
        
        with torch.no_grad():
            output = model(x)
        
        assert 'pred_logits' in output
        assert 'pred_boxes' in output
    
    def test_config_parameters_applied(self, device):
        """Test that config parameters are correctly applied."""
        num_classes = 3
        num_queries = 50
        
        # build_model expects config with nested 'model' key
        config = {
            'model': {
                'num_classes': num_classes,
                'num_queries': num_queries,
                'hidden_dim': 128,
                'backbone_embed_dim': 32,
                'backbone_depths': [1, 1],  # Need at least 2 stages for proper downsampling
                'backbone_num_heads': [2, 4],  # Must match backbone_depths length
                'num_encoder_layers': 1,
                'num_decoder_layers': 1,
                'num_heads': 4,
                'dim_feedforward': 256,
                'dropout': 0.0,
                'trainable_pseudo_features': False
            }
        }
        
        model = build_model(config).to(device)
        model.eval()
        
        x = torch.randn(2, 1, 16, 32, 32, device=device)
        
        with torch.no_grad():
            output = model(x)
        
        # Check num_classes + 1 in logits (background class added)
        assert output['pred_logits'].shape[-1] == num_classes + 1
        # Check num_queries
        assert output['pred_logits'].shape[1] == num_queries
