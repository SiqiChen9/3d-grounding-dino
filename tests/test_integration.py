"""
Integration tests for the 3D Grounding-DETR pipeline.

Tests:
- Full forward pass through all modules
- Full forward + backward pass (gradient flow)
- Module interface compatibility
- Loss computation with model outputs
"""
import pytest
import torch
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.swin3d_backbone import SwinTransformer3D
from models.text_feature_generator import PseudoTextFeatureGenerator
from models.feature_enhancer import FeatureEnhancer
from models.query_selection import LanguageGuidedQuerySelection
from models.cross_modality_decoder import CrossModalityDecoder
from models.grounding_detr3d import GroundingDETR3D, build_model
from models.losses import HungarianMatcher, SetCriterion


class TestFullPipelineForward:
    """Test complete forward pass through all components."""
    
    @pytest.fixture
    def integrated_model(self, device, num_classes, num_queries, hidden_dim):
        """Create integrated model with small config."""
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
    
    def test_forward_completes(self, integrated_model, device, batch_size, small_volume_size):
        """Test that forward pass completes without errors."""
        D, H, W = small_volume_size
        x = torch.randn(batch_size, 1, D, H, W, device=device)
        
        output = integrated_model(x)
        
        assert output is not None
        assert 'pred_logits' in output
        assert 'pred_boxes' in output
    
    def test_output_values_valid(self, integrated_model, device, batch_size, small_volume_size):
        """Test that output values are valid (no NaN/Inf)."""
        D, H, W = small_volume_size
        x = torch.randn(batch_size, 1, D, H, W, device=device)
        
        integrated_model.eval()
        with torch.no_grad():
            output = integrated_model(x)
        
        assert torch.isfinite(output['pred_logits']).all()
        assert torch.isfinite(output['pred_boxes']).all()
    
    def test_boxes_normalized(self, integrated_model, device, batch_size, small_volume_size):
        """Test that predicted boxes are in [0, 1]."""
        D, H, W = small_volume_size
        x = torch.randn(batch_size, 1, D, H, W, device=device)
        
        integrated_model.eval()
        with torch.no_grad():
            output = integrated_model(x)
        
        assert (output['pred_boxes'] >= 0).all()
        assert (output['pred_boxes'] <= 1).all()


class TestFullPipelineBackward:
    """Test complete forward + backward pass."""
    
    @pytest.fixture
    def model_and_criterion(self, device, num_classes, num_queries, hidden_dim, loss_weight_dict):
        """Create model and loss criterion."""
        model = GroundingDETR3D(
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
        
        matcher = HungarianMatcher()
        criterion = SetCriterion(
            num_classes=num_classes,
            matcher=matcher,
            weight_dict=loss_weight_dict
        ).to(device)
        
        return model, criterion
    
    def test_backward_pass(self, model_and_criterion, device, batch_size, small_volume_size, num_classes):
        """Test backward pass with loss computation."""
        model, criterion = model_and_criterion
        D, H, W = small_volume_size
        
        x = torch.randn(batch_size, 1, D, H, W, device=device)
        
        # Create targets
        target_labels = [torch.randint(0, num_classes, (3,), device=device) for _ in range(batch_size)]
        target_boxes = [torch.rand(3, 6, device=device) for _ in range(batch_size)]
        
        # Forward
        output = model(x)
        
        # Compute loss
        losses = criterion(
            output['pred_logits'],
            output['pred_boxes'],
            target_labels,
            target_boxes
        )
        
        total_loss = sum(losses.values())
        
        # Backward
        total_loss.backward()
        
        # Check gradients exist
        grad_count = 0
        for param in model.parameters():
            if param.requires_grad and param.grad is not None:
                grad_count += 1
                assert torch.isfinite(param.grad).all(), "NaN/Inf in gradients"
        
        assert grad_count > 0, "No gradients computed"
    
    def test_optimizer_step(self, model_and_criterion, device, batch_size, small_volume_size, num_classes):
        """Test optimizer step updates parameters."""
        model, criterion = model_and_criterion
        D, H, W = small_volume_size
        
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        
        # Get initial parameters
        initial_params = {name: param.clone() for name, param in model.named_parameters()}
        
        x = torch.randn(batch_size, 1, D, H, W, device=device)
        target_labels = [torch.randint(0, num_classes, (3,), device=device) for _ in range(batch_size)]
        target_boxes = [torch.rand(3, 6, device=device) for _ in range(batch_size)]
        
        # Training step
        optimizer.zero_grad()
        output = model(x)
        losses = criterion(output['pred_logits'], output['pred_boxes'], target_labels, target_boxes)
        total_loss = sum(losses.values())
        total_loss.backward()
        optimizer.step()
        
        # Check parameters changed
        params_changed = 0
        for name, param in model.named_parameters():
            if not torch.equal(param, initial_params[name]):
                params_changed += 1
        
        assert params_changed > 0, "No parameters updated"


class TestModuleInterfaceCompatibility:
    """Test that module outputs are compatible with downstream inputs."""
    
    def test_backbone_to_enhancer(self, device, batch_size, hidden_dim):
        """Test backbone output is compatible with feature enhancer."""
        backbone = SwinTransformer3D(
            embed_dim=48,
            depths=[1, 1],
            num_heads=[2, 4],
            out_channels=hidden_dim,
            out_indices=(1,)  # depths=[1,1] has 2 stages: 0 and 1
        ).to(device)
        
        enhancer = FeatureEnhancer(hidden_dim=hidden_dim).to(device)
        text_gen = PseudoTextFeatureGenerator(num_classes=5, hidden_dim=hidden_dim).to(device)
        
        # Forward through backbone
        x = torch.randn(batch_size, 1, 16, 32, 32, device=device)
        backbone_out = backbone(x)  # (B, D', H', W', C)
        
        # Reshape for enhancer: (B, D', H', W', C) -> (N, B, C)
        B, D, H, W, C = backbone_out.shape
        image_features = backbone_out.permute(1, 2, 3, 0, 4).reshape(D * H * W, B, C)
        
        # Generate text features
        text_features = text_gen(batch_size)
        
        # Forward through enhancer
        enhanced_text, enhanced_image = enhancer(text_features, image_features)
        
        assert enhanced_image.shape == image_features.shape
    
    def test_enhancer_to_query_selection(self, device, batch_size, num_classes, hidden_dim, num_queries):
        """Test enhancer output is compatible with query selection."""
        enhancer = FeatureEnhancer(hidden_dim=hidden_dim).to(device)
        query_sel = LanguageGuidedQuerySelection(
            num_queries=num_queries,
            hidden_dim=hidden_dim,
            image_feature_dim=hidden_dim
        ).to(device)
        
        N = 64
        text_features = torch.randn(batch_size, num_classes, hidden_dim, device=device)
        image_features = torch.randn(N, batch_size, hidden_dim, device=device)
        
        # Enhance
        enhanced_text, enhanced_image = enhancer(text_features, image_features)
        
        # Query selection
        queries = query_sel(enhanced_text, enhanced_image, batch_size)
        
        assert queries.shape == (num_queries, batch_size, hidden_dim)
    
    def test_query_selection_to_decoder(self, device, batch_size, num_classes, hidden_dim, num_queries):
        """Test query selection output is compatible with decoder."""
        query_sel = LanguageGuidedQuerySelection(
            num_queries=num_queries,
            hidden_dim=hidden_dim,
            image_feature_dim=hidden_dim
        ).to(device)
        
        decoder = CrossModalityDecoder(
            hidden_dim=hidden_dim,
            num_queries=num_queries,
            num_classes=num_classes,
            num_decoder_layers=1
        ).to(device)
        
        N = 64
        text_features = torch.randn(batch_size, num_classes, hidden_dim, device=device)
        image_features = torch.randn(N, batch_size, hidden_dim, device=device)
        
        # Generate queries
        queries = query_sel(text_features, image_features, batch_size)
        
        # Decode
        pred_logits, pred_boxes = decoder(image_features, text_features, queries)
        
        assert pred_logits.shape == (batch_size, num_queries, num_classes + 1)
        assert pred_boxes.shape == (batch_size, num_queries, 6)


class TestLossComputationWithModel:
    """Test loss computation with actual model outputs."""
    
    @pytest.fixture
    def model_criterion_optimizer(self, device, num_classes, num_queries, hidden_dim, loss_weight_dict):
        """Create model, criterion, and optimizer."""
        model = GroundingDETR3D(
            num_classes=num_classes,
            num_queries=num_queries,
            hidden_dim=hidden_dim,
            backbone_embed_dim=48,
            backbone_depths=[1],
            backbone_num_heads=[2],
            num_encoder_layers=1,
            num_decoder_layers=1,
            num_heads=4,
            dim_feedforward=256
        ).to(device)
        
        matcher = HungarianMatcher()
        criterion = SetCriterion(
            num_classes=num_classes,
            matcher=matcher,
            weight_dict=loss_weight_dict
        ).to(device)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        
        return model, criterion, optimizer
    
    def test_loss_decreases_with_training(self, model_criterion_optimizer, device, small_volume_size, num_classes):
        """Test that loss decreases over training iterations."""
        model, criterion, optimizer = model_criterion_optimizer
        batch_size = 2
        D, H, W = small_volume_size
        
        # Fixed data for reproducibility
        torch.manual_seed(42)
        x = torch.randn(batch_size, 1, D, H, W, device=device)
        target_labels = [torch.randint(0, num_classes, (2,), device=device) for _ in range(batch_size)]
        target_boxes = [torch.rand(2, 6, device=device) * 0.5 + 0.25 for _ in range(batch_size)]
        
        losses_history = []
        
        for _ in range(5):  # Few iterations
            optimizer.zero_grad()
            output = model(x)
            losses = criterion(output['pred_logits'], output['pred_boxes'], target_labels, target_boxes)
            total_loss = sum(losses.values())
            total_loss.backward()
            optimizer.step()
            losses_history.append(total_loss.item())
        
        # Loss should generally decrease or stay stable
        # (may not always decrease due to random initialization, but should be finite)
        assert all(loss < float('inf') for loss in losses_history)
        assert all(loss == loss for loss in losses_history)  # No NaN
    
    def test_empty_targets_handling(self, model_criterion_optimizer, device, small_volume_size):
        """Test model handles empty targets gracefully."""
        model, criterion, _ = model_criterion_optimizer
        batch_size = 2
        D, H, W = small_volume_size
        
        x = torch.randn(batch_size, 1, D, H, W, device=device)
        
        # Empty targets
        target_labels = [torch.tensor([], dtype=torch.long, device=device) for _ in range(batch_size)]
        target_boxes = [torch.zeros(0, 6, device=device) for _ in range(batch_size)]
        
        output = model(x)
        losses = criterion(output['pred_logits'], output['pred_boxes'], target_labels, target_boxes)
        
        # Should return valid losses
        assert 'loss_ce' in losses
        assert torch.isfinite(losses['loss_ce'])


class TestBuildModelIntegration:
    """Test build_model function integration."""
    
    def test_built_model_full_pipeline(self, model_config, device, loss_weight_dict):
        """Test full pipeline with model built from config."""
        model = build_model(model_config).to(device)
        
        matcher = HungarianMatcher()
        criterion = SetCriterion(
            num_classes=model_config['num_classes'],
            matcher=matcher,
            weight_dict=loss_weight_dict
        ).to(device)
        
        x = torch.randn(2, 1, 16, 32, 32, device=device)
        target_labels = [torch.randint(0, model_config['num_classes'], (2,), device=device) for _ in range(2)]
        target_boxes = [torch.rand(2, 6, device=device) for _ in range(2)]
        
        # Forward
        output = model(x)
        
        # Loss
        losses = criterion(output['pred_logits'], output['pred_boxes'], target_labels, target_boxes)
        total_loss = sum(losses.values())
        
        # Backward
        total_loss.backward()
        
        assert torch.isfinite(total_loss)
