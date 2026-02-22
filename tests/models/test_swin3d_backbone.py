"""
Unit tests for models/swin3d_backbone.py

Tests:
- PatchEmbed3D: output shape, patch dimensions
- WindowAttention3D: output shape, attention computation
- SwinTransformerBlock3D: output shape, residual connection
- PatchMerging3D: output shape, downsampling ratio
- SwinTransformer3D: end-to-end output shape
"""
import pytest
import torch
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from models.swin3d_backbone import (
    PatchEmbed3D,
    WindowAttention3D,
    SwinTransformerBlock3D,
    PatchMerging3D,
    SwinTransformer3D
)


class TestPatchEmbed3D:
    """Tests for PatchEmbed3D module."""
    
    def test_output_shape(self, device):
        """Test that output shape is correct after patch embedding."""
        batch_size = 2
        in_channels = 1
        D, H, W = 32, 64, 64
        patch_size = (4, 4, 4)
        embed_dim = 96
        
        model = PatchEmbed3D(
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim
        ).to(device)
        
        x = torch.randn(batch_size, in_channels, D, H, W, device=device)
        output = model(x)
        
        # Expected output: (B, D', H', W', embed_dim)
        # where D' = D // patch_size[0], etc.
        expected_D = D // patch_size[0]
        expected_H = H // patch_size[1]
        expected_W = W // patch_size[2]
        
        assert output.shape == (batch_size, expected_D, expected_H, expected_W, embed_dim)
    
    def test_different_patch_sizes(self, device):
        """Test with different patch sizes."""
        batch_size = 1
        D, H, W = 16, 32, 32
        
        for patch_size in [(2, 2, 2), (4, 4, 4), (2, 4, 4)]:
            model = PatchEmbed3D(patch_size=patch_size, embed_dim=48).to(device)
            x = torch.randn(batch_size, 1, D, H, W, device=device)
            output = model(x)
            
            expected_shape = (
                batch_size,
                D // patch_size[0],
                H // patch_size[1],
                W // patch_size[2],
                48
            )
            assert output.shape == expected_shape, f"Failed for patch_size={patch_size}"
    
    def test_gradient_flow(self, device):
        """Test that gradients flow through the module."""
        model = PatchEmbed3D(embed_dim=48).to(device)
        x = torch.randn(1, 1, 16, 32, 32, device=device, requires_grad=True)
        
        output = model(x)
        loss = output.sum()
        loss.backward()
        
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()


class TestWindowAttention3D:
    """Tests for WindowAttention3D module."""
    
    def test_output_shape(self, device):
        """Test that output shape matches input shape."""
        batch_size = 2
        window_size = (4, 4, 4)  # 指定窗口大小
        seq_len = 4 * 4 * 4  # = 64，与窗口大小匹配
        dim = 96
        
        model = WindowAttention3D(
            dim=dim, 
            num_heads=4,
            window_size=window_size  # 传入匹配的窗口大小
        ).to(device)
        x = torch.randn(batch_size, seq_len, dim, device=device)
        output = model(x)
        
        assert output.shape == x.shape
    
    def test_different_num_heads(self, device):
        """Test with different number of attention heads."""
        batch_size = 1
        window_size = (4, 4, 2)  # 4*4*2 = 32
        seq_len = 32
        dim = 96
        
        for num_heads in [1, 2, 4, 8]:
            # dim must be divisible by num_heads
            if dim % num_heads != 0:
                continue
            model = WindowAttention3D(dim=dim, num_heads=num_heads, window_size=window_size).to(device)
            x = torch.randn(batch_size, seq_len, dim, device=device)
            output = model(x)
            assert output.shape == x.shape
    
    def test_gradient_flow(self, device):
        """Test that gradients flow properly."""
        window_size = (2, 2, 4)  # 2*2*4 = 16
        model = WindowAttention3D(dim=64, num_heads=4, window_size=window_size).to(device)
        x = torch.randn(1, 16, 64, device=device, requires_grad=True)
        
        output = model(x)
        loss = output.sum()
        loss.backward()
        
        assert x.grad is not None


class TestSwinTransformerBlock3D:
    """Tests for SwinTransformerBlock3D module."""
    
    def test_output_shape(self, device):
        """Test that output shape matches input shape."""
        batch_size = 2
        D, H, W = 4, 8, 8
        dim = 96
        
        model = SwinTransformerBlock3D(dim=dim, num_heads=4).to(device)
        x = torch.randn(batch_size, D, H, W, dim, device=device)
        output = model(x)
        
        assert output.shape == x.shape
    
    def test_residual_connection(self, device):
        """Test that residual connection works (output != input but related)."""
        model = SwinTransformerBlock3D(dim=64, num_heads=4).to(device)
        model.eval()  # Disable dropout
        
        x = torch.randn(1, 4, 8, 8, 64, device=device)
        output = model(x)
        
        # Output should be different from input (due to attention + MLP)
        assert not torch.allclose(output, x)
    
    def test_gradient_flow(self, device):
        """Test gradient flow through the block."""
        model = SwinTransformerBlock3D(dim=64, num_heads=4).to(device)
        x = torch.randn(1, 4, 8, 8, 64, device=device, requires_grad=True)
        
        output = model(x)
        loss = output.sum()
        loss.backward()
        
        assert x.grad is not None


class TestPatchMerging3D:
    """Tests for PatchMerging3D module."""
    
    def test_output_shape(self, device):
        """Test that output shape is correctly downsampled."""
        batch_size = 2
        D, H, W = 8, 16, 16
        dim = 96
        
        model = PatchMerging3D(dim=dim).to(device)
        x = torch.randn(batch_size, D, H, W, dim, device=device)
        output = model(x)
        
        # Expected: spatial dimensions halved, channels doubled
        expected_shape = (batch_size, D // 2, H // 2, W // 2, dim * 2)
        assert output.shape == expected_shape
    
    def test_downsampling_ratio(self, device):
        """Test that downsampling ratio is 2x in each spatial dimension."""
        model = PatchMerging3D(dim=48).to(device)
        
        for D, H, W in [(4, 8, 8), (8, 16, 16), (16, 32, 32)]:
            x = torch.randn(1, D, H, W, 48, device=device)
            output = model(x)
            
            assert output.shape[1] == D // 2
            assert output.shape[2] == H // 2
            assert output.shape[3] == W // 2
    
    def test_channel_dimension(self, device):
        """Test that channel dimension is doubled."""
        for dim in [48, 96, 192]:
            model = PatchMerging3D(dim=dim).to(device)
            x = torch.randn(1, 4, 8, 8, dim, device=device)
            output = model(x)
            
            assert output.shape[-1] == dim * 2


class TestSwinTransformer3D:
    """Tests for SwinTransformer3D module."""
    
    def test_output_shape(self, device):
        """Test end-to-end output shape."""
        batch_size = 1
        D, H, W = 32, 64, 64
        embed_dim = 48
        
        model = SwinTransformer3D(
            in_channels=1,
            embed_dim=embed_dim,
            depths=[1, 1, 1, 1],
            num_heads=[2, 4, 8, 16]
        ).to(device)
        
        x = torch.randn(batch_size, 1, D, H, W, device=device)
        output = model(x)
        
        # Output should be (B, D', H', W', C')
        assert len(output.shape) == 5
        assert output.shape[0] == batch_size
        # Spatial dimensions should be reduced
        assert output.shape[1] < D
        assert output.shape[2] < H
        assert output.shape[3] < W
    
    def test_with_output_projection(self, device):
        """Test with output channel projection."""
        batch_size = 1
        out_channels = 256
        
        model = SwinTransformer3D(
            embed_dim=48,
            depths=[1, 1],
            num_heads=[2, 4],
            out_channels=out_channels,
            out_indices=(1,)  # depths=[1,1] means 2 stages: 0 and 1
        ).to(device)
        
        x = torch.randn(batch_size, 1, 16, 32, 32, device=device)
        output = model(x)
        
        assert output.shape[-1] == out_channels
    
    def test_gradient_flow(self, device):
        """Test gradient flow through entire backbone."""
        model = SwinTransformer3D(
            embed_dim=48,
            depths=[1, 1],
            num_heads=[2, 4],
            out_indices=(1,)  # depths=[1,1] means 2 stages: 0 and 1
        ).to(device)
        
        x = torch.randn(1, 1, 16, 32, 32, device=device, requires_grad=True)
        output = model(x)
        loss = output.sum()
        loss.backward()
        
        assert x.grad is not None
        # Check that model parameters have gradients
        for param in model.parameters():
            if param.requires_grad:
                assert param.grad is not None
    
    def test_deterministic_output(self, device):
        """Test that eval mode gives deterministic output."""
        model = SwinTransformer3D(
            embed_dim=48,
            depths=[1],
            num_heads=[2],
            out_indices=(0,)  # depths=[1] means 1 stage: only stage 0
        ).to(device)
        model.eval()
        
        x = torch.randn(1, 1, 16, 32, 32, device=device)
        
        with torch.no_grad():
            output1 = model(x)
            output2 = model(x)
        
        assert torch.allclose(output1, output2)


class TestBackboneFeatureQuality:
    """Tests for backbone feature extraction quality."""

    def test_overfit_simple_classification(self, device):
        """Test that backbone can learn to classify simple distinct distributions."""
        model = SwinTransformer3D(
            depths=[2, 2],
            embed_dim=24,
            in_channels=1,
            num_heads=[3, 6],
            patch_size=(2, 2, 2),
            window_size=(4, 4, 4),
            #out_indices=(1,)
        ).to(device)

        out_channels = 48
        classifier = torch.nn.Linear(out_channels, 2).to(device)

        optimizer = torch.optim.Adam(
            list(model.parameters()) + list(classifier.parameters()),
            lr=1e-2
        )
        criterion = torch.nn.CrossEntropyLoss()

        model.train()
        classifier.train()

        D, H, W = 16, 16, 16
        batch_size = 8
        final_loss = 1.0

        for _ in range(30):
            optimizer.zero_grad()

            x0 = torch.randn(batch_size // 2, 1, D, H, W, device=device)
            y0 = torch.zeros(batch_size // 2, dtype=torch.long, device=device)

            x1 = torch.randn(batch_size // 2, 1, D, H, W, device=device) + 2.0
            y1 = torch.ones(batch_size // 2, dtype=torch.long, device=device)

            x = torch.cat([x0, x1], dim=0)
            y = torch.cat([y0, y1], dim=0)

            features = model(x)
            gap = features.mean(dim=(1, 2, 3))

            logits = classifier(gap)
            loss = criterion(logits, y)

            loss.backward()
            optimizer.step()

            final_loss = loss.item()

        assert final_loss < 0.01, (
            f"Model failed to conform to simple distribution. Final loss: {final_loss}"
        )
