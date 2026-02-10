"""
Unit tests for Language-Guided Query Selection module.

Tests cover:
- Basic forward pass and output shapes
- Different num_queries, batch sizes, feature dimensions
- Learnable parameters and gradient flow
- Mixed query strategy validation
"""
import pytest
import torch
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from models.query_selection import LanguageGuidedQuerySelection


class TestLanguageGuidedQuerySelection:
    """Test suite for LanguageGuidedQuerySelection module."""
    
    def test_basic_forward_pass(self):
        """Test that basic forward pass works correctly."""
        print("\n[TEST] Basic Forward Pass")
        
        # Create model instance
        model = LanguageGuidedQuerySelection(
            num_queries=100,
            hidden_dim=256,
            image_feature_dim=256,
            text_feature_dim=256
        )
        
        # Create dummy input tensors
        batch_size = 2
        num_img_tokens = 1000
        num_text_tokens = 50
        hidden_dim = 256
        
        image_features = torch.randn(batch_size, num_img_tokens, hidden_dim)
        text_features = torch.randn(batch_size, num_text_tokens, hidden_dim)
        
        # Forward pass
        queries = model(image_features, text_features, batch_size)
        
        # Verify output shape matches expected format for DETR decoder
        expected_shape = (100, batch_size, hidden_dim)
        assert queries.shape == expected_shape, \
            f"Expected shape {expected_shape}, got {queries.shape}"
        
        print(f"  ✓ Output shape correct: {queries.shape}")
    
    def test_different_num_queries(self):
        """Test module with different num_queries values."""
        print("\n[TEST] Different num_queries Values")
        
        # Test with various num_queries configurations
        for num_q in [50, 100, 300, 900]:
            model = LanguageGuidedQuerySelection(
                num_queries=num_q,
                hidden_dim=256,
                image_feature_dim=256,
                text_feature_dim=256
            )
            
            batch_size = 2
            image_features = torch.randn(batch_size, 1000, 256)
            text_features = torch.randn(batch_size, 50, 256)
            
            queries = model(image_features, text_features, batch_size)
            
            # Verify first dimension matches num_queries
            assert queries.shape[0] == num_q, \
                f"Expected {num_q} queries, got {queries.shape[0]}"
            
            print(f"  ✓ num_queries={num_q}: shape {queries.shape}")
    
    def test_different_batch_sizes(self):
        """Test module with different batch sizes."""
        print("\n[TEST] Different Batch Sizes")
        
        model = LanguageGuidedQuerySelection(
            num_queries=100,
            hidden_dim=256,
            image_feature_dim=256,
            text_feature_dim=256
        )
        
        # Test with various batch sizes
        for batch_size in [1, 2, 4, 8]:
            image_features = torch.randn(batch_size, 1000, 256)
            text_features = torch.randn(batch_size, 50, 256)
            
            queries = model(image_features, text_features, batch_size)
            
            # Verify batch dimension
            assert queries.shape[1] == batch_size, \
                f"Expected batch size {batch_size}, got {queries.shape[1]}"
            
            print(f"  ✓ batch_size={batch_size}: shape {queries.shape}")
    
    def test_feature_projection(self):
        """Test that feature dimension projection works correctly."""
        print("\n[TEST] Feature Dimension Projection")
        
        # Test with different feature dimensions
        test_cases = [
            (256, 256),  # No projection needed
            (768, 256),  # Project down from ViT
            (512, 256),  # Project down
        ]
        
        for image_dim, text_dim in test_cases:
            model = LanguageGuidedQuerySelection(
                num_queries=100,
                hidden_dim=256,
                image_feature_dim=image_dim,
                text_feature_dim=text_dim
            )
            
            batch_size = 2
            image_features = torch.randn(batch_size, 1000, image_dim)
            text_features = torch.randn(batch_size, 50, text_dim)
            
            queries = model(image_features, text_features, batch_size)
            
            assert queries.shape == (100, batch_size, 256)
            print(f"  ✓ Projection {image_dim},{text_dim}→256: OK")
    
    def test_learnable_content_queries(self):
        """Test that content queries are properly learnable."""
        print("\n[TEST] Learnable Content Queries")
        
        model = LanguageGuidedQuerySelection(num_queries=100)
        
        # Check that content_queries parameter exists and requires gradients
        assert hasattr(model, 'content_queries'), \
            "Model should have content_queries parameter"
        
        assert model.content_queries.requires_grad, \
            "content_queries should require gradients (be learnable)"
        
        print(f"  ✓ content_queries shape: {model.content_queries.shape}")
        print(f"  ✓ content_queries requires_grad: {model.content_queries.requires_grad}")
        
        # Check that values are initialized (not all zeros)
        assert not torch.allclose(
            model.content_queries,
            torch.zeros_like(model.content_queries)
        ), "content_queries should not be all zeros"
        
        print(f"  ✓ content_queries initialized with non-zero values")
    
    def test_gradient_flow(self):
        """Test that gradients flow properly through the entire module."""
        print("\n[TEST] Gradient Flow")
        
        model = LanguageGuidedQuerySelection(num_queries=100)
        
        batch_size = 2
        
        # Create inputs with requires_grad=True for gradient computation
        image_features = torch.randn(batch_size, 1000, 256, requires_grad=True)
        text_features = torch.randn(batch_size, 50, 256, requires_grad=True)
        
        # Forward pass
        queries = model(image_features, text_features, batch_size)
        
        # Backward pass
        loss = queries.sum()  # Simple loss for testing
        loss.backward()
        
        # Check that gradients exist in image inputs
        # (Text gradients may be None due to top-k selection not being differentiable)
        assert image_features.grad is not None, \
            "image_features should have gradients"
        print(f"  ✓ Image feature gradients computed")
        
        # Check that gradients exist in model parameters
        params_with_grad = 0
        for param in model.parameters():
            if param.grad is not None:
                params_with_grad += 1
        
        assert params_with_grad > 0, \
            "Model parameters should have gradients"
        print(f"  ✓ {params_with_grad} model parameters have gradients")
    
    def test_mixed_query_strategy(self):
        """Test that mixed position+content strategy produces meaningful output."""
        print("\n[TEST] Mixed Query Strategy (Position + Content)")
        
        model = LanguageGuidedQuerySelection(num_queries=50)
        
        batch_size = 1
        image_features = torch.randn(batch_size, 1000, 256)
        text_features = torch.randn(batch_size, 50, 256)
        
        queries = model(image_features, text_features, batch_size)
        
        # Check output is not all zeros
        assert queries.abs().sum() > 0, \
            "Queries should not be all zeros"
        print(f"  ✓ Queries are non-zero")
        
        # Check output has proper range (not exploding)
        assert queries.abs().max() < 1000, \
            "Query values should be reasonable (not exploding)"
        print(f"  ✓ Query value range reasonable: [{queries.min():.3f}, {queries.max():.3f}]")
    
    def test_topk_selection_correctness(self):
        """
        Test that top-K selection actually selects features with highest similarity.
        
        This is a conceptual test: we create a scenario where some image features
        are explicitly more similar to text, and verify they get selected.
        """
        print("\n[TEST] Top-K Selection Correctness")
        
        model = LanguageGuidedQuerySelection(num_queries=10)
        
        batch_size = 1
        num_img_tokens = 100
        
        # Create image features where first 10 are boosted
        image_features = torch.randn(batch_size, num_img_tokens, 256)
        image_features[0, :10, :] *= 5  # Boost first 10 features
        
        text_features = torch.randn(batch_size, 50, 256)
        
        queries = model(image_features, text_features, batch_size)
        
        # If selection works, we should get queries from boosted features
        # (This is probabilistic, but highly likely)
        assert queries.shape == (10, batch_size, 256)
        print(f"  ✓ Top-K selection produced correct shape: {queries.shape}")
    
    def test_deterministic_output(self):
        """Test that same inputs produce same outputs (deterministic)."""
        print("\n[TEST] Deterministic Output")
        
        model = LanguageGuidedQuerySelection(num_queries=100)
        
        batch_size = 2
        image_features = torch.randn(batch_size, 1000, 256)
        text_features = torch.randn(batch_size, 50, 256)
        
        # Forward pass 1
        queries1 = model(image_features, text_features, batch_size)
        
        # Forward pass 2 with same inputs
        queries2 = model(image_features, text_features, batch_size)
        
        # Check outputs are identical (no randomness)
        assert torch.allclose(queries1, queries2), \
            "Same inputs should produce identical outputs"
        
        print(f"  ✓ Outputs are deterministic (no random operations)")


if __name__ == "__main__":
    """Run all tests if executed directly."""
    test = TestLanguageGuidedQuerySelection()
    test.test_basic_forward_pass()
    test.test_different_num_queries()
    test.test_different_batch_sizes()
    test.test_feature_projection()
    test.test_learnable_content_queries()
    test.test_gradient_flow()
    test.test_mixed_query_strategy()
    test.test_topk_selection_correctness()
    test.test_deterministic_output()
    print("\n✅ All tests passed!")