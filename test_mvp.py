"""
Test script for verifying the 3D Grounding-DETR implementation.
Tests data loading, model forward pass, and basic functionality.
"""
import torch
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from datasets import RSNAVolumeDataset, collate_fn
from models import GroundingDETR3D, build_model
from models.losses import HungarianMatcher, SetCriterion
from models.feature_enhancer import FeatureEnhancer, FeatureEnhancerLayer
from torch.utils.data import DataLoader


def test_data_loading():
    """Test data loading pipeline."""
    print("\n" + "="*60)
    print("TEST 1: Data Loading")
    print("="*60)
    
    try:
        dataset = RSNAVolumeDataset(
            data_dir='./datasets',
            volume_size=(64, 128, 128),
            train=True,
            augment=False,
            num_samples=1,
            image_format='dcm',  # 'dcm' or 'jpeg'
        )
        
        print(f"✓ Dataset created with {len(dataset)} samples")
        
        if len(dataset) > 0:
            sample = dataset[0]
            print(f"✓ Sample loaded:")
            print(f"  - Volume shape: {sample['volume'].shape}")
            print(f"  - Number of boxes: {len(sample['boxes'])}")
            print(f"  - Number of labels: {len(sample['labels'])}")
            print(f"  - Study ID: {sample['study_id']}")
            
            # Test dataloader
            loader = DataLoader(dataset, batch_size=1, collate_fn=collate_fn)
            batch = next(iter(loader))
            print(f"✓ DataLoader works:")
            print(f"  - Batch volumes shape: {batch['volumes'].shape}")
            print(f"  - Batch size: {len(batch['boxes'])}")
        
        print("\n✓ TEST 1 PASSED")
        return True
        
    except Exception as e:
        print(f"\n✗ TEST 1 FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_model_forward():
    """Test model forward pass."""
    print("\n" + "="*60)
    print("TEST 2: Model Forward Pass")
    print("="*60)
    
    try:
        # Create minimal config
        config = {
            'model': {
                'num_classes': 5,
                'num_queries': 10,  # Reduced for testing
                'hidden_dim': 128,  # Reduced for testing
                'backbone_embed_dim': 48,  # Reduced for testing
                'backbone_depths': [1, 1, 2, 1],  # Reduced for testing
                'backbone_num_heads': [2, 4, 8, 16],
                'num_encoder_layers': 2,  # Reduced for testing
                'num_decoder_layers': 2,  # Reduced for testing
                'num_heads': 4,
                'dim_feedforward': 512,
                'dropout': 0.1
            }
        }
        
        # Build model
        model = build_model(config)
        print(f"✓ Model created with {model.get_num_params():,} parameters")
        
        # Create dummy input
        batch_size = 2
        volume = torch.randn(batch_size, 1, 32, 64, 64)  # Smaller for testing
        print(f"✓ Created test input: {volume.shape}")
        
        # Forward pass
        model.eval()
        with torch.no_grad():
            outputs = model(volume)
        
        print(f"✓ Forward pass successful:")
        print(f"  - pred_logits shape: {outputs['pred_logits'].shape}")
        print(f"  - pred_boxes shape: {outputs['pred_boxes'].shape}")
        if 'class_tokens' in outputs:
            print(f"  - class_tokens shape: {outputs['class_tokens'].shape}")
        
        # Check output shapes
        assert outputs['pred_logits'].shape == (batch_size, config['model']['num_queries'], 
                                                config['model']['num_classes'] + 1)
        assert outputs['pred_boxes'].shape == (batch_size, config['model']['num_queries'], 6)
        
        print("\n✓ TEST 2 PASSED")
        return True
        
    except Exception as e:
        print(f"\n✗ TEST 2 FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_loss_computation():
    """Test loss computation."""
    print("\n" + "="*60)
    print("TEST 3: Loss Computation")
    print("="*60)
    
    try:
        num_classes = 5
        num_queries = 10
        
        # Create matcher and criterion
        matcher = HungarianMatcher(
            cost_class=1.0,
            cost_bbox=5.0,
            cost_giou=2.0
        )
        
        weight_dict = {
            'loss_ce': 2.0,
            'loss_l1': 5.0,
            'loss_giou': 2.0
        }
        
        criterion = SetCriterion(
            num_classes=num_classes,
            matcher=matcher,
            weight_dict=weight_dict,
            eos_coef=0.1
        )
        print("✓ Criterion created")
        
        # Create dummy predictions and targets
        batch_size = 2
        pred_logits = torch.randn(batch_size, num_queries, num_classes + 1)
        pred_boxes = torch.rand(batch_size, num_queries, 6)
        
        target_labels = [
            torch.tensor([1, 2], dtype=torch.int64),
            torch.tensor([0], dtype=torch.int64)
        ]
        target_boxes = [
            torch.rand(2, 6),
            torch.rand(1, 6)
        ]
        
        print("✓ Created dummy predictions and targets")
        
        # Compute loss
        losses = criterion(pred_logits, pred_boxes, target_labels, target_boxes)
        
        print(f"✓ Loss computation successful:")
        for k, v in losses.items():
            print(f"  - {k}: {v.item():.4f}")
        
        # Check that losses are finite
        assert all(torch.isfinite(v) for v in losses.values())
        
        print("\n✓ TEST 3 PASSED")
        return True
        
    except Exception as e:
        print(f"\n✗ TEST 3 FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_training_step():
    """Test a single training step."""
    print("\n" + "="*60)
    print("TEST 4: Training Step")
    print("="*60)
    
    try:
        # Create minimal config
        config = {
            'model': {
                'num_classes': 5,
                'num_queries': 10,
                'hidden_dim': 128,
                'backbone_embed_dim': 48,
                'backbone_depths': [1, 1, 2, 1],
                'backbone_num_heads': [2, 4, 8, 16],
                'num_encoder_layers': 2,
                'num_decoder_layers': 2,
                'num_heads': 4,
                'dim_feedforward': 512,
                'dropout': 0.1
            }
        }
        
        # Build model
        model = build_model(config)
        model.train()
        
        # Create criterion
        matcher = HungarianMatcher()
        weight_dict = {'loss_ce': 2.0, 'loss_l1': 5.0, 'loss_giou': 2.0}
        criterion = SetCriterion(
            num_classes=config['model']['num_classes'],
            matcher=matcher,
            weight_dict=weight_dict
        )
        
        # Create optimizer
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        
        print("✓ Model, criterion, and optimizer created")
        
        # Create dummy batch
        volumes = torch.randn(2, 1, 32, 64, 64)
        target_labels = [
            torch.tensor([1], dtype=torch.int64),
            torch.tensor([2], dtype=torch.int64)
        ]
        target_boxes = [
            torch.rand(1, 6),
            torch.rand(1, 6)
        ]
        
        # Training step
        optimizer.zero_grad()
        outputs = model(volumes)
        losses = criterion(
            outputs['pred_logits'],
            outputs['pred_boxes'],
            target_labels,
            target_boxes
        )
        loss = losses['loss_total']
        loss.backward()
        optimizer.step()
        
        print(f"✓ Training step successful:")
        print(f"  - Total loss: {loss.item():.4f}")
        
        # Check that gradients were computed
        has_grad = any(p.grad is not None for p in model.parameters())
        assert has_grad, "No gradients computed"
        
        print("\n✓ TEST 4 PASSED")
        return True
        
    except Exception as e:
        print(f"\n✗ TEST 4 FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_feature_enhancer():
    """Test Feature Enhancer module."""
    print("\n" + "="*60)
    print("TEST 5: Feature Enhancer")
    print("="*60)
    
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize Feature Enhancer
        hidden_dim = 256
        num_heads = 8
        dropout = 0.1
        num_layers = 2
        
        enhancer = FeatureEnhancer(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            num_layers=num_layers
        ).to(device)
        
        print(f"✓ Feature Enhancer created with {num_layers} layers")
        
        # Create dummy inputs
        batch_size = 2
        num_classes = 5
        num_spatial_tokens = 32  # (2x4x4 for 64x64x64 volume with 16x downsampling)
        
        text_features = torch.randn(batch_size, num_classes, hidden_dim, device=device)
        image_features = torch.randn(num_spatial_tokens, batch_size, hidden_dim, device=device)
        
        print(f"✓ Dummy inputs created:")
        print(f"  - Text features shape: {text_features.shape}")
        print(f"  - Image features shape: {image_features.shape}")
        
        # Forward pass
        enhanced_text, enhanced_image = enhancer(text_features, image_features)
        
        print(f"✓ Forward pass successful:")
        print(f"  - Enhanced text shape: {enhanced_text.shape}")
        print(f"  - Enhanced image shape: {enhanced_image.shape}")
        
        # Verify output shapes
        assert enhanced_text.shape == text_features.shape, \
            f"Text shape mismatch: {enhanced_text.shape} != {text_features.shape}"
        assert enhanced_image.shape == image_features.shape, \
            f"Image shape mismatch: {enhanced_image.shape} != {image_features.shape}"
        
        print(f"✓ Output shapes verified")
        
        # Test gradient flow
        loss = enhanced_text.sum() + enhanced_image.sum()
        loss.backward()
        
        has_grad = any(p.grad is not None for p in enhancer.parameters())
        assert has_grad, "No gradients computed"
        
        print(f"✓ Gradient flow verified")
        
        # Test with single layer
        enhancer_single = FeatureEnhancer(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            num_layers=1
        ).to(device)
        
        enhanced_text_single, enhanced_image_single = enhancer_single(text_features, image_features)
        assert enhanced_text_single.shape == text_features.shape
        assert enhanced_image_single.shape == image_features.shape
        
        print(f"✓ Single-layer variant works")
        
        # Test FeatureEnhancerLayer directly
        layer = FeatureEnhancerLayer(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout
        ).to(device)
        
        text_out, image_out = layer(text_features, image_features)
        assert text_out.shape == text_features.shape
        assert image_out.shape == image_features.shape
        
        print(f"✓ FeatureEnhancerLayer tested directly")
        
        print("\n✓ TEST 5 PASSED")
        return True
        
    except Exception as e:
        print(f"\n✗ TEST 5 FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("\n" + "="*60)
    print("3D GROUNDING-DETR MVP TEST SUITE")
    print("="*60)
    
    tests = [
        ("Data Loading", test_data_loading),
        ("Model Forward Pass", test_model_forward),
        ("Loss Computation", test_loss_computation),
        ("Training Step", test_training_step),
        ("Feature Enhancer", test_feature_enhancer)
    ]
    
    results = []
    for test_name, test_func in tests:
        passed = test_func()
        results.append((test_name, passed))
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    for test_name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {test_name}")
    
    total_passed = sum(1 for _, passed in results if passed)
    total_tests = len(results)
    
    print(f"\nTotal: {total_passed}/{total_tests} tests passed")
    print("="*60 + "\n")
    
    return all(passed for _, passed in results)


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
