"""
Sanity check model: Large fully-connected network for overfitting test.
Used to verify training pipeline can reduce loss and overfit the data.
"""
import torch
import torch.nn as nn
from typing import Dict


class LargeFullyConnectedNet(nn.Module):
    """
    Large fully-connected network for sanity checking.
    
    This model is designed to:
    1. Have enough capacity to easily overfit the training data
    2. Match the same input/output interface as GroundingDETR3D
    3. Verify that the loss function and training loop work correctly
    
    If this model cannot reduce loss or overfit, it indicates issues with:
    - Loss function implementation
    - Data pipeline
    - Training configuration
    """
    
    def __init__(
        self,
        input_size: tuple = (128, 128, 128),
        num_queries: int = 100,
        num_classes: int = 5,
        hidden_dims: list = None
    ):
        """
        Args:
            input_size: Input volume size (D, H, W)
            num_queries: Number of queries to predict
            num_classes: Number of object classes
            hidden_dims: List of hidden layer dimensions
        """
        super().__init__()
        
        if hidden_dims is None:
            # large enough to overfit
            hidden_dims = [4096, 2048, 1024, 512]
        
        self.num_queries = num_queries
        self.num_classes = num_classes
        
        # Calculate input dimension
        input_dim = input_size[0] * input_size[1] * input_size[2]
        
        # Calculate output dimension
        # For each query: (num_classes+1) for logits + 6 for bbox
        output_dim_per_query = (num_classes + 1) + 6
        output_dim = num_queries * output_dim_per_query
        
        # Build layers
        layers = []
        
        # Input layer
        layers.append(nn.Flatten())
        layers.append(nn.Linear(input_dim, hidden_dims[0]))
        layers.append(nn.LayerNorm(hidden_dims[0]))  # Add layer norm for stability
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Dropout(0.1))
        
        # Hidden layers
        for i in range(len(hidden_dims) - 1):
            layers.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))
            layers.append(nn.LayerNorm(hidden_dims[i + 1]))  # Add layer norm
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(0.1))
        
        # Output layer with small initialization
        self.output_layer = nn.Linear(hidden_dims[-1], output_dim)
        
        self.network = nn.Sequential(*layers)
        
        # Store dimensions for reshaping
        self.output_dim_per_query = output_dim_per_query
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize network weights with careful scaling."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # Use Xavier for better stability than Kaiming
                nn.init.xavier_uniform_(m.weight, gain=0.5)  # Reduced gain
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
        # Special initialization for output layer - small but not too small
        # gain=0.01 was too small, outputs stuck at 0
        # gain=0.1 provides better initialization while maintaining stability
        nn.init.xavier_uniform_(self.output_layer.weight, gain=0.1)
        nn.init.constant_(self.output_layer.bias, 0)
    
    def forward(self, volumes: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            volumes: (B, 1, D, H, W) - CT volumes
        
        Returns:
            dict with:
                - 'pred_logits': (B, num_queries, num_classes+1)
                - 'pred_boxes': (B, num_queries, 6)
        """
        B = volumes.shape[0]
        
        # Forward through main network
        features = self.network(volumes)  # (B, hidden_dim)
        
        # Apply output layer
        output = self.output_layer(features)  # (B, num_queries * output_dim_per_query)
        
        # Reshape to (B, num_queries, output_dim_per_query)
        output = output.view(B, self.num_queries, self.output_dim_per_query)
        
        # Split into logits and boxes
        pred_logits = output[:, :, :self.num_classes + 1]  # (B, num_queries, num_classes+1)
        pred_boxes_raw = output[:, :, self.num_classes + 1:]  # (B, num_queries, 6)
        
        # Apply sigmoid to boxes to normalize to [0, 1]
        pred_boxes = torch.sigmoid(pred_boxes_raw)
        
        return {
            'pred_logits': pred_logits,
            'pred_boxes': pred_boxes
        }
    
    def get_num_params(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def build_sanity_check_model(config: dict) -> LargeFullyConnectedNet:
    """
    Build large FC network for sanity checking.
    
    Args:
        config: Configuration dictionary
    
    Returns:
        Initialized LargeFullyConnectedNet
    """
    model_config = config.get('model', {})
    data_config = config.get('data', {})
    
    # Get target width from data config and compute a default input size
    # The actual input size varies per sample, but we need a fixed size for this simple model
    target_width = data_config.get('target_width', 64)
    # Assume a typical volume ratio, using target_width for all dimensions as a baseline
    input_size = (target_width, target_width, target_width)
    
    model = LargeFullyConnectedNet(
        input_size=input_size,
        num_queries=model_config.get('num_queries', 100),
        num_classes=model_config.get('num_classes', 5),
        hidden_dims=[4096, 2048, 1024, 512]
    )
    
    num_params = model.get_num_params()
    print(f"Sanity Check Model built with {num_params:,} parameters")
    print(f"  Input size: {input_size}")
    print(f"  Num queries: {model.num_queries}")
    print(f"  Num classes: {model.num_classes}")
    
    return model
