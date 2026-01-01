# pytorch mlp (multi-layer perceptron) for fraud detection
# this is a simple neural network for tabular data

import torch
import torch.nn as nn

class FraudMLP(nn.Module):
    """
    multi-layer perceptron for binary fraud classification
    
    architecture:
        input (38 features) → hidden1 (128) → hidden2 (64) → hidden3 (32) → output (1)
    
    uses relu activation and dropout for regularization
    """
    
    def __init__(self, input_dim: int = 38, hidden_dims: list = None):
        """
        initialize model
        
        args:
            input_dim: number of input features (default 38)
            hidden_dims: list of hidden layer sizes (default [128, 64, 32])
        """
        super().__init__()
        
        if hidden_dims is None:
            hidden_dims = [128, 64, 32]
        
        layers = []
        prev_dim = input_dim
        
        # build hidden layers
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.3))  # dropout to prevent overfitting
            prev_dim = hidden_dim
        
        # output layer (binary classification)
        layers.append(nn.Linear(prev_dim, 1))
        # note: we use BCEWithLogitsLoss which includes sigmoid, so no sigmoid here
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        """
        forward pass
        
        args:
            x: input tensor of shape (batch_size, input_dim)
            
        returns:
            logits of shape (batch_size,) - raw scores before sigmoid
        """
        return self.network(x).squeeze()
    
    def predict_proba(self, x):
        """
        get probability predictions (after sigmoid)
        
        args:
            x: input tensor
            
        returns:
            probabilities between 0 and 1
        """
        with torch.no_grad():
            logits = self.forward(x)
            probs = torch.sigmoid(logits)
            return probs

def count_parameters(model):
    """count trainable parameters in model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
