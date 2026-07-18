"""
Loss functions for training.

Key concept:
  - Loss = L(y_true, y_pred): measures prediction error
  - Gradient dL/dy_pred: used in backward pass to start gradient flow
"""

import numpy as np


class CrossEntropy:
    """
    Cross-Entropy Loss for multi-class classification.
    
    Formula:
      L = -Σ(y_true * log(y_pred))
    
    Where:
      - y_true: one-hot encoded targets (batch_size, num_classes)
      - y_pred: softmax probabilities (batch_size, num_classes)
    
    Gradient w.r.t. softmax output: dL/dy_pred = -y_true / y_pred
    But typically combined with softmax: dL/dz = (y_pred - y_true)
    """
    
    def __call__(self, y_true, y_pred):
        """
        Compute cross-entropy loss
        
        Args:
            y_true: One-hot encoded targets (batch_size, num_classes)
            y_pred: Softmax probabilities (batch_size, num_classes)
        
        Returns:
            loss: Scalar average loss
        """
        batch_size = y_true.shape[0]
        
        # Add small epsilon to prevent log(0)
        epsilon = 1e-7
        y_pred_clipped = np.clip(y_pred, epsilon, 1 - epsilon)
        
        # Cross-entropy: -Σ(y_true * log(y_pred))
        loss = -np.sum(y_true * np.log(y_pred_clipped)) / batch_size
        return loss
    
    def gradient(self, y_true, y_pred):
        """
        Compute gradient of loss w.r.t. output.
        
        When combined with softmax, this simplifies to:
        dL/dz = y_pred - y_true
        
        Args:
            y_true: One-hot encoded targets
            y_pred: Softmax probabilities
        
        Returns:
            gradient: dL/dz (same shape as y_pred)
        """
        # For softmax + cross-entropy, gradient is simply (y_pred - y_true)
        return y_pred - y_true
