"""
Activation functions and their derivatives.

Key concept (chain rule):
  - Forward: y = activation(x)
  - Backward: dy/dx = derivative(x)
  - Used in: dL/dz = dL/dy * dy/dx (propagating gradient backwards)
"""

import numpy as np


# ============= SIGMOID =============
def sigmoid(x):
    """
    Sigmoid function: σ(x) = 1 / (1 + e^(-x))
    Range: (0, 1)
    Use case: LSTM gates, binary classification
    """
    # Clip to prevent overflow
    x_clipped = np.clip(x, -500, 500)
    return 1 / (1 + np.exp(-x_clipped))


def sigmoid_derivative(x):
    """
    Sigmoid derivative: σ'(x) = σ(x) * (1 - σ(x))
    Note: x here is the PRE-ACTIVATION (z), not the output
    """
    sig = sigmoid(x)
    return sig * (1 - sig)


# ============= RELU =============
def relu(x):
    """
    ReLU (Rectified Linear Unit): max(0, x)
    Use case: Hidden layers (most common activation)
    Advantages: Simple, non-saturating, sparse activation
    """
    return np.maximum(0, x)


def relu_derivative(x):
    """
    ReLU derivative: 
      - 1 if x > 0
      - 0 if x <= 0
    """
    return (x > 0).astype(float)


# ============= TANH =============
def tanh(x):
    """
    Hyperbolic tangent: tanh(x) = (e^x - e^(-x)) / (e^x + e^(-x))
    Range: (-1, 1)
    Use case: LSTM cell state, RNN hidden layers
    Advantage: Centered at 0, converges faster than sigmoid
    """
    return np.tanh(x)


def tanh_derivative(x):
    """
    Tanh derivative: tanh'(x) = 1 - tanh^2(x)
    """
    t = np.tanh(x)
    return 1 - t ** 2


# ============= SOFTMAX =============
def softmax(x):
    """
    Softmax: converts logits to probability distribution
    s_i(x) = e^(x_i) / Σ(e^(x_j))  for all j
    
    Uses: Output layer for multi-class classification
    Returns: Probability distribution (sum = 1)
    
    Args:
        x: Input logits of shape (batch_size, num_classes)
    
    Returns:
        probabilities: Softmax output of shape (batch_size, num_classes)
    """
    # Subtract max for numerical stability
    x_shifted = x - np.max(x, axis=1, keepdims=True)
    exp_x = np.exp(x_shifted)
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)
