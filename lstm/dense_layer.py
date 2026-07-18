"""
Dense (Fully Connected) Layer for output.

Similar to MLP's DenseLayer but reused as output layer after LSTM.
"""

import numpy as np


class DenseLayer:
    """
    Fully connected layer for neural network.
    
    Performs: output = activation(input @ W + b)
    
    Args:
        input_size: Number of input features
        output_size: Number of output features (neurons)
        activation_fn: Activation function (e.g., relu, sigmoid)
        activation_derivative: Derivative of activation function
    """
    
    def __init__(self, input_size, output_size, activation_fn=None, activation_derivative=None):
        self.input_size = input_size
        self.output_size = output_size
        self.activation_fn = activation_fn
        self.activation_derivative = activation_derivative
        
        # He initialization for weights (optimal for ReLU)
        self.W = np.random.randn(input_size, output_size) * np.sqrt(2.0 / input_size)
        self.b = np.zeros((1, output_size))
        
        # Gradients (computed during backward pass)
        self.dW = None
        self.db = None
        
        # Cache for backward pass
        self.input_cache = None
        self.z_cache = None  # Pre-activation (z = x @ W + b)
    
    def forward(self, x):
        """
        Forward pass: compute output = activation(x @ W + b)
        
        Args:
            x: Input of shape (batch_size, input_size)
        
        Returns:
            output: Activated output of shape (batch_size, output_size)
        """
        # Ensure x is 2D
        if x.ndim == 1:
            x = x.reshape(1, -1)
        
        # Cache input for backward pass
        self.input_cache = x
        
        # Linear transformation: z = x @ W + b
        self.z_cache = np.dot(x, self.W) + self.b
        
        # Apply activation if provided
        if self.activation_fn is not None:
            output = self.activation_fn(self.z_cache)
        else:
            output = self.z_cache
        
        return output
    
    def backward(self, dL_dout):
        """
        Backward pass: compute gradients using chain rule.
        
        Args:
            dL_dout: Gradient of loss w.r.t. output of shape (batch_size, output_size)
        
        Returns:
            dL_dinput: Gradient of loss w.r.t. input of shape (batch_size, input_size)
        """
        batch_size = self.input_cache.shape[0]
        
        # If activation function exists, apply its derivative
        if self.activation_derivative is not None:
            dL_dz = dL_dout * self.activation_derivative(self.z_cache)
        else:
            dL_dz = dL_dout
        
        # Compute gradients
        self.dW = np.dot(self.input_cache.T, dL_dz) / batch_size
        self.db = np.sum(dL_dz, axis=0, keepdims=True) / batch_size
        
        # Propagate to previous layer: dL/dinput = dL/dz @ W^T
        dL_dinput = np.dot(dL_dz, self.W.T)
        
        return dL_dinput
    
    def get_gradients(self):
        """Return computed gradients"""
        return self.dW, self.db
    
    def update_parameters(self, dW_update, db_update):
        """Update parameters using optimizer updates"""
        self.W -= dW_update
        self.b -= db_update
