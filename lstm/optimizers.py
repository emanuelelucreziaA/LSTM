"""
Optimizers for weight updates.

Key concept:
  - Optimizer receives gradient (dL/dW, dL/db)
  - Returns update step: W_new = W_old - update
  - Different strategies: SGD (simple), Adam (adaptive learning rates)
"""

import numpy as np


class SGD:
    """
    Stochastic Gradient Descent with optional momentum.
    
    Update rule (no momentum):
      W_new = W - learning_rate * dW
    
    With momentum (classic momentum):
      v = momentum * v + learning_rate * dW
      W_new = W - v
    
    Args:
        learning_rate: Step size for updates (typically 0.001 - 0.1)
        momentum: Acceleration parameter (typically 0.9, use 0 to disable)
    """
    
    def __init__(self, learning_rate=0.01, momentum=0.0):
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.velocity_W = None
        self.velocity_b = None
    
    def update(self, dW, db):
        """
        Compute update step for weights and biases.
        
        Args:
            dW: Gradient w.r.t. weights
            db: Gradient w.r.t. biases
        
        Returns:
            update_W, update_b: Update steps to subtract from W and b
        """
        # Initialize velocity on first call
        if self.velocity_W is None:
            self.velocity_W = np.zeros_like(dW)
            self.velocity_b = np.zeros_like(db)
        
        # Momentum update
        self.velocity_W = self.momentum * self.velocity_W + self.learning_rate * dW
        self.velocity_b = self.momentum * self.velocity_b + self.learning_rate * db
        
        return self.velocity_W, self.velocity_b


class Adam:
    """
    Adam Optimizer (Adaptive Moment Estimation).
    Combines momentum and RMSprop for adaptive learning rates.
    
    Paper: Kingma & Ba, 2014
    
    Update rules:
      m_t = β1 * m_(t-1) + (1 - β1) * g_t           (first moment estimate)
      v_t = β2 * v_(t-1) + (1 - β2) * g_t^2         (second moment estimate)
      m_hat = m_t / (1 - β1^t)                       (bias correction)
      v_hat = v_t / (1 - β2^t)
      θ = θ - α * m_hat / (√v_hat + ε)
    
    Args:
        learning_rate: Step size (default: 0.001)
        beta1: Exponential decay for 1st moment (default: 0.9)
        beta2: Exponential decay for 2nd moment (default: 0.999)
        epsilon: Small constant for numerical stability (default: 1e-8)
    """
    
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        
        # First moment estimates (mean)
        self.m_W = None
        self.m_b = None
        
        # Second moment estimates (variance)
        self.v_W = None
        self.v_b = None
        
        # Time step counter for bias correction
        self.t = 0
    
    def update(self, dW, db):
        """
        Perform Adam update step.
        
        Args:
            dW: Gradient w.r.t. weights
            db: Gradient w.r.t. biases
        
        Returns:
            update_W, update_b: Adaptive update steps
        """
        # Initialize on first call
        if self.m_W is None:
            self.m_W = np.zeros_like(dW)
            self.m_b = np.zeros_like(db)
            self.v_W = np.zeros_like(dW)
            self.v_b = np.zeros_like(db)
        
        self.t += 1
        
        # Update biased first moment estimate
        self.m_W = self.beta1 * self.m_W + (1 - self.beta1) * dW
        self.m_b = self.beta1 * self.m_b + (1 - self.beta1) * db
        
        # Update biased second raw moment estimate
        self.v_W = self.beta2 * self.v_W + (1 - self.beta2) * (dW ** 2)
        self.v_b = self.beta2 * self.v_b + (1 - self.beta2) * (db ** 2)
        
        # Bias correction
        m_W_hat = self.m_W / (1 - self.beta1 ** self.t)
        m_b_hat = self.m_b / (1 - self.beta1 ** self.t)
        v_W_hat = self.v_W / (1 - self.beta2 ** self.t)
        v_b_hat = self.v_b / (1 - self.beta2 ** self.t)
        
        # Compute updates
        update_W = self.learning_rate * m_W_hat / (np.sqrt(v_W_hat) + self.epsilon)
        update_b = self.learning_rate * m_b_hat / (np.sqrt(v_b_hat) + self.epsilon)
        
        return update_W, update_b
