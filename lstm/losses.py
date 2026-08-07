"""
Loss functions for training.

Key concept:
  - Loss = L(y_true, y_pred): measures prediction error
  - Gradient dL/dy_pred: used in backward pass to start gradient flow
"""

import numpy as np


class MSELoss:
    """
    Mean Squared Error loss for regression.

    Formula:
      L = mean((y_pred - y_true)^2)

    Where:
      - y_true: ground-truth targets, any shape
      - y_pred: network predictions, same shape as y_true

    Gradient w.r.t. y_pred:
      dL/dy_pred = 2 * (y_pred - y_true) / n
    """

    def __call__(self, y_true, y_pred):
        """
        Compute MSE loss.

        Args:
            y_true: Ground-truth targets (any shape)
            y_pred: Predictions (same shape as y_true)

        Returns:
            loss: Scalar average loss
        """
        y_true = np.asarray(y_true, dtype=np.float32).reshape(y_pred.shape)
        y_pred = np.asarray(y_pred, dtype=np.float32)
        return float(np.mean((y_pred - y_true) ** 2))

    def gradient(self, y_true, y_pred):
        """
        Compute gradient of MSE w.r.t. y_pred.

        Args:
            y_true: Ground-truth targets
            y_pred: Predictions

        Returns:
            gradient: dL/dy_pred (same shape as y_pred)
        """
        y_true = np.asarray(y_true, dtype=np.float32).reshape(y_pred.shape)
        y_pred = np.asarray(y_pred, dtype=np.float32)
        diff = y_pred - y_true
        return 2.0 * diff / diff.size
