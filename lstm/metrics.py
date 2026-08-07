import numpy as np


def mse_loss(y_true, y_pred):
    """Mean squared error — shapes: y_pred is expected to be array-like shape (N,...)"""
    y_true = np.asarray(y_true, dtype=np.float32).reshape(y_pred.shape)
    y_pred = np.asarray(y_pred, dtype=np.float32)
    return np.mean((y_pred - y_true) ** 2)
