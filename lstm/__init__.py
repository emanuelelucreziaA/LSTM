"""
LSTM (Long Short-Term Memory) Neural Network Package

A from-scratch implementation of LSTM networks for sequence processing.
Includes LSTM cells, layers, and network composition.
"""

# Core layer imports - lightweight, no dependencies
from .lstm_cell import LSTMCell
from .lstm_layer import LSTMLayer
from .dense_layer import DenseLayer
from .network import LSTMNetwork

# Activation functions
from .activations import (
    sigmoid, sigmoid_derivative,
    tanh, tanh_derivative,
    relu, relu_derivative,
    softmax
)

# Loss and optimizer imports
from .losses import CrossEntropy
from .optimizers import SGD, Adam

# Lazy import for data utilities to avoid slow I/O at import time
def __getattr__(name):
    """Lazy import data utilities on demand"""
    if name == 'MNISTLoader':
        from .data import MNISTLoader
        return MNISTLoader
    elif name == 'SequenceDataLoader':
        from .data import SequenceDataLoader
        return SequenceDataLoader
    elif name == 'one_hot_encode':
        from .data import one_hot_encode
        return one_hot_encode
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    'LSTMCell', 'LSTMLayer', 'DenseLayer', 'LSTMNetwork',
    'sigmoid', 'sigmoid_derivative', 'tanh', 'tanh_derivative',
    'relu', 'relu_derivative', 'softmax',
    'CrossEntropy', 'SGD', 'Adam',
    'MNISTLoader', 'SequenceDataLoader', 'one_hot_encode'
]
