"""Test basic LSTM operations"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
from lstm.lstm_cell import LSTMCell
from lstm.lstm_layer import LSTMLayer
from lstm.dense_layer import DenseLayer


def test_lstm_cell():
    """Test LSTM cell forward/backward pass"""
    print("Testing LSTM Cell...")
    
    batch_size = 4
    input_size = 28
    hidden_size = 64
    
    # Create cell
    cell = LSTMCell(input_size, hidden_size)
    
    # Test forward pass
    x_t = np.random.randn(batch_size, input_size)
    h_prev = np.zeros((batch_size, hidden_size))
    C_prev = np.zeros((batch_size, hidden_size))
    
    h_t, C_t = cell.forward(x_t, h_prev, C_prev)
    
    assert h_t.shape == (batch_size, hidden_size), f"Wrong h_t shape: {h_t.shape}"
    assert C_t.shape == (batch_size, hidden_size), f"Wrong C_t shape: {C_t.shape}"
    
    print("✓ Forward pass OK")
    
    # Test backward pass
    dh_t = np.random.randn(batch_size, hidden_size)
    dC_t = np.zeros((batch_size, hidden_size))
    
    dx_t, dh_prev, dC_prev = cell.backward(dh_t, dC_t)
    
    assert dx_t.shape == (batch_size, input_size), f"Wrong dx_t shape: {dx_t.shape}"
    assert dh_prev.shape == (batch_size, hidden_size), f"Wrong dh_prev shape: {dh_prev.shape}"
    assert dC_prev.shape == (batch_size, hidden_size), f"Wrong dC_prev shape: {dC_prev.shape}"
    
    print("✓ Backward pass OK")


def test_lstm_layer():
    """Test LSTM layer with sequences"""
    print("\nTesting LSTM Layer...")
    
    batch_size = 4
    seq_len = 28
    input_size = 28
    hidden_size = 64
    
    # Create layer
    layer = LSTMLayer(input_size, hidden_size)
    
    # Test forward pass
    X = np.random.randn(batch_size, seq_len, input_size)
    H, h_final, C_final = layer.forward(X)
    
    assert H.shape == (batch_size, seq_len, hidden_size), f"Wrong H shape: {H.shape}"
    assert h_final.shape == (batch_size, hidden_size), f"Wrong h_final shape: {h_final.shape}"
    assert C_final.shape == (batch_size, hidden_size), f"Wrong C_final shape: {C_final.shape}"
    
    print("✓ Forward pass OK")


def test_dense_layer():
    """Test dense output layer"""
    print("\nTesting Dense Layer...")
    
    batch_size = 4
    input_size = 128
    output_size = 10
    
    layer = DenseLayer(input_size, output_size)
    
    # Forward pass
    x = np.random.randn(batch_size, input_size)
    y = layer.forward(x)
    
    assert y.shape == (batch_size, output_size), f"Wrong output shape: {y.shape}"
    print("✓ Forward pass OK")
    
    # Backward pass
    dL_dy = np.random.randn(batch_size, output_size)
    dL_dx = layer.backward(dL_dy)
    
    assert dL_dx.shape == (batch_size, input_size), f"Wrong gradient shape: {dL_dx.shape}"
    print("✓ Backward pass OK")


if __name__ == '__main__':
    print("="*70)
    print("Running LSTM Unit Tests")
    print("="*70)
    
    test_lstm_cell()
    test_lstm_layer()
    test_dense_layer()
    
    print("\n" + "="*70)
    print("All tests passed! ✓")
    print("="*70)
