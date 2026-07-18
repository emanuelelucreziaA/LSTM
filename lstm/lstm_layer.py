"""
LSTM Layer: processes entire sequences using LSTM cells.

Manages:
- Stacking LSTM cells across timesteps
- Forward pass through entire sequence
- Backward pass (BPTT) through entire sequence
- Managing cell and hidden states
"""

import copy
import numpy as np
from .lstm_cell import LSTMCell


class LSTMLayer:
    """
    LSTM Layer: processes sequences of variable length.
    
    Args:
        input_size: Size of input at each timestep
        hidden_size: Size of hidden state
    """
    
    def __init__(self, input_size, hidden_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # Use a single LSTM cell that processes all timesteps
        self.lstm_cell = LSTMCell(input_size, hidden_size)
        
        # Cache for backward pass
        self.cache = None
        self.step_caches = None
        self.gate_optimizers = None

    def set_optimizer(self, optimizer):
        """Create independent optimizer state per LSTM gate."""
        if optimizer is None:
            raise ValueError("Optimizer cannot be None")
        self.gate_optimizers = [copy.deepcopy(optimizer) for _ in range(4)]
    
    def forward(self, X):
        """
        Forward pass: process entire sequence.
        
        Args:
            X: Input sequence of shape (batch_size, seq_len, input_size)
        
        Returns:
            H: All hidden states (batch_size, seq_len, hidden_size)
            h_final: Final hidden state (batch_size, hidden_size)
            C_final: Final cell state (batch_size, hidden_size)
        """
        batch_size, seq_len, _ = X.shape
        
        # Initialize hidden and cell states
        h_t = np.zeros((batch_size, self.hidden_size))
        C_t = np.zeros((batch_size, self.hidden_size))
        
        # Store all hidden states
        H = np.zeros((batch_size, seq_len, self.hidden_size))
        
        # Process each timestep
        self.step_caches = []
        for t in range(seq_len):
            x_t = X[:, t, :]  # Input at timestep t (batch_size, input_size)
            h_t, C_t = self.lstm_cell.forward(x_t, h_t, C_t)
            H[:, t, :] = h_t
            self.step_caches.append(self.lstm_cell.cache)

        # Cache for backward pass
        self.cache = (X, H, seq_len)

        return H, h_t, C_t
    
    def backward(self, dL_dH, dL_dh_final=None, dL_dC_final=None):
        """
        Backward pass: BPTT (Backpropagation Through Time).
        
        Args:
            dL_dH: Gradient of loss w.r.t. all hidden states (batch_size, seq_len, hidden_size)
            dL_dh_final: Gradient w.r.t. final hidden state (optional)
            dL_dC_final: Gradient w.r.t. final cell state (optional)
        
        Returns:
            dL_dX: Gradient w.r.t. input (batch_size, seq_len, input_size)
        """
        if self.cache is None or self.step_caches is None:
            raise ValueError("Must call forward before backward")
        
        X, H, seq_len = self.cache
        batch_size = X.shape[0]
        
        # Initialize gradients
        dX = np.zeros_like(X)
        dh_t = np.zeros((batch_size, self.hidden_size), dtype=np.float32)
        dC_t = np.zeros((batch_size, self.hidden_size), dtype=np.float32)
        
        # Add final state gradients if provided
        if dL_dh_final is not None:
            dh_t += dL_dh_final
        if dL_dC_final is not None:
            dC_t += dL_dC_final

        if dL_dH is None:
            dL_dH = np.zeros_like(H)
        
        # Initialize accumulated gradients for all gates
        total_dW_f = np.zeros_like(self.lstm_cell.W_f)
        total_db_f = np.zeros_like(self.lstm_cell.b_f)
        total_dW_i = np.zeros_like(self.lstm_cell.W_i)
        total_db_i = np.zeros_like(self.lstm_cell.b_i)
        total_dW_c = np.zeros_like(self.lstm_cell.W_c)
        total_db_c = np.zeros_like(self.lstm_cell.b_c)
        total_dW_o = np.zeros_like(self.lstm_cell.W_o)
        total_db_o = np.zeros_like(self.lstm_cell.b_o)
        
        # Backward through time
        for t in range(seq_len - 1, -1, -1):
            dh_t += dL_dH[:, t, :]
            
            # Restore the cache for this timestep
            self.lstm_cell.cache = self.step_caches[t]
            
            # Backprop through the LSTM cell
            dx_t, dh_prev, dC_prev = self.lstm_cell.backward(dh_t, dC_t)
            
            # Store input gradient
            dX[:, t, :] = dx_t
            
            # Accumulate gradients across timesteps
            grads = self.lstm_cell.get_gradients()
            total_dW_f += grads['dW_f']
            total_db_f += grads['db_f']
            total_dW_i += grads['dW_i']
            total_db_i += grads['db_i']
            total_dW_c += grads['dW_c']
            total_db_c += grads['db_c']
            total_dW_o += grads['dW_o']
            total_db_o += grads['db_o']
            
            # Prepare gradients for previous timestep
            dh_t = dh_prev
            dC_t = dC_prev
        
        # Store accumulated gradients in the LSTM cell
        self.lstm_cell.set_gradients(
            total_dW_f, total_db_f,
            total_dW_i, total_db_i,
            total_dW_c, total_db_c,
            total_dW_o, total_db_o,
        )
        
        return dX
    
    def get_gradients(self):
        """Get all gradients from LSTM cell"""
        return self.lstm_cell.get_gradients()
    
    def update_parameters(self, optimizer):
        """
        Update LSTM cell parameters using optimizer.
        
        Args:
            optimizer: Optimizer instance (SGD, Adam, etc.)
        """
        if self.gate_optimizers is None:
            self.set_optimizer(optimizer)

        grads = self.lstm_cell.get_gradients()
        gates = [('f', 'W_f', 'b_f'), ('i', 'W_i', 'b_i'), ('c', 'W_c', 'b_c'), ('o', 'W_o', 'b_o')]

        for gate_idx, (gate, w_attr, b_attr) in enumerate(gates):
            dW = grads[f'dW_{gate}']
            db = grads[f'db_{gate}']
            gate_optimizer = self.gate_optimizers[gate_idx]

            dW_update, db_update = gate_optimizer.update(dW, db)

            W = getattr(self.lstm_cell, w_attr)
            b = getattr(self.lstm_cell, b_attr)

            setattr(self.lstm_cell, w_attr, W - dW_update)
            setattr(self.lstm_cell, b_attr, b - db_update)
