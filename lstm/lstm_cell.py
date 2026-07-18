"""
LSTM (Long Short-Term Memory) Cell implementation.

Key concepts:
- LSTM has 3 gates: input gate, forget gate, output gate
- Cell state (C) carries long-term memory through the sequence
- Hidden state (H) is the short-term output
- All computations are element-wise multiplied (Hadamard product ⊙)

Forward pass:
    f_t = σ(W_f @ [h_(t-1), x_t] + b_f)                    # Forget gate
    i_t = σ(W_i @ [h_(t-1), x_t] + b_i)                    # Input gate  
    C̃_t = tanh(W_c @ [h_(t-1), x_t] + b_c)                # Candidate cell state
    C_t = f_t ⊙ C_(t-1) + i_t ⊙ C̃_t                        # Cell state update
    o_t = σ(W_o @ [h_(t-1), x_t] + b_o)                    # Output gate
    h_t = o_t ⊙ tanh(C_t)                                  # Hidden state output

Notes:
- Weight initialization can be configured per-cell (Xavier or He) via the
    `init` argument to `LSTMCell`.
- Activation derivative helpers in `lstm.activations` are expected to take
    pre-activation values (the "z" inputs) when computing derivatives.
"""

import numpy as np
from .activations import sigmoid, sigmoid_derivative, tanh, tanh_derivative


class LSTMCell:
    """
    LSTM Cell: processes one timestep of input.
    
    Args:
        input_size: Size of input at each timestep
        hidden_size: Size of hidden state (LSTM cell size)
    """
    
    def __init__(self, input_size, hidden_size, init='xavier'):
        """
        Args:
            input_size: Size of input at each timestep
            hidden_size: Size of hidden state (LSTM cell size)
            init: Weight initialization method. One of 'xavier' (default) or 'he'.
                  Use 'he' when using ReLU activations in surrounding layers.
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # All gates share concatenated input: [h_(t-1), x_t]
        concat_size = hidden_size + input_size
        
        # Initialize weights and biases for all 4 operations (forget, input, cell, output)
        # Initialization strategy can be chosen via `init`:
        # - 'xavier' (default): good general-purpose init for tanh/sigmoid
        # - 'he': recommended for ReLU-based networks
        if init not in ('xavier', 'he'):
            raise ValueError("init must be one of ('xavier', 'he')")

        if init == 'xavier':
            limit = np.sqrt(6.0 / (concat_size + hidden_size))
            W_init = lambda shape: np.random.uniform(-limit, limit, shape)
        else:  # he initialization (He normal)
            std = np.sqrt(2.0 / concat_size)
            W_init = lambda shape: np.random.randn(*shape) * std

        # Forget gate
        self.W_f = W_init((concat_size, hidden_size))
        self.b_f = np.zeros((1, hidden_size))
        
        # Input gate
        self.W_i = W_init((concat_size, hidden_size))
        self.b_i = np.zeros((1, hidden_size))
        
        # Cell candidate (tanh)
        self.W_c = W_init((concat_size, hidden_size))
        self.b_c = np.zeros((1, hidden_size))
        
        # Output gate
        self.W_o = W_init((concat_size, hidden_size))
        self.b_o = np.zeros((1, hidden_size))
        
        # Gradients
        self.dW_f = self.dW_i = self.dW_c = self.dW_o = None
        self.db_f = self.db_i = self.db_c = self.db_o = None
        
        # Cache for backward pass
        self.cache = None
    
    def forward(self, x_t, h_prev, C_prev):
        """
        Forward pass for one timestep.
        
        Args:
            x_t: Input at timestep t, shape (batch_size, input_size)
            h_prev: Hidden state from previous timestep, shape (batch_size, hidden_size)
            C_prev: Cell state from previous timestep, shape (batch_size, hidden_size)
        
        Returns:
            h_t: Hidden state at timestep t, shape (batch_size, hidden_size)
            C_t: Cell state at timestep t, shape (batch_size, hidden_size)

        Notes:
            This method caches pre-activation values (z_f, z_i, z_c, z_o)
            which are used during the backward pass. Activation derivative
            helpers in `lstm.activations` are expected to accept these
            pre-activation (z) values when computing derivatives.
        """
        # Accept 1D inputs (single sample) and treat them as batch_size=1
        squeezed = False
        if x_t.ndim == 1:
            x_t = np.expand_dims(x_t, 0)
            squeezed = True
        if h_prev.ndim == 1:
            h_prev = np.expand_dims(h_prev, 0)
            squeezed = True
        if C_prev.ndim == 1:
            C_prev = np.expand_dims(C_prev, 0)
            squeezed = True

        batch_size = x_t.shape[0]
        
        # Concatenate previous hidden state with current input
        # concat = [h_{t-1}, x_t]
        concat = np.hstack([h_prev, x_t])  # (batch_size, hidden_size + input_size)
        
        # Compute gates using sigmoid activation (outputs in range [0,1])
        # z_f = concat @ W_f + b_f
        z_f = np.dot(concat, self.W_f) + self.b_f
        # f_t = σ(z_f)
        f_t = sigmoid(z_f)  # Forget gate

        # z_i = concat @ W_i + b_i
        z_i = np.dot(concat, self.W_i) + self.b_i
        # i_t = σ(z_i)
        i_t = sigmoid(z_i)  # Input gate

        # z_c = concat @ W_c + b_c
        z_c = np.dot(concat, self.W_c) + self.b_c
        # C̃_t = tanh(z_c)
        C_tilde = tanh(z_c)  # Candidate cell state (tanh outputs in range [-1,1])

        # Cell state update: C_t = f_t ⊙ C_{t-1} + i_t ⊙ C̃_t
        C_t = f_t * C_prev + i_t * C_tilde

        # z_o = concat @ W_o + b_o
        z_o = np.dot(concat, self.W_o) + self.b_o
        # o_t = σ(z_o)
        o_t = sigmoid(z_o)  # Output gate

        # h_t = o_t ⊙ tanh(C_t)
        h_t = o_t * tanh(C_t)  # Final hidden state
        
        # Cache for backward pass (include whether inputs were originally 1D)
        self.cache = (x_t, h_prev, C_prev, concat, f_t, i_t, C_tilde, C_t, o_t,
                      z_f, z_i, z_c, z_o, squeezed)

        # If inputs were 1D, return squeezed outputs for convenience
        if squeezed:
            return np.squeeze(h_t, axis=0), np.squeeze(C_t, axis=0)

        return h_t, C_t
    
    def backward(self, dh_t, dC_t):
        """
        Backward pass for one timestep (BPTT - Backprop Through Time).
        
        Args:
            dh_t: Gradient of loss w.r.t. hidden state at t, shape (batch_size, hidden_size)
            dC_t: Gradient of loss w.r.t. cell state at t, shape (batch_size, hidden_size)
        
        Returns:
            dx_t: Gradient w.r.t. input x_t, shape (batch_size, input_size)
            dh_prev: Gradient w.r.t. previous hidden state, shape (batch_size, hidden_size)
            dC_prev: Gradient w.r.t. previous cell state, shape (batch_size, hidden_size)
        
        Notes:
            This method expects that activation derivative functions (for
            sigmoid/tanh) accept pre-activation (z) values; the forward
            pass caches these values and they are used here.
        """
        if self.cache is None:
            raise ValueError("Must call forward before backward")
        
        (x_t, h_prev, C_prev, concat, f_t, i_t, C_tilde, C_t, o_t,
         z_f, z_i, z_c, z_o, squeezed) = self.cache

        # Accept 1D gradients for single-sample cases
        if dh_t.ndim == 1:
            dh_t = np.expand_dims(dh_t, 0)
        if dC_t.ndim == 1:
            dC_t = np.expand_dims(dC_t, 0)
        
        batch_size = x_t.shape[0]
        
        # Gradient through output gate and tanh(C_t)
        # tanh_C_t = tanh(C_t)
        tanh_C_t = tanh(C_t)

        # dL/do = dh_t ⊙ tanh(C_t)
        dL_do = dh_t * tanh_C_t  # Gradient for output gate
        # dL/d(tanh(C_t)) = dh_t ⊙ o_t
        dL_dtanh_C = dh_t * o_t  # Gradient through tanh
        # dL/dC_from_h = dL/d(tanh(C_t)) ⊙ tanh'(C_t)
        dL_dC_raw = dL_dtanh_C * tanh_derivative(C_t)  # Chain rule: tanh derivative

        # Accumulate gradients into dC_t (including gradient passed in dC_t)
        dC_t = dC_t + dL_dC_raw  # Add gradient from cell state

        # Gradient for input gate and candidate cell state
        # dL/di = dC_t ⊙ C̃_t
        dL_di = dC_t * C_tilde
        # dL/dC̃ = dC_t ⊙ i_t
        dL_dC_tilde = dC_t * i_t
        # dL/dz_c = dL/dC̃ ⊙ tanh'(z_c)
        dL_dz_c = dL_dC_tilde * tanh_derivative(z_c)

        # Gradient for forget gate
        # dL/df = dC_t ⊙ C_{t-1}
        dL_df = dC_t * C_prev
        # dL/dz_f = dL/df ⊙ σ'(z_f)
        dL_dz_f = dL_df * sigmoid_derivative(z_f)

        # Gradient for output gate
        # dL/dz_o = dL/do ⊙ σ'(z_o)
        dL_dz_o = dL_do * sigmoid_derivative(z_o)

        # Gradient for input gate
        # dL/dz_i = dL/di ⊙ σ'(z_i)
        dL_dz_i = dL_di * sigmoid_derivative(z_i)
        
        # Compute weight gradients
        self.dW_f = np.dot(concat.T, dL_dz_f) / batch_size
        self.db_f = np.sum(dL_dz_f, axis=0, keepdims=True) / batch_size
        
        self.dW_i = np.dot(concat.T, dL_dz_i) / batch_size
        self.db_i = np.sum(dL_dz_i, axis=0, keepdims=True) / batch_size
        
        self.dW_c = np.dot(concat.T, dL_dz_c) / batch_size
        self.db_c = np.sum(dL_dz_c, axis=0, keepdims=True) / batch_size
        
        self.dW_o = np.dot(concat.T, dL_dz_o) / batch_size
        self.db_o = np.sum(dL_dz_o, axis=0, keepdims=True) / batch_size
        
        # Gradient through concatenation
        d_concat = (np.dot(dL_dz_f, self.W_f.T) + 
                   np.dot(dL_dz_i, self.W_i.T) + 
                   np.dot(dL_dz_c, self.W_c.T) + 
                   np.dot(dL_dz_o, self.W_o.T))
        
        # Split gradient back to hidden state and input
        dh_prev = d_concat[:, :self.hidden_size]
        dx_t = d_concat[:, self.hidden_size:]
        
        # Gradient for previous cell state (through forget gate)
        dC_prev = dC_t * f_t
        
        # Squeeze outputs if forward received 1D inputs
        if squeezed:
            return np.squeeze(dx_t, axis=0), np.squeeze(dh_prev, axis=0), np.squeeze(dC_prev, axis=0)

        return dx_t, dh_prev, dC_prev
    
    def get_gradients(self):
        """Return all computed gradients"""
        return {
            'dW_f': self.dW_f, 'db_f': self.db_f,
            'dW_i': self.dW_i, 'db_i': self.db_i,
            'dW_c': self.dW_c, 'db_c': self.db_c,
            'dW_o': self.dW_o, 'db_o': self.db_o,
        }
    
    def set_gradients(self, dW_f, db_f, dW_i, db_i, dW_c, db_c, dW_o, db_o):
        """Set all gradients (for accumulation across batch)"""
        self.dW_f = dW_f
        self.db_f = db_f
        self.dW_i = dW_i
        self.db_i = db_i
        self.dW_c = dW_c
        self.db_c = db_c
        self.dW_o = dW_o
        self.db_o = db_o
    
    def update_parameters(self, dW_f_update, db_f_update, dW_i_update, db_i_update,
                         dW_c_update, db_c_update, dW_o_update, db_o_update):
        """Update all parameters using optimizer updates"""
        self.W_f -= dW_f_update
        self.b_f -= db_f_update
        self.W_i -= dW_i_update
        self.b_i -= db_i_update
        self.W_c -= dW_c_update
        self.b_c -= db_c_update
        self.W_o -= dW_o_update
        self.b_o -= db_o_update
