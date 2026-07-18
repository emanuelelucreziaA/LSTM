"""
LSTM Network: Composition of LSTM layers and Dense output layer.

Architecture:
  Input Sequence -> LSTM Layer -> Dense Output -> Predictions

Supports both regression and classification tasks.
"""

import copy
import numpy as np


class LSTMNetwork:
    """
    LSTM Network: stack of LSTM layers followed by dense output layer.
    
    Supports regression (continuous outputs) and classification (discrete classes):
    - LSTM layers process the entire sequence
    - Dense layer maps final LSTM hidden state to output values or class logits
    """
    
    def __init__(self):
        self.layers = []  # LSTM and Dense layers
        self.optimizer = None
        self.loss_fn = None
    
    def add_lstm_layer(self, lstm_layer):
        """Add an LSTM layer to the network"""
        self.layers.append(lstm_layer)
        return self
    
    def add_dense_layer(self, dense_layer):
        """Add a dense output layer to the network"""
        self.layers.append(dense_layer)
        return self
    
    def set_optimizer(self, optimizer):
        """Set optimizer for weight updates"""
        self.optimizer = optimizer
        return self
    
    def set_loss(self, loss_fn):
        """Set loss function"""
        self.loss_fn = loss_fn
        return self
    
    def forward(self, X):
        """
        Forward pass: propagate input through all layers.
        
        Args:
            X: Input sequence of shape (batch_size, seq_len, input_size)
        
        Returns:
            output: Network output (logits) of shape (batch_size, num_classes)
        """
        layer_input = X
        last_h_final = None

        # Pass full sequence outputs between stacked LSTM layers
        for layer in self.layers[:-1]:  # LSTM layers
            if hasattr(layer, 'forward'):
                # LSTM layer returns (H, h_final, C_final)
                H, h_final, C_final = layer.forward(layer_input)
                # Forward the full hidden-state sequence to the next LSTM layer
                layer_input = H
                last_h_final = h_final

        # Final dense layer: takes final hidden state as input
        dense_input = last_h_final if last_h_final is not None else layer_input
        output = self.layers[-1].forward(dense_input)

        return output
    
    def backward(self, dL_doutput):
        """
        Backward pass: propagate gradient through all layers (reverse order).
        
        Args:
            dL_doutput: Gradient of loss w.r.t. network output
        
        Returns:
            None (gradients stored in each layer)
        """
        # Backprop through final dense output layer first
        dL_dh = self.layers[-1].backward(dL_doutput)

        # Backprop through any preceding LSTM layers
        for layer in reversed(self.layers[:-1]):
            if hasattr(layer, 'cache') and layer.cache is not None:
                _, H, _ = layer.cache
                dL_dH = np.zeros_like(H)
            else:
                dL_dH = None

            dL_dh = layer.backward(dL_dH, dL_dh_final=dL_dh)

        return dL_dh
    
    def update_weights(self):
        """
        Update weights and biases using optimizer.
        
        Each layer keeps its own optimizer state so adaptive methods like Adam
        can maintain separate moment estimates per parameter tensor.
        """
        if self.optimizer is None:
            raise ValueError("Optimizer not set. Use model.set_optimizer()")

        for layer in self.layers:
            if not hasattr(layer, 'optimizer') or layer.optimizer is None:
                layer.optimizer = copy.deepcopy(self.optimizer)

            # LSTM layer parameter update
            if hasattr(layer, 'lstm_cell'):
                layer.update_parameters(layer.optimizer)
            # Dense layer parameter update
            elif hasattr(layer, 'get_gradients'):
                dW, db = layer.get_gradients()
                if dW is not None and db is not None:
                    update_W, update_b = layer.optimizer.update(dW, db)
                    layer.update_parameters(update_W, update_b)
    
    def predict(self, X):
        """
        Predict: forward pass returning raw outputs.
        
        For regression: returns continuous predictions.
        For classification: returns logits (use predict_proba for probabilities).
        
        Args:
            X: Input sequence of shape (batch_size, seq_len, input_size)
        
        Returns:
            predictions: Raw output of shape (batch_size, output_size), always 2D
        """
        output = self.forward(X)
        # Ensure output is always 2D: (batch_size, output_size)
        if output.ndim == 1:
            output = output.reshape(-1, 1)
        return output
    
    def predict_proba(self, X):
        """
        Get probability predictions for classification.
        
        For classification tasks only. Applies softmax to logits.
        
        Args:
            X: Input sequence
        
        Returns:
            probabilities: Class probabilities of shape (batch_size, num_classes)
        """
        from .activations import softmax
        
        logits = self.forward(X)
        return softmax(logits)
    
    def predict_class(self, X):
        """
        Get class predictions for classification.
        
        For classification tasks only. Returns argmax of softmax.
        
        Args:
            X: Input sequence
        
        Returns:
            class_indices: Predicted class indices of shape (batch_size,)
        """
        probabilities = self.predict_proba(X)
        return np.argmax(probabilities, axis=1)

    def get_weights(self):
        """Serialize model weights for saving."""
        weights = []
        for layer in self.layers:
            if hasattr(layer, 'lstm_cell'):
                cell = layer.lstm_cell
                weights.append({
                    'type': 'lstm',
                    'W_f': cell.W_f.copy(),
                    'b_f': cell.b_f.copy(),
                    'W_i': cell.W_i.copy(),
                    'b_i': cell.b_i.copy(),
                    'W_c': cell.W_c.copy(),
                    'b_c': cell.b_c.copy(),
                    'W_o': cell.W_o.copy(),
                    'b_o': cell.b_o.copy(),
                })
            elif hasattr(layer, 'W') and hasattr(layer, 'b'):
                weights.append({
                    'type': 'dense',
                    'W': layer.W.copy(),
                    'b': layer.b.copy(),
                })
        return weights

    def set_weights(self, weights):
        """Load serialized model weights into the network."""
        if len(weights) != len(self.layers):
            raise ValueError('Weight count does not match model layers')

        for layer, weight_dict in zip(self.layers, weights):
            if weight_dict['type'] == 'lstm':
                cell = layer.lstm_cell
                cell.W_f = weight_dict['W_f'].copy()
                cell.b_f = weight_dict['b_f'].copy()
                cell.W_i = weight_dict['W_i'].copy()
                cell.b_i = weight_dict['b_i'].copy()
                cell.W_c = weight_dict['W_c'].copy()
                cell.b_c = weight_dict['b_c'].copy()
                cell.W_o = weight_dict['W_o'].copy()
                cell.b_o = weight_dict['b_o'].copy()
            elif weight_dict['type'] == 'dense':
                layer.W = weight_dict['W'].copy()
                layer.b = weight_dict['b'].copy()
            else:
                raise ValueError(f"Unknown layer type in weights: {weight_dict['type']}")
