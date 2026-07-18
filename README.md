# LSTM from Scratch

A from-scratch implementation of LSTM (Long Short-Term Memory) Recurrent Neural Networks for time-series forecasting and sequence modeling.

## Features

- **LSTM Cell**: Full implementation with all 4 gates (forget, input, cell, output) and BPTT
- **LSTM Layer**: Processes entire sequences with proper gradient accumulation
- **Dense Output Layer**: Regression/Classification head
- **Optimizers**: SGD with momentum, Adam
- **Loss Functions**: Cross-Entropy (classification), MSE (regression)
- **Activation Functions**: Sigmoid, Tanh, ReLU, Softmax
- **Training Pipeline**: Includes model serialization, metrics tracking, and evaluation

## Project Structure

```
LSTM/
├── lstm/
│   ├── __init__.py           # Package init
│   ├── lstm_cell.py          # LSTM cell implementation
│   ├── lstm_layer.py         # LSTM layer (processes sequences)
│   ├── dense_layer.py        # Fully connected output layer
│   ├── network.py            # Network composition
│   ├── activations.py        # Activation functions
│   ├── losses.py             # Loss functions (CrossEntropy)
│   ├── optimizers.py         # Optimizers (SGD, Adam)
│   └── data.py               # synthetic sequence data generation
├── train.py                  # Training script
├── evaluate.py               # Evaluation script
├── tests/
│   └── test_basics.py        # Unit tests
├── notebooks/
│   └── analysis.ipynb        # Analysis notebook
├── data/                     # Dataset directory
└── requirements.txt          # Dependencies
```

## Getting Started

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Train Model

Train the LSTM on AirPassengers monthly time-series data:

```bash
python train.py
```

This will:
- Load `data/AirPassengers.csv` (1949-1960 monthly passenger counts)
- Normalize using training set statistics
- Split into train (70%), validation (15%), and test (15%) sets
- Hyperparameter search over hidden sizes and learning rates
- Train the model for 50 epochs
- Save trained weights and scaler to `lstm_model.pkl`
- Save metrics (train/val/test losses) to `data/`

### 3. Run Analysis Notebook

After training, run the Jupyter notebook for visualization:

```bash
jupyter notebook notebooks/analysis.ipynb
```

The notebook will:
- Load pre-trained model weights from `lstm_model.pkl`
- Visualize training history
- Evaluate on test set with actual vs. predicted plots
- Generate a 24-month recursive forecast

### 4. Run Tests

```bash
python tests/test_basics.py
```

Tests basic forward/backward passes for:
- LSTM Cell (gate computations, gradient flow)
- LSTM Layer (sequence processing)
- Dense Layer (linear transformations)

## Architecture

```
Input Sequence (batch_size, seq_len, input_size)
          ↓
    LSTM Layer (hidden_size=64)
          ↓
    Dense Layer (output_size=1, no activation)
          ↓
    Regression Output (continuous values)
```

### How LSTM Works

The LSTM cell maintains two states across timesteps:
- **Hidden State (h_t)**: Short-term memory, output at each timestep
- **Cell State (C_t)**: Long-term memory, robust to vanishing gradients

Four gates control information flow:
1. **Forget Gate (f_t)**: σ(W_f @ [h_{t-1}, x_t] + b_f) — Controls cell state decay
2. **Input Gate (i_t)**: σ(W_i @ [h_{t-1}, x_t] + b_i) — Controls new information gating
3. **Cell Candidate (C̃_t)**: tanh(W_c @ [h_{t-1}, x_t] + b_c) — Candidate values to add
4. **Output Gate (o_t)**: σ(W_o @ [h_{t-1}, x_t] + b_o) — Controls hidden state output

Cell state update: `C_t = f_t ⊙ C_{t-1} + i_t ⊙ C̃_t`
Hidden state update: `h_t = o_t ⊙ tanh(C_t)`

## Model Details

### LSTM Configuration (AirPassengers)
- Input size: 1 (univariate time series)
- Sequence length: 12 (lagged 12 months)
- Hidden size: 64 (optimized via grid search)
- Output size: 1 (regression: next month prediction)

### Training Configuration
- Optimizer: Adam (learning rate: 0.001–0.005)
- Loss: Mean Squared Error (MSE)
- Epochs: 50
- Batch size: 16
- Normalization: Training set mean/std applied to all splits

## Implementation Notes

### Gradient Computation
- Full **BPTT** (Backpropagation Through Time) with gradient accumulation across timesteps
- Proper chain rule application through all 4 gates
- Numerical stability: sigmoid clipping to [-500, 500], gradient normalization by batch size

### Normalization
- Data normalized using training set mean and std
- Same scaler applied to validation and test sets
- Predictions denormalized for interpretability

### Serialization
- Model weights saved as pickle files containing layer parameters and metadata
- Scaler statistics saved for consistent inference
- Loss histories saved as NumPy arrays for analysis

### Known Limitations
- No gradient clipping implemented (may experience vanishing/exploding gradients on very long sequences)
- No dropout or regularization layers
- Educational implementation; production systems should use PyTorch, TensorFlow, or similar

## References

- Hochreiter, S., & Schmidhuber, J. (1997). "Long Short-Term Memory". Neural Computation, 9(8).
- Graves, A. (2012). "Supervised Sequence Labelling with Recurrent Neural Networks".
- Kingma, D. P., & Ba, J. (2014). "Adam: A Method for Stochastic Optimization". arXiv:1412.6980.
- Goodfellow, I., Bengio, Y., & Courville, A. (2016). "Deep Learning". MIT Press.
