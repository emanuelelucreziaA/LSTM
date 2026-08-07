"""
Training script: Train LSTM for AirPassengers regression.

Usage:
    python train.py

Architecture:
    Sequence Input (seq_len, 1) -> LSTM (hidden_size=64) -> Dense (1)
"""

import os
from dotenv import load_dotenv
load_dotenv()
import sys
PROJECT_ROOT = os.getenv('PROJECT_ROOT', os.path.abspath(os.path.dirname(__file__)))
sys.path.insert(0, PROJECT_ROOT)
import numpy as np

# Data directory (can be set in .env)
DATA_DIR = os.getenv('DATA_DIR', os.path.join(PROJECT_ROOT, 'data'))
os.makedirs(DATA_DIR, exist_ok=True)

from lstm.lstm_layer import LSTMLayer
from lstm.dense_layer import DenseLayer
from lstm.network import LSTMNetwork
from lstm.optimizers import Adam
from lstm.losses import MSELoss
from lstm.time_series import (
    prepare_air_passengers,
    normalize_time_series_data,
)

_loss_fn = MSELoss()
import pickle


def build_model(input_size=1, output_size=1, hidden_size=64):
    """Build an LSTM network for AirPassengers regression."""
    print("\n" + "="*70)
    print("Building LSTM Model for regression")
    print(f"Architecture: Sequence(seq_len, {input_size}) -> LSTM(hidden={hidden_size}) -> Dense({output_size})")
    print("="*70)

    model = LSTMNetwork()
    model.add_lstm_layer(LSTMLayer(
        input_size=input_size,
        hidden_size=hidden_size
    ))

    model.add_dense_layer(DenseLayer(
        input_size=hidden_size,
        output_size=output_size,
        activation_fn=None,
        activation_derivative=None
    ))

    model.set_optimizer(Adam(learning_rate=0.001))

    print(f"✓ Model built with {sum(1 for layer in model.layers)} layer(s)")
    return model


def train_epoch(model, X_train, y_train, batch_size=32):
    """Train for one epoch on AirPassengers regression."""
    num_samples = X_train.shape[0]
    num_batches = int(np.ceil(num_samples / batch_size))
    total_loss = 0.0

    indices = np.random.permutation(num_samples)
    X_shuffled = X_train[indices]
    y_shuffled = y_train[indices]

    for batch_idx in range(num_batches):
        start = batch_idx * batch_size
        end = min(start + batch_size, num_samples)
        X_batch = X_shuffled[start:end]
        y_batch = y_shuffled[start:end]

        logits = model.forward(X_batch)

        y_batch = y_batch.reshape(logits.shape)
        loss = _loss_fn(y_batch, logits)
        dL_doutput = _loss_fn.gradient(y_batch, logits)

        # Backpropagate and update weights
        model.backward(dL_doutput)
        model.update_weights()

        total_loss += loss

        if (batch_idx + 1) % max(1, num_batches // 10) == 0:
            print(f"  Batch {batch_idx+1}/{num_batches}, Loss: {loss:.4f}")

    return total_loss / max(1, num_batches)


def evaluate(model, X_test, y_test, batch_size=32):
    """Evaluate model on regression data using MSE."""
    num_samples = X_test.shape[0]
    total_loss = 0.0
    num_batches = (num_samples + batch_size - 1) // batch_size

    for batch_idx in range(num_batches):
        start = batch_idx * batch_size
        end = min(start + batch_size, num_samples)
        X_batch = X_test[start:end]
        y_batch = y_test[start:end]

        logits = model.forward(X_batch)

        loss = _loss_fn(y_batch, logits)
        total_loss += loss

    avg_loss = total_loss / num_batches
    return avg_loss


def main():
    """Main training pipeline"""
    print("\n" + "="*70)
    print("LSTM Training Pipeline - AirPassengers regression")
    print("="*70)

    print("\nLoading AirPassengers time-series data...")
    X_train, y_train, X_val, y_val, X_test, y_test = prepare_air_passengers(
        DATA_DIR,
        seq_len=12,
        horizon=1,
        train_ratio=0.7,
        val_ratio=0.15,
    )
    X_train, y_train, X_val, y_val, X_test, y_test, scaler = normalize_time_series_data(
        X_train, y_train, X_val, y_val, X_test, y_test
    )
    output_size = 1
    hidden_size = 64

    print(f"Training data: {X_train.shape} (batch_size, seq_len, input_size)")
    print(f"Training labels: {y_train.shape}")
    print(f"Validation data: {X_val.shape}")
    print(f"Validation labels: {y_val.shape}")
    print(f"Normalized using mean={scaler['mean']:.4f}, std={scaler['std']:.4f}")
    print(f"Test data: {X_test.shape}")
    print(f"Test labels: {y_test.shape}")

    model = build_model(
        input_size=X_train.shape[2],
        output_size=output_size,
        hidden_size=hidden_size,
    )

    num_epochs = 50
    batch_size = 16

    print(f"\nTraining with:")
    print(f"  Epochs: {num_epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  Optimizer: Adam (lr=0.001)")
    print("  Loss: MSE")

    train_losses = []
    val_losses = []
    test_losses = []

    for epoch in range(num_epochs):
        print(f"\n{'='*70}")
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"{'='*70}")

        train_loss = train_epoch(model, X_train, y_train, batch_size)
        train_losses.append(train_loss)
        print(f"Training Loss: {train_loss:.4f}")
        val_loss = evaluate(model, X_val, y_val, batch_size=batch_size)
        val_losses.append(val_loss)
        print(f"Validation MSE: {val_loss:.4f}")
        test_loss = evaluate(model, X_test, y_test, batch_size)
        test_losses.append(test_loss)
        print(f"Test MSE: {test_loss:.4f}")

    model_save_path = os.path.join(PROJECT_ROOT, 'lstm_model.pkl')
    print(f"\nSaving model info to {model_save_path}...")

    try:
        model_data = {
            'hidden_size': hidden_size,
            'output_size': output_size,
            'train_losses': train_losses,
            'val_losses': val_losses,
            'test_losses': test_losses,
            'weights': model.get_weights(),
            'scaler': scaler,
        }
        with open(model_save_path, 'wb') as f:
            pickle.dump(model_data, f)
        print("✓ Model metadata and weights saved successfully")
    except Exception as e:
        print(f"✗ Failed to save model metadata: {e}")

    np.save(os.path.join(DATA_DIR, 'train_losses.npy'), np.array(train_losses))
    np.save(os.path.join(DATA_DIR, 'val_losses.npy'), np.array(val_losses))
    np.save(os.path.join(DATA_DIR, 'test_losses.npy'), np.array(test_losses))
    print("✓ Metrics saved to data/")

    print("\n" + "="*70)
    print("Training Complete!")
    print("="*70)


if __name__ == '__main__':
    main()
