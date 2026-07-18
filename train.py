"""
Training script: Train LSTM on synthetic sequence classification.

Usage:
    python train.py

Architecture:
    Sequence Input (seq_len, 1) -> LSTM (hidden_size=128) -> Dense (10) -> Softmax
"""

import csv
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
from lstm.activations import softmax
from lstm.losses import CrossEntropy
from lstm.optimizers import Adam
from lstm.data import SequenceDataLoader, one_hot_encode
import pickle


def build_model(input_size=1, output_size=10, hidden_size=128, regression=False):
    """Build an LSTM network for classification or regression."""
    mode = 'regression' if regression else 'classification'
    print("\n" + "="*70)
    print(f"Building LSTM Model for {mode}")
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
    model.set_loss(CrossEntropy() if not regression else None)

    print(f"✓ Model built with {sum(1 for layer in model.layers)} layer(s)")
    return model


def mse_loss(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.float32).reshape(y_pred.shape)
    y_pred = np.asarray(y_pred, dtype=np.float32)
    return np.mean((y_pred - y_true) ** 2)


def prepare_air_passengers(
    data_dir,
    seq_len=12,
    horizon=1,
    train_ratio=0.7,
    val_ratio=0.15,
):
    csv_path = os.path.join(data_dir, 'AirPassengers.csv')
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"AirPassengers.csv not found in {data_dir}")

    dates = []
    values = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            dates.append(row[0])
            values.append(float(row[1]))

    values = np.array(values, dtype=np.float32)
    num_samples = len(values) - seq_len - horizon + 1
    X = np.zeros((num_samples, seq_len, 1), dtype=np.float32)
    y = np.zeros((num_samples,), dtype=np.float32)
    for i in range(num_samples):
        X[i, :, 0] = values[i : i + seq_len]
        y[i] = values[i + seq_len + horizon - 1]

    split1 = int(len(X) * train_ratio)
    split2 = split1 + int(len(X) * val_ratio)
    return X[:split1], y[:split1], X[split1:split2], y[split1:split2], X[split2:], y[split2:]


def normalize_time_series_data(X_train, y_train, X_val, y_val, X_test, y_test):
    """Normalize dataset using training split statistics."""
    values = np.concatenate([X_train.ravel(), y_train.ravel()])
    mean = float(np.mean(values))
    std = float(np.std(values))
    if std == 0.0:
        std = 1.0

    X_train = (X_train - mean) / std
    X_val = (X_val - mean) / std
    X_test = (X_test - mean) / std

    y_train = (y_train - mean) / std
    y_val = (y_val - mean) / std
    y_test = (y_test - mean) / std

    scaler = {'mean': mean, 'std': std}
    return X_train, y_train, X_val, y_val, X_test, y_test, scaler


def inverse_scale(y, scaler):
    return y * scaler['std'] + scaler['mean']


def train_epoch(model, X_train, y_train, batch_size=32, regression=False):
    """Train for one epoch (toy sanity check)."""
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

        if regression:
            y_batch = y_batch.reshape(logits.shape)
            loss = mse_loss(y_batch, logits)
            dL_doutput = 2.0 * (logits - y_batch) / logits.shape[0]
        else:
            probs = softmax(logits)
            loss = model.loss_fn(y_batch, probs)
            dL_doutput = probs - y_batch

        # Backpropagate and update weights
        model.backward(dL_doutput)
        model.update_weights()

        total_loss += loss

        if (batch_idx + 1) % max(1, num_batches // 10) == 0:
            print(f"  Batch {batch_idx+1}/{num_batches}, Loss: {loss:.4f}")

    return total_loss / max(1, num_batches)


def evaluate(model, X_test, y_test, batch_size=32, regression=False):
    """Evaluate model on test or regression data."""
    num_samples = X_test.shape[0]
    total_loss = 0.0
    num_batches = (num_samples + batch_size - 1) // batch_size
    correct = 0

    for batch_idx in range(num_batches):
        start = batch_idx * batch_size
        end = min(start + batch_size, num_samples)
        X_batch = X_test[start:end]
        y_batch = y_test[start:end]

        logits = model.forward(X_batch)

        if regression:
            loss = mse_loss(y_batch, logits)
            total_loss += loss
        else:
            probs = softmax(logits)
            loss = model.loss_fn(y_batch, probs)
            total_loss += loss
            predictions = np.argmax(probs, axis=1)
            correct += np.sum(predictions == np.argmax(y_batch, axis=1))

    avg_loss = total_loss / num_batches
    accuracy = correct / num_samples if not regression else None
    return accuracy, avg_loss


def main():
    """Main training pipeline"""
    dataset_mode = os.getenv('DATASET', 'air-passengers').lower()
    print("\n" + "="*70)
    print(f"LSTM Training Pipeline - dataset mode: {dataset_mode}")
    print("="*70)

    if dataset_mode == 'air-passengers':
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
        regression = True
        output_size = 1
        hidden_size = 64
    else:
        print("\nLoading synthetic toy classification data...")
        loader = SequenceDataLoader(
            seq_len=20,
            input_size=1,
            num_classes=10,
            train_samples=1000,
            test_samples=200,
            noise_level=0.05,
            seed=42,
        )
        X_train, y_train, X_test, y_test = loader.load_data()
        X_val, y_val = X_test, y_test
        regression = False
        output_size = 10
        hidden_size = 64

    print(f"Training data: {X_train.shape} (batch_size, seq_len, input_size)")
    print(f"Training labels: {y_train.shape}")
    if dataset_mode == 'air-passengers':
        print(f"Validation data: {X_val.shape}")
        print(f"Validation labels: {y_val.shape}")
        print(f"Normalized using mean={scaler['mean']:.4f}, std={scaler['std']:.4f}")
    print(f"Test data: {X_test.shape}")
    print(f"Test labels: {y_test.shape}")

    if not regression:
        y_train = one_hot_encode(y_train, num_classes=output_size)
        y_test = one_hot_encode(y_test, num_classes=output_size)

    model = build_model(
        input_size=X_train.shape[2],
        output_size=output_size,
        hidden_size=hidden_size,
        regression=regression,
    )

    num_epochs = 50
    batch_size = 16

    print(f"\nTraining with:")
    print(f"  Epochs: {num_epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  Optimizer: Adam (lr=0.001)")
    print(f"  Loss: {'MSE' if regression else 'CrossEntropy'}")

    train_losses = []
    val_losses = []
    test_losses = []
    test_metrics = []

    for epoch in range(num_epochs):
        print(f"\n{'='*70}")
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"{'='*70}")

        train_loss = train_epoch(model, X_train, y_train, batch_size, regression=regression)
        train_losses.append(train_loss)
        print(f"Training Loss: {train_loss:.4f}")
        if dataset_mode == 'air-passengers':
            _, val_loss = evaluate(model, X_val, y_val, batch_size=batch_size, regression=regression)
            val_losses.append(val_loss)
            print(f"Validation MSE: {val_loss:.4f}")
        test_acc, test_loss = evaluate(model, X_test, y_test, batch_size, regression=regression)
        test_losses.append(test_loss)
        if regression:
            print(f"Test MSE: {test_loss:.4f}")
        else:
            test_metrics.append(test_acc)
            print(f"Test Loss: {test_loss:.4f}")
            print(f"Test Accuracy: {test_acc:.4f} ({int(test_acc * 100)}%)")

    model_save_path = os.path.join(PROJECT_ROOT, 'lstm_model.pkl')
    print(f"\nSaving model info to {model_save_path}...")

    try:
        model_data = {
            'dataset_mode': dataset_mode,
            'hidden_size': hidden_size,
            'output_size': output_size,
            'train_losses': train_losses,
            'val_losses': val_losses,
            'test_losses': test_losses,
            'test_metrics': test_metrics,
            'weights': model.get_weights(),
            'scaler': scaler if dataset_mode == 'air-passengers' else None,
        }
        with open(model_save_path, 'wb') as f:
            pickle.dump(model_data, f)
        print("✓ Model metadata and weights saved successfully")
    except Exception as e:
        print(f"✗ Failed to save model metadata: {e}")

    np.save(os.path.join(DATA_DIR, 'train_losses.npy'), np.array(train_losses))
    if val_losses:
        np.save(os.path.join(DATA_DIR, 'val_losses.npy'), np.array(val_losses))
    np.save(os.path.join(DATA_DIR, 'test_losses.npy'), np.array(test_losses))
    if test_metrics:
        np.save(os.path.join(DATA_DIR, 'test_accuracies.npy'), np.array(test_metrics))
    print("✓ Metrics saved to data/")

    print("\n" + "="*70)
    print("Training Complete!")
    print("="*70)


if __name__ == '__main__':
    main()
