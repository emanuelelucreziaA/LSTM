"""
Evaluation script: Load trained model and evaluate on test set

Usage:
    python evaluate.py
"""

import csv
import os
from dotenv import load_dotenv
load_dotenv()
import sys
PROJECT_ROOT = os.getenv('PROJECT_ROOT', os.path.abspath(os.path.dirname(__file__)))
sys.path.insert(0, PROJECT_ROOT)
import numpy as np

import pickle
from lstm.lstm_layer import LSTMLayer
from lstm.dense_layer import DenseLayer
from lstm.network import LSTMNetwork
from lstm.activations import softmax
from lstm.data import SequenceDataLoader


def load_model(weights_path):
    """Load model metadata"""
    try:
        with open(weights_path, 'rb') as f:
            model_data = pickle.load(f)

        print("✓ Model metadata loaded successfully")
        return model_data
    except FileNotFoundError:
        print(f"✗ Model weights not found at {weights_path}")
        print("  First train the model using: python train.py")
        return None


def mse_loss(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.float32).reshape(y_pred.shape)
    y_pred = np.asarray(y_pred, dtype=np.float32)
    return np.mean((y_pred - y_true) ** 2)


def prepare_air_passengers(data_dir, seq_len=12, horizon=1, train_ratio=0.7, val_ratio=0.15):
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


def build_model(input_size=1, output_size=10, hidden_size=128):
    """Build model architecture (matching training)"""
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
    
    return model


def confusion_matrix(y_true, y_pred, num_classes=10):
    """Compute confusion matrix"""
    cm = np.zeros((num_classes, num_classes), dtype=int)
    
    for i in range(len(y_true)):
        cm[int(y_true[i]), int(y_pred[i])] += 1
    
    return cm


def analyze_predictions(y_true, y_pred):
    """Analyze model predictions"""
    print("\n" + "="*70)
    print("Prediction Analysis")
    print("="*70)
    
    # Accuracy
    accuracy = np.mean(y_true == y_pred)
    print(f"\nOverall Accuracy: {accuracy:.4f} ({int(accuracy*100)}%)")
    
    # Per-class accuracy
    print("\nPer-Class Accuracy:")
    for digit in range(10):
        mask = y_true == digit
        if np.sum(mask) > 0:
            class_acc = np.mean(y_pred[mask] == digit)
            count = np.sum(mask)
            print(f"  Digit {digit}: {class_acc:.4f} ({count} samples)")
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    print("\nConfusion Matrix (first 5x5):")
    print(cm[:5, :5])


def main():
    """Main evaluation pipeline"""
    print("\n" + "="*70)
    print("LSTM Evaluation: synthetic sequence classification")
    print("="*70)
    
    # Load model metadata
    model_path = os.path.join(PROJECT_ROOT, 'lstm_model.pkl')
    model_data = load_model(model_path)

    if model_data is None:
        return

    dataset_mode = os.getenv('DATASET', 'air-passengers').lower()
    print(f"\nDataset mode: {dataset_mode}")

    if dataset_mode == 'air-passengers':
        print("\nLoading AirPassengers test data...")
        X_train, y_train, X_val, y_val, X_test, y_test = prepare_air_passengers(
            os.path.join(PROJECT_ROOT, 'data'),
            seq_len=12,
            horizon=1,
            train_ratio=0.7,
            val_ratio=0.15,
        )
        regression = True
        output_size = 1
        hidden_size = 64
    else:
        print("\nLoading synthetic sequential test data...")
        loader = SequenceDataLoader(
            seq_len=50,
            input_size=1,
            num_classes=10,
            train_samples=5000,
            test_samples=1000,
            noise_level=0.1,
            seed=42,
        )
        X_train, y_train, X_test, y_test = loader.load_data()
        regression = False
        output_size = 10
        hidden_size = 128

    # Build and setup model
    print("\nBuilding model architecture...")
    model = build_model(
        input_size=1,
        output_size=output_size,
        hidden_size=hidden_size,
    )

    if 'weights' not in model_data or model_data['weights'] is None:
        print("✗ No trained weights found in saved model data.")
        print("  Train the model first using: python train.py")
        return

    model.set_weights(model_data['weights'])

    # Predictions
    print("\nGenerating predictions...")
    if regression:
        scaler = model_data.get('scaler')
        if scaler is not None:
            X_test_norm = (X_test - scaler['mean']) / scaler['std']
        else:
            X_test_norm = X_test

        logits = model.forward(X_test_norm)
        y_pred = np.squeeze(logits, axis=-1)
        if scaler is not None:
            y_pred = y_pred * scaler['std'] + scaler['mean']

        test_loss = mse_loss(y_test, y_pred)
        print(f"Test MSE: {test_loss:.4f}")
    else:
        y_pred = model.predict(X_test)
        analyze_predictions(y_test, y_pred)

    # Training history
    if 'test_metrics' in model_data and model_data['test_metrics']:
        print("\n" + "="*70)
        print("Training History")
        print("="*70)
        print("\nTest Metrics by Epoch:")
        for epoch, metric in enumerate(model_data['test_metrics']):
            print(f"  Epoch {epoch+1}: {metric:.4f}")

    if 'test_losses' in model_data:
        print("\nTest Losses by Epoch:")
        for epoch, loss in enumerate(model_data['test_losses']):
            print(f"  Epoch {epoch+1}: {loss:.4f}")

    print("\n" + "="*70)
    print("Evaluation Complete!")
    print("="*70)


if __name__ == '__main__':
    main()
