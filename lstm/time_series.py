"""Shared time-series utilities for AirPassengers regression."""

import csv
import os

import numpy as np


def prepare_air_passengers(
    data_dir,
    seq_len=12,
    horizon=1,
    train_ratio=0.7,
    val_ratio=0.15,
):
    """Load AirPassengers.csv and build train/val/test sequence splits."""
    csv_path = os.path.join(data_dir, 'AirPassengers.csv')
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"AirPassengers.csv not found in {data_dir}")

    values = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
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
    """Normalize all splits using training split statistics only."""
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
    """Convert normalized predictions back to original scale."""
    return y * scaler['std'] + scaler['mean']
