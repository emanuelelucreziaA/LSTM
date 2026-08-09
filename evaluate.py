"""
Evaluation script: Load trained model and evaluate on the AirPassengers test set.

Usage:
    python evaluate.py
"""

import os
import numpy as np
import pickle

from lstm.bootstrap import initialize_environment

PROJECT_ROOT, _ = initialize_environment()

from lstm.network import build_lstm_regression_model
from lstm.losses import MSELoss
from lstm.time_series import prepare_air_passengers, inverse_scale

_loss_fn = MSELoss()


def load_model(weights_path):
    """Load trained model checkpoint."""
    try:
        with open(weights_path, "rb") as f:
            model_data = pickle.load(f)

        print("✓ Model metadata loaded successfully")
        return model_data

    except FileNotFoundError:
        print(f"✗ Model weights not found at {weights_path}")
        print("  First train the model using: python train.py")
        return None

def main():
    """Main evaluation pipeline"""
    print("\n" + "="*70)
    print("LSTM Evaluation: AirPassengers regression")
    print("="*70)
    
    # Load model metadata
    model_path = os.path.join(PROJECT_ROOT, 'lstm_model.pkl')
    model_data = load_model(model_path)

    if model_data is None:
        return

    print("\nLoading AirPassengers test data...")
    _, _, _, _, X_test, y_test = prepare_air_passengers(
        os.path.join(PROJECT_ROOT, 'data'),
        seq_len=12,
        horizon=1,
        train_ratio=0.7,
        val_ratio=0.15,
    )

    output_size = int(model_data.get('output_size', 1))
    hidden_size = int(model_data.get('hidden_size', 64))

    print("\nBuilding model architecture...")

    model = build_lstm_regression_model(
        input_size=X_test.shape[2],
        output_size=output_size,
        hidden_size=hidden_size,
    )

    if "weights" not in model_data or model_data["weights"] is None:
        print("✗ No trained weights found.")
        print("  Train the model first using: python train.py")
        return

    model.set_weights(model_data["weights"])

    # Predictions
    print("\nGenerating predictions...")
    scaler = model_data.get('scaler')
    if scaler is None:
        print("✗ Missing scaler in saved model data.")
        print("  Re-train the model using: python train.py")
        return

    X_test_norm = (X_test - scaler['mean']) / scaler['std']
    logits = model.forward(X_test_norm)
    y_pred = np.squeeze(logits, axis=-1)
    y_pred = inverse_scale(y_pred, scaler)
    # y_test from prepare_air_passengers is already in original scale.
    y_test_original = y_test

    test_loss = _loss_fn(y_test_original, y_pred)
    print(f"Test MSE: {test_loss:.4f}")

    # Training history
    if 'test_losses' in model_data:
        print("\n" + "="*70)
        print("Training History")
        print("="*70)
        print("\nTest Losses by Epoch:")
        for epoch, loss in enumerate(model_data["test_losses"], start=1):
            print(f"  Epoch {epoch}: {loss:.4f}")

    print("\n" + "=" * 70)
    print("Evaluation Complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()