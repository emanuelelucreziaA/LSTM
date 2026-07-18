"""
Data utilities: synthetic sequential dataset loading and preprocessing.
"""

import numpy as np


def one_hot_encode(y, num_classes=10):
    """
    Convert class labels to one-hot encoding.

    Args:
        y: Class labels (batch_size,)
        num_classes: Number of classes

    Returns:
        y_one_hot: One-hot encoded labels (batch_size, num_classes)
    """
    batch_size = len(y)
    one_hot = np.zeros((batch_size, num_classes), dtype=np.float32)
    one_hot[np.arange(batch_size), y] = 1.0
    return one_hot


class SequenceDataLoader:
    """
    Synthetic sequence data loader for RNN classification.

    Generates labelled sequences with distinct waveform patterns.
    This is a more natural dataset for RNNs than treating MNIST images
    as sequences.
    """

    def __init__(
        self,
        seq_len=50,
        input_size=1,
        num_classes=10,
        train_samples=5000,
        test_samples=1000,
        noise_level=0.1,
        seed=42,
    ):
        self.seq_len = seq_len
        self.input_size = input_size
        self.num_classes = num_classes
        self.train_samples = train_samples
        self.test_samples = test_samples
        self.noise_level = noise_level
        self.seed = seed
        self.rng = np.random.default_rng(seed)

        self.x_train = None
        self.y_train = None
        self.x_test = None
        self.y_test = None

    def load_data(self):
        """
        Generate synthetic sequence training and test data.

        Returns:
            (x_train, y_train, x_test, y_test)
        """
        x_train, y_train = self._generate_data(self.train_samples)
        x_test, y_test = self._generate_data(self.test_samples)

        self.x_train = x_train.astype(np.float32)
        self.y_train = y_train.astype(np.int64)
        self.x_test = x_test.astype(np.float32)
        self.y_test = y_test.astype(np.int64)

        print(
            f"✓ Synthetic sequence data generated: train={x_train.shape}, test={x_test.shape}"
        )
        return self.x_train, self.y_train, self.x_test, self.y_test

    def _generate_data(self, num_samples):
        """Generate a labelled dataset of waveform sequences."""
        X = np.zeros((num_samples, self.seq_len, self.input_size), dtype=np.float32)
        y = np.zeros((num_samples,), dtype=np.int64)

        for idx in range(num_samples):
            label = int(self.rng.integers(low=0, high=self.num_classes))
            X[idx] = self._create_sequence(label)
            y[idx] = label

        return X, y

    def _create_sequence(self, label):
        """Create a single sequence for the given class label."""
        t = np.linspace(0.0, 1.0, self.seq_len, dtype=np.float32)
        phase = (label / self.num_classes) * 2.0 * np.pi
        frequency = 1.0 + label * 0.08

        if label % 3 == 0:
            signal = np.sin(2.0 * np.pi * frequency * t + phase)
        elif label % 3 == 1:
            signal = np.cos(2.0 * np.pi * frequency * t + phase)
        else:
            signal = np.sin(2.0 * np.pi * frequency * t + phase) * (1.0 - t)

        # Introduce an additional per-class modulation
        signal = signal * (1.0 + 0.1 * label)
        noise = self.rng.normal(scale=self.noise_level, size=signal.shape).astype(np.float32)
        signal = signal + noise

        return signal.reshape(self.seq_len, self.input_size)

    def get_batch(self, indices, mode='train'):
        """Get a batch by index for either train or test set."""
        if mode == 'train':
            x = self.x_train[indices]
            y = self.y_train[indices]
        else:
            x = self.x_test[indices]
            y = self.y_test[indices]
        return x, y

    def get_all_train(self):
        """Return the full training set."""
        return self.x_train, self.y_train

    def get_all_test(self):
        """Return the full test set."""
        return self.x_test, self.y_test


class MNISTLoader:
    """
    Compatibility wrapper for legacy MNISTLoader usage.

    This loader now generates synthetic sequence data, but preserves
    the old import interface for notebooks and scripts that still
    instantiate MNISTLoader(data_dir).
    """

    def __init__(self, data_dir=None, **kwargs):
        if isinstance(data_dir, str):
            print(
                "Warning: MNISTLoader now generates synthetic sequence data. "
                "Ignoring the data_dir path argument."
            )
        self._loader = SequenceDataLoader(**kwargs)

    def load_data(self):
        return self._loader.load_data()

    def get_batch(self, indices, mode='train'):
        return self._loader.get_batch(indices, mode=mode)

    def get_all_train(self):
        return self._loader.get_all_train()

    def get_all_test(self):
        return self._loader.get_all_test()


# Backwards compatibility alias for scripts that still import MNISTLoader
# (preferred loader is SequenceDataLoader)
MNISTLoader = MNISTLoader
