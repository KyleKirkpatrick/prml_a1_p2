from pathlib import Path
import struct

import kagglehub
import numpy as np


def load_idx_images(path: Path) -> np.ndarray:
    with path.open("rb") as file:
        magic, count, rows, columns = struct.unpack(
            ">IIII",
            file.read(16),
        )

        images = np.frombuffer(
            file.read(),
            dtype=np.uint8,
        )

    return images.reshape(count, rows, columns)


def load_idx_labels(path: Path) -> np.ndarray:
    with path.open("rb") as file:
        magic, count = struct.unpack(
            ">II",
            file.read(8),
        )

        labels = np.frombuffer(
            file.read(),
            dtype=np.uint8,
        )

    return labels


def load_fashion_mnist():
    dataset_path = Path(
        kagglehub.dataset_download("zalando-research/fashionmnist")
    )

    x_train = load_idx_images(
        dataset_path / "train-images-idx3-ubyte"
    )
    y_train = load_idx_labels(
        dataset_path / "train-labels-idx1-ubyte"
    )

    x_test = load_idx_images(
        dataset_path / "t10k-images-idx3-ubyte"
    )
    y_test = load_idx_labels(
        dataset_path / "t10k-labels-idx1-ubyte"
    )

    return (x_train, y_train), (x_test, y_test)
