from pathlib import Path
import struct

import kagglehub
import numpy as np


def load_idx_images(path: Path) -> np.ndarray:
    """Reads IDX image files and returns a NumPy array of images.
    
    Returns:
        A NumPy array of shape (num_samples, rows, columns) 
        in data type uint8, where each image is represented 
        as a 2D array of pixel values.
    """

    with path.open("rb") as file:
        magic, count, rows, columns = struct.unpack(
            # reads the header of the IDX file and unpacks it into four unsigned integers
            # image file header structure:
            # >: big-endian
            # I: unsigned 32-bit int: magic number
            # I: unsigned 32-bit int: number of images
            # I: unsigned 32-bit int: number of rows
            # I: unsigned 32-bit int: number of columns
            ">IIII",
            file.read(16),
        )

        images = np.frombuffer(
            # reads the image after the header and stores it in the NumPy array
            file.read(),
            dtype=np.uint8,     # each pixel is represented as an unsigned 8-bit integer
        )

    return images.reshape(count, rows, columns) # reshapes the flat array into a 3D array of shape (num_samples, rows, columns)


def load_idx_labels(path: Path) -> np.ndarray:
    """Reads IDX label files and returns a NumPy array of labels."""
    with path.open("rb") as file:
        magic, count = struct.unpack(
            # reads the header of the IDX file and unpacks it into two unsigned integers
            # label file header structure:
            # >: big-endian
            # I: unsigned 32-bit int: magic number
            # I: unsigned 32-bit int: number of labels
            ">II",
            file.read(8),
        )

        labels = np.frombuffer(
            # reads the label data after the header and stores it in the NumPy array
            file.read(),
            dtype=np.uint8,     # each label is represented as an unsigned 8-bit integer
        )

    return labels


def load_fashion_mnist():
    """Loads the Fashion-MNIST dataset from Kaggle and returns training and test sets.
    
    Returns:
        Tuple of (x_train, y_train), (x_test, y_test)
    """
    dataset_path = Path(
        kagglehub.dataset_download("zalando-research/fashionmnist")
    )

    # distinguish the training and test sets by loading the corresponding IDX files
    # uses the same format as PRML lab 3 sample code
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
