from pathlib import Path

import kagglehub


DATASET_HANDLE = "zalando-research/fashionmnist"


def get_fashion_mnist_path() -> Path:
    dataset_path = kagglehub.dataset_download(DATASET_HANDLE)

    path = Path(dataset_path)
    print(f"Fashion-MNIST dataset available at: {path}")

    return path
