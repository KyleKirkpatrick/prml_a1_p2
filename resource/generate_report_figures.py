from pathlib import Path
import sys

import joblib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from data import load_fashion_mnist


FIGURE_DIR = PROJECT_ROOT / "report" / "figures"
MODEL_PATH = PROJECT_ROOT / "models" / "logisticregression_saga_iter300.joblib"
CLASS_NAMES = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]


def save_figure(name):
    path = FIGURE_DIR / name
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved {path.relative_to(PROJECT_ROOT)}")


def plot_training_examples(x_train, y_train):
    plt.figure(figsize=(10, 4))
    for index in range(10):
        plt.subplot(2, 5, index + 1)
        plt.imshow(x_train[index], cmap="gray")
        plt.title(CLASS_NAMES[y_train[index]])
        plt.axis("off")
    plt.suptitle("Fashion-MNIST training examples")
    save_figure("training_examples.png")


def plot_confusion_matrix(y_test, y_pred):
    matrix = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(9, 8))
    plt.imshow(matrix, cmap="Blues")
    plt.colorbar(label="Number of images")
    tick_positions = np.arange(len(CLASS_NAMES))
    plt.xticks(tick_positions, CLASS_NAMES, rotation=45, ha="right")
    plt.yticks(tick_positions, CLASS_NAMES)
    threshold = matrix.max() / 2
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            colour = "white" if matrix[row, column] > threshold else "black"
            plt.text(column, row, matrix[row, column], ha="center", va="center", color=colour)
    plt.xlabel("Predicted class")
    plt.ylabel("True class")
    plt.title("Fashion-MNIST confusion matrix")
    save_figure("confusion_matrix.png")


def plot_classification_report(y_test, y_pred):
    report = classification_report(
        y_test,
        y_pred,
        target_names=CLASS_NAMES,
        output_dict=True,
    )
    rows = CLASS_NAMES + ["accuracy", "macro avg", "weighted avg"]
    columns = ["precision", "recall", "f1-score", "support"]
    values = []
    for row in rows:
        if row == "accuracy":
            values.append(["", "", f"{report[row]:.4f}", f"{int(report['weighted avg']['support'])}"])
        else:
            values.append([
                f"{report[row][column]:.4f}" if column != "support" else str(int(report[row][column]))
                for column in columns
            ])

    figure, axis = plt.subplots(figsize=(9, 5.5))
    axis.axis("off")
    table = axis.table(
        cellText=values,
        rowLabels=rows,
        colLabels=columns,
        cellLoc="center",
        rowLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.5)
    axis.set_title("Fashion-MNIST classification report")
    save_figure("classification_report.png")


def plot_predictions(x_test, y_test, y_pred, indices, title, filename, title_colour=None):
    plt.figure(figsize=(10, 4))
    for position, index in enumerate(indices[:10]):
        plt.subplot(2, 5, position + 1)
        plt.imshow(x_test[index], cmap="gray")
        title = f"Pred: {CLASS_NAMES[y_pred[index]]}\nTrue: {CLASS_NAMES[y_test[index]]}"
        if title_colour is None:
            plt.title(title, fontsize=8)
        else:
            plt.title(title, color=title_colour, fontsize=8)
        plt.axis("off")
    plt.suptitle(title)
    save_figure(filename)


def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Trained model not found: {MODEL_PATH}")

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    (x_train, y_train), (x_test, y_test) = load_fashion_mnist()
    model = joblib.load(MODEL_PATH)
    x_test_flat = x_test.reshape(x_test.shape[0], -1).astype("float32") / 255.0
    y_pred = model.predict(x_test_flat)

    plot_training_examples(x_train, y_train)
    plot_confusion_matrix(y_test, y_pred)
    plot_classification_report(y_test, y_pred)
    correct_indices = np.flatnonzero(y_pred == y_test)
    incorrect_indices = np.flatnonzero(y_pred != y_test)
    plot_predictions(
        x_test,
        y_test,
        y_pred,
        correct_indices,
        "Correct test predictions",
        "correct_predictions.png",
    )
    plot_predictions(
        x_test,
        y_test,
        y_pred,
        incorrect_indices,
        "Incorrect test predictions",
        "incorrect_predictions.png",
        title_colour="red",
    )


if __name__ == "__main__":
    main()
