import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

# saving and loading models
import joblib

# import the fashion_mnist loader from data.py
from data import load_fashion_mnist


def model_path(model, output_dir=Path("models")):
    """Return a descriptive path for a fitted model configuration."""
    model_type = type(model).__name__.lower()
    solver = getattr(model, "solver", "default")
    iterations = getattr(model, "max_iter", "default")
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir / f"{model_type}_{solver}_iter{iterations}.joblib"

# download (as needed) and load the Fashion MNIST dataset
(x_train, y_train), (x_test, y_test) = load_fashion_mnist()

print("Training set shape:", x_train.shape)
print("Test set shape:", x_test.shape)
print("Training image dtype:", x_train.dtype)
print("Test image dtype:", x_test.dtype)
print("Training label dtype:", y_train.dtype)
print("Test label dtype:", y_test.dtype)
print("Training pixel range:", (x_train.min(), x_train.max()))
print("Test pixel range:", (x_test.min(), x_test.max()))

class_names = [
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

print("Training class counts:")
for label, count in enumerate(np.bincount(y_train, minlength=len(class_names))):
    print(f"  {label} ({class_names[label]}): {count}")

print("Test class counts:")
for label, count in enumerate(np.bincount(y_test, minlength=len(class_names))):
    print(f"  {label} ({class_names[label]}): {count}")

print("First training image label:", y_train[0], f"({class_names[y_train[0]]})")
print("First training image pixel matrix:\n", x_train[0])

plt.figure(figsize=(10,5))
for i in range(5):
    plt.subplot(1, 5, i+1)
    plt.imshow(x_train[i], cmap='gray')
    plt.title(class_names[y_train[i]])      # adjusted to use the name labels
    plt.axis('off')
plt.show()


# Flatten each 28x28 image into a 784-length vector
x_train_flat = x_train.reshape(x_train.shape[0], -1)
x_test_flat = x_test.reshape(x_test.shape[0], -1)

# Normalize pixel values to range [0, 1]
x_train_flat = x_train_flat.astype('float32') / 255.0
x_test_flat = x_test_flat.astype('float32') / 255.0

# Using 'saga' solver for large datasets and multinomial classification
model = LogisticRegression(
    solver='saga',
    # multi_class='multinomial',    # 'multinomial' is the default for 'saga' solver, so this line can be omitted
    max_iter=100,      # Number of iterations
    verbose=1,         # Show training progress
    # n_jobs=-1          # deprecated
)

# Train the model
model.fit(x_train_flat, y_train)

# Saving model
saved_model_path = model_path(model)
joblib.dump(model, saved_model_path)
print(f"Saved model to: {saved_model_path}")

# Accuracy on the test set
accuracy = model.score(x_test_flat, y_test)
print(f"Test Accuracy: {accuracy:.4f}")

# Predictions for detailed evaluation
y_pred = model.predict(x_test_flat)

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cm)

# Classification report
print("\nClassification Report:\n", classification_report(y_test, y_pred))

plt.figure(figsize=(10,5))
for i in range(10):
    plt.subplot(2, 5, i+1)
    plt.imshow(x_test[i], cmap='gray')
    plt.title(f"Pred: {class_names[y_pred[i]]}\nTrue: {class_names[y_test[i]]}")    # adjusted to use the name labels
    plt.axis('off')
plt.show()

# Find indices of correct and incorrect predictions
correct_indices = np.where(y_pred == y_test)[0]
incorrect_indices = np.where(y_pred != y_test)[0]

# Visualize first 10 correct predictions
plt.figure(figsize=(10, 5))
for i, idx in enumerate(correct_indices[:10]):
    plt.subplot(2, 5, i+1)
    plt.imshow(x_test[idx], cmap='gray')
    plt.title(f"Pred: {class_names[y_pred[idx]]}\nTrue: {class_names[y_test[idx]]}")    # adjusted to use the name labels
    plt.axis('off')
plt.suptitle("Correct Predictions")
plt.show()

# Visualize first 10 incorrect predictions
plt.figure(figsize=(10, 5))
for i, idx in enumerate(incorrect_indices[:10]):
    plt.subplot(2, 5, i+1)
    plt.imshow(x_test[idx], cmap='gray')
    plt.title(f"Pred: {class_names[y_pred[idx]]}\nTrue: {class_names[y_test[idx]]}", color='red')
    plt.axis('off')
plt.suptitle("Incorrect Predictions")
plt.show()