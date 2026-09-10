import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.datasets import mnist


# Load data: (x_train, y_train) for training, (x_test, y_test) for testing
(x_train, y_train), (x_test, y_test) = mnist.load_data()

print("Training set shape:", x_train.shape)
print("Test set shape:", x_test.shape)

plt.figure(figsize=(10,5))
for i in range(5):
    plt.subplot(1, 5, i+1)
    plt.imshow(x_train[i], cmap='gray')
    plt.title(f"Label: {y_train[i]}")
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
    multi_class='multinomial',
    max_iter=100,      # Number of iterations
    verbose=1,         # Show training progress
    n_jobs=-1          # Use all CPU cores for speed
)

# Train the model
model.fit(x_train_flat, y_train)

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
    plt.title(f"Pred: {y_pred[i]}\nTrue: {y_test[i]}")
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
    plt.title(f"Pred: {y_pred[idx]}\nTrue: {y_test[idx]}")
    plt.axis('off')
plt.suptitle("Correct Predictions")
plt.show()

# Visualize first 10 incorrect predictions
plt.figure(figsize=(10, 5))
for i, idx in enumerate(incorrect_indices[:10]):
    plt.subplot(2, 5, i+1)
    plt.imshow(x_test[idx], cmap='gray')
    plt.title(f"Pred: {y_pred[idx]}\nTrue: {y_test[idx]}", color='red')
    plt.axis('off')
plt.suptitle("Incorrect Predictions")
plt.show()