# -------------------------------
# Fashion-MNIST: Logistic Regression Example
# -------------------------------

# 1: Import required libraries
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import fashion_mnist
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler

# -------------------------------
# 2: Load the Fashion-MNIST dataset
# -------------------------------
# Fashion-MNIST has 60,000 training and 10,000 test images
# Each image is 28x28 grayscale
(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()

# Class labels for reference
class_names = [
    'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
    'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'
]

print("Training set size:", X_train.shape)
print("Test set size:", X_test.shape)


# -------------------------------
# 3: Visualise some example images
# -------------------------------
plt.figure(figsize=(10, 4))
for i in range(10):
    plt.subplot(2, 5, i+1)
    plt.imshow(X_train[i], cmap='gray')
    plt.title(class_names[y_train[i]])
    plt.axis('off')
plt.show()


