import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.datasets import mnist


# Load data: (x_train, y_train) for training, (x_test, y_test) for testing
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Put train and test together into a whole dataset (we do this because we want to use k-fold cross validation)
X = np.concatenate([X_train, X_test])
y = np.concatenate([y_train, y_test])

print("Dataset shape:", X.shape)


plt.figure(figsize=(10,5))
for i in range(5):
    plt.subplot(1, 5, i+1)
    plt.imshow(X[i], cmap='gray')
    plt.title(f"Label: {y[i]}")
    plt.axis('off')
plt.show()


# Flatten images (28*28 -> 784)
X = X.reshape(-1, 28*28)

# Normalize pixel values (0-255 -> 0-1)
X = X / 255.0

kf = KFold(n_splits=5, shuffle=True, random_state=42)

model_v0 = LogisticRegression(
    penalty=None,     # No penalty is added
    solver='lbfgs',   # Solver that supports no regularization
    max_iter=500,     # Number of iterations
    random_state=42
)

model_l1 = LogisticRegression(
    penalty="l1",              # L1 regularization only
    solver="liblinear",        # 'liblinear' or 'saga' works for L1
    C=1.0,                     # Inverse of regularization strength
    max_iter=500,              # Ensure convergence
    random_state=42
)

model_l2 = LogisticRegression(
    penalty="l2",              # L2 regularization only
    solver="lbfgs",            # Default solver works well with L2
    C=1.0,                     # Inverse of regularization strength
    max_iter=500,              # Ensure convergence
    random_state=42
)

model_l1l2 = LogisticRegression(
    penalty="elasticnet",      # Use both L1 and L2
    solver="saga",             # Required for elasticnet
    l1_ratio=0.5,              # Balance between L1 and L2 (0 = L2 only, 1 = L1 only)
    C=1.0,                     # Inverse of regularization strength
    max_iter=500,              # Ensure convergence
    random_state=42
)


print("Evaluating base model...")
scores_v0 = cross_val_score(model_v0, X, y, cv=kf, scoring="accuracy", n_jobs=-1)

print("Evaluating L1 model...")
scores_l1 = cross_val_score(model_l1, X, y, cv=kf, scoring="accuracy", n_jobs=-1)

print("Evaluating L2 model...")
scores_l2 = cross_val_score(model_l2, X, y, cv=kf, scoring="accuracy", n_jobs=-1)

print("Evaluating L1 and L2 model...")
scores_l1l2 = cross_val_score(model_l1l2, X, y, cv=kf, scoring="accuracy", n_jobs=-1)

# 6. Results
print("\nResults (5-fold cross-validation on MNIST subset):")
print(f"V0 Accuracy: {scores_v0.mean():.4f} ± {scores_v0.std():.4f}")
print(f"L1 Accuracy: {scores_l1.mean():.4f} ± {scores_l1.std():.4f}")
print(f"L2 Accuracy: {scores_l2.mean():.4f} ± {scores_l2.std():.4f}")
print(f"L1+L2 Accuracy: {scores_l1l2.mean():.4f} ± {scores_l1l2.std():.4f}")


