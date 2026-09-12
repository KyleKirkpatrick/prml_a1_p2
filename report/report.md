## Problem Introduction
This project implements a solution to train a machine-learning model to label images depicting various articles of clothing. It uses the publicly accessible `Fashion-MNIST` dataset for training and testing. The solution is adapted from previous code examples using the MNIST database of handwritten digits, restructured to import the fashion database instead. 

By implementing this solution, I aim to explore the following questions:
- How is the Fashion-MNIST dataset structured? 
- How is data curated for ML training?
- How is logistic regression implemented for training a ML model?

## Dataset Description
The `Fashion-MNIST` dataset contains 70,000 images of clothing articles curated for training machine-learning algorithms.[^1] The images are pre-processed to 28x28 grayscale images with high-contrast emphasising distinctive features while minimising file size. Each image has an associated class label represented by a number that can be matched to a list of clothing types (T-shirt/top, Trouser, Pullover, Dress, etc.). 

The Kaggle distribution of `Fashion-MNIST` includes a CSV format that combines the labels and the images, which I am not using for my implementation. The IDX format is the same format used for the `MNIST` handwriting dataset, and is simpler to use for my implementation. It includes four files; training images, training labels, testing images, and testing labels.

The files each start with a header that defines the structure of the data. First, a magic number codes the datatype in the dataset (`0x08` unsigned byte) and the number of dimensions of the matrix. Then, the next value stores the number of items in the set. For the image sets, the third and fourth values in the header store the number of rows and the number of columns, respectively. The value in this header shows 28 rows and 28 columns representing the image size of 28x28 pixels. With this header data, we can determine where the pixels for one image end and the next image begin within the large binary blob. As the images are 28x28, each image is 784 pixels long.

## Justification for Using Logistic Regression
Logistic regression is appropriate for this task because it can assign each image to one of ten classes using the pixel values as input features. In this implementation, the `saga` solver fits a multinomial classifier with one output probability for each clothing class. The predicted label is the class with the highest probability.

Logistic regression provides a useful baseline for Fashion-MNIST. It is relatively simple to train, its preprocessing requirements are clear, and its coefficients represent the contribution of each pixel to the class decisions. It also provides measurable precision, recall, and F1-scores for each class, which makes its behaviour straightforward to inspect.

The main limitation is that the model uses a linear decision function over flattened pixels. It does not directly represent local shapes, edges, or spatial relationships between neighbouring pixels. This limits its ability to distinguish visually similar classes. The test results show this limitation most clearly for Shirt, which had a recall of `0.5680`, compared with a recall of `0.9580` for Trouser.

## Data Retrieval
The assignment has tasked me with accessing the `Fashion-MNIST` dataset through Kaggle, so I went to extra effort to implement it using this repository. The easy way to import the dataset would be to import it from TensorFlow:

``` 
from tensorflow.keras.datasets import fashion_mnist as mnist
```

If it was imported in this way, the sample code this project was based on could be used largely unmodified. Instead, I created a script named `data.py` and used the `kagglehub` library provided by kaggle to import `Fashion-MNIST`.[^1] 

```
dataset_path = Path(kagglehub.dataset_download("zalando-research/fashionmnist"))
```

`kagglehub.dataset_download()` retrieves the dataset if it is not already cached on the current machine. The local path to the dataset is then stored in `dataset_path`. I add a section name to the path to extract particular portions of the dataset. 

```
x_train = load_idx_images(dataset_path / "train-images-idx3-ubyte")
```

`load_idx_images()` is a function I wrote to parse the images from a given section of the database. It reads the remaining image bytes into a NumPy array and reshapes that array using the dimensions from the header. This follows the IDX format specified in the [MNIST Github repository](https://github.com/sunsided/mnist).[^2] The files in `Fashion-MNIST` are in the same IDX format as the `MNIST` handwritten digit database. The image file header consists of four 32-bit integers representing the magic number, number of images, number of rows, and number of columns, respectively. The integers are stored in big-endian format, with the most significant bit first. A string, `">IIII"`, provides the layout to `struct.unpack()`. I do not use the magic number.

```
def load_idx_images(path: Path) -> np.ndarray:
   with path.open("rb") as file:
        magic, count, rows, columns = struct.unpack(">IIII", file.read(16),)
        images = np.frombuffer(file.read(), dtype=np.uint8,)
    return images.reshape(count, rows, columns)
```

After the header is extracted, the images are read one pixel at a time. Each pixel is an unsigned 8-bit integer. The pixels are stored in the images array. After the pixels have been read, `reshape()` is used to structure the pixel data into separate images based on the number of images (count) and the resolution (rows & columns).

Labels are handled similarly in the `load_idx_labels` function. Labels are only integer values, so they must be reconciled later with an array of label names. The images and labels are returned from `data.py` together. 
```
return (x_train, y_train), (x_test, y_test)
```

The images and labels are imported into the `main.py` script using the following code:
```
from data import load_fashion_mnist
(x_train, y_train), (x_test, y_test) = load_fashion_mnist()
```
I have structured the imported data in the same way that it is presented when imported from `tensorflow.keras.datasets`, so the implementation in `main.py` is the same with either import method. 

## Data Exploration
The loader returned `x_train` with shape `(60000, 28, 28)` and `x_test` with shape `(10000, 28, 28)`. The image arrays use the `uint8` data type. Raw pixel values range from `0` to `255`, where `0` represents black and `255` represents the highest grayscale intensity in the stored image.

The labels use the `uint8` data type and values from `0` to `9`. The class names are T-shirt/top, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, and Ankle boot. Each class contains 6,000 training images and 1,000 test images, so the supplied split is balanced across the ten classes.

The program displays labelled examples from the training data. It also prints the complete 28 x 28 pixel matrix for the first training image. That image has label `9`, corresponding to Ankle boot. The matrix contains mostly zero-valued background pixels and higher values around the outline and body of the item. This confirms that the image data contains grayscale intensity values rather than binary pixels.

Before training, I reshaped each image from a 28 x 28 matrix into a vector of 784 features. I then converted the vectors to `float32` and normalised each pixel by dividing by `255`. The resulting feature range is `[0, 1]`, which provides a consistent numeric scale for the logistic-regression model.

## Building the Logistic Regression Model
### i. Load required Python libraries/packages
I used `uv` to add the required libraries to the project, which creates a `pyproject.toml` file that lists the dependencies and a `uv.lock` file that stores the package names and their repository locations. To load the project on any given machine first install `uv`, clone the project repo, and then run this command in the project directory:
```
uv sync
```
This will install all needed dependencies and create a local virtual environment for the project. 

With the dependencies installed, the following code in `main.py` imports the needed libraries:
```
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
```
`data.py` is also imported to ingest the `Fashion-MNIST` dataset as outlined in [Data Retrieval](#data-retrieval)
```
from data import load_fashion_mnist
(x_train, y_train), (x_test, y_test) = load_fashion_mnist()
```

The target variables are `y_train` and `y_test`. The corresponding image arrays, `x_train` and `x_test`, are the input features. I retained the supplied 60,000-image training set and 10,000-image test set. The test set was not used to select regularisation settings, so it provides a held-out evaluation of the final fitted model.
### ii. Select the target variable (labels)

The target variable is the label array. `y_train` contains the training labels and `y_test` contains the test labels. Each value identifies one of the ten Fashion-MNIST classes. The image arrays are the input features associated with these labels.

### iii. Prepare the data (e.g., flatten images, normalize pixel values)

Each 28 x 28 image was reshaped into a vector containing 784 pixel features. I converted the pixel vectors to `float32` and divided each value by `255`, changing the feature range from `[0, 255]` to `[0, 1]`. This produces the numeric input used by logistic regression.

### iv. Split the data into training and validation sets

The dataset already provides separate training and test sets. I used the 60,000 training images to fit the final model and retained the 10,000 test images for final evaluation. The test set was not used as a validation set. Regularisation settings were compared with three-fold stratified cross-validation on a stratified 10,000-image subset of the training data.

### v. Initialize a logistic regression classifier

The final classifier uses the `saga` solver with pure L2 regularisation. In scikit-learn 1.9, this is represented by `l1_ratio=0.0`. The model uses `max_iter=300`, `tol=1e-3`, and `random_state=42`.

```python
model = LogisticRegression(
    l1_ratio=0.0,
    solver="saga",
    tol=1e-3,
    max_iter=300,
    random_state=42,
)
```

### vi. Train (fit) the model on the training data

The model was fitted using the normalised training features and their associated labels:

```python
model.fit(x_train_flat, y_train)
```

The saved model reported `n_iter_=[246]`, meaning that the final fit stopped before reaching the configured maximum of 300 iterations.

### vii. Evaluate predictions on unseen (validation/test) data

After fitting, I generated predictions for the held-out test features. The final model achieved a test accuracy of `0.8439`. I also calculated a confusion matrix and a classification report containing precision, recall, F1-score, and support for each class.

## Results Analysis
### i. Generate a classification report (precision, recall, F1-score)

The final classification report produced a macro-average precision of `0.8427`, macro-average recall of `0.8439`, and macro-average F1-score of `0.8430`. Trouser had the highest recall at `0.9580`, while Shirt had the lowest recall at `0.5680`.

### ii. Create a confusion matrix to visualize model performance

The confusion matrix shows that the model classified footwear classes more consistently than several upper-body clothing classes. The largest errors involved Shirt being predicted as T-shirt/top, Pullover, Coat, or Dress. Pullover was also often confused with Coat and Shirt. These errors are consistent with the similar silhouettes and overlapping pixel patterns of these classes.

### iii. Show correct predictions using example images

The program displays examples from the test set with their predicted and true class names. These examples provide visual evidence of predictions where the model assigned the correct class.

### iv. Show misclassified examples and explore potential causes

The program also displays incorrect predictions with the predicted and true class names. The most difficult class was Shirt, which had a recall of `0.5680`. The errors may result from the 28 x 28 image resolution, similar shapes between upper-body classes, and the linear decision boundary produced from flattened pixel features.

### v. Optionally, include “corrected” images or improvements based on your insights(e.g., better preprocessing, tuning, or visualization)

## Regularization in Logistic Regression
• Explain the concept of regularization (L1, L2) in logistic regression.
• Describe how regularization helps prevent overfitting.
• Discuss how you applied or could apply regularization in this modeling task and how
it affects model performance on Fashion-MNIST.

Regularisation adds a penalty to the model coefficients during training. It discourages unnecessarily large coefficients and can reduce sensitivity to noise in the training data. L1 regularisation encourages some coefficients to become exactly zero, while L2 regularisation reduces the magnitude of all coefficients without usually removing them completely. The parameter `C` controls the inverse of regularisation strength: smaller values apply stronger regularisation.

I compared L1 and L2 regularisation using three-fold stratified cross-validation on a stratified subset of 10,000 training images. Both comparisons used the `saga` solver, `max_iter=300`, `tol=1e-3`, and `random_state=42`. L1 produced an accuracy of `0.8364 +/- 0.0023`, while L2 produced `0.8311 +/- 0.0018`. The L1 fits reached the iteration limit, so this comparison should be reported as provisional rather than as definitive evidence that L1 is superior.

The final model uses pure L2 regularisation and was evaluated separately on the held-out test set. It achieved an accuracy of `0.8439`.

## Saving and Using the Trained Model
• Show how you save the trained model for future use
• Explain how the saved model can be used to make predictions on new, unseen data.
• (Optional) Provide a short code snippet showing how to load the model and predict
on a new input.

The fitted estimator is saved with joblib in the `models/` directory. Its filename is generated from the estimator type, solver, and maximum iteration setting. The current file is `models/logisticregression_saga_iter300.joblib`.

```python
saved_model_path = model_path(model)
joblib.dump(model, saved_model_path)

loaded_model = joblib.load(saved_model_path)
prediction = loaded_model.predict(x_test_flat[:1])
```

The input to `loaded_model.predict()` must use the same preprocessing as the training data: each image must be flattened to 784 features and its pixel values must be normalised to `[0, 1]`. The saved model predicted class `9` for the first test image, matching the prediction from the fitted model.

## References
[^1]:Zalando Research, “Fashion-MNIST,” Kaggle dataset. [Online]. Available: https://www.kaggle.com/datasets/zalando-research/fashionmnist. [Accessed: Sep. 10, 2026].

[^2]: M. Mayer (sunsided), “MNIST,” GitHub repository, containing specifications credited to Y. LeCun and C. Cortes. [Online]. Available: https://github.com/sunsided/mnist. [Accessed: Sep. 10, 2026].