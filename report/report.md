## Problem Introduction
This project implements a solution to train a machine-learning model to label images depicting various articles of clothing. It uses the publicly accessible `Fashion-MNIST` dataset for training and testing. The solution is adapted from previous code examples using the MNIST database of handwritten digits, restructured to import the fashion database instead. 

By implementing this solution, I aim to explore the following questions:
- How is the Fashion-MNIST dataset structured? 
- How is data curated for ML training?
- How is logistic regression implemented for training a ML model?

## Dataset Description
The `Fashion-MNIST` dataset contains 70,000 images of clothing articles curated for training machine-learning algorithms.[^1] The images are pre-processed to 28x28 grayscale images with high-contrast emphasising distinctive features while minimising file size. Each image has an associated class label represented by a number that can be matched to a list of clothing types (T-shirt/top, Trouser, Pullover, Dress, etc.). 

The kaggle distribution of `Fashion-MNIST` includes a CSV format that combines the labels and the images, which I am not using for my implementation. The IDX format is the same format used for the `MNIST` handwriting dataset, and is simpler to use for my implementation. It includes four files; training images, training labels, testing images, and testing labels. 

The files each start with a header that defines the structure of the data. First, a magic number codes the datatype in the dataset (`0x08` unsigned byte) and the number of dimensions of the matrix. Then, the next value stores the number of items in the set. For the image sets, the 3rd and 4th values in the header store the number of rows and the number of columns, respectively. The images are 28x28 pixels in size corresponding to 28 rows and 28 columns. With this header data, we can determine where the pixels for one image end and the next image begin within the large binary blob. As the images are 28x28, each image is 784 pixels long. 

## Justification for Using Logistic Regression
Logistic regression is appropriate for this task because it can assign each image to one of ten classes using the pixel values as input features. In this implementation, the `saga` solver fits a multinomial classifier with one output probability for each clothing class. The predicted label is the class with the highest probability.

Logistic regression provides a useful baseline for Fashion-MNIST. It is relatively simple to train, its preprocessing requirements are clear, and its coefficients represent the contribution of each pixel to the class decisions. It also provides measurable precision, recall, and F1-scores for each class, which makes its behaviour straightforward to inspect.

The main limitation is that the model uses a linear decision function over flattened pixels. It does not directly represent local shapes, edges, or spatial relationships between neighbouring pixels. This limits its ability to distinguish visually similar classes. The test results show this limitation most clearly for Shirt, which had a recall of `0.5680`, compared with a recall of `0.9580` for Trouser.

## Data Retrieval
The assignment has tasked me with accessing the `Fashion-MNIST` dataset through kaggle, so I went to extra effort to implement it using this repository. The easy way to import the dataset would be to import it from tensorflow:

``` 
from tensorflow.keras.datasets import fashion-mnist as mnist
```

If it was imported in this way, the sample code this project was based on could be used largely unmodified. Instead, I created a script named `data.py` and used the `kagglehub` library provided by kaggle to import `Fashion-MNIST`.[^1] 

```
dataset_path = Path(kagglehub.dataset_download("zalando-research/fashionmnist"))
```

`kagglehub.dataset_download()` retrieves the dataset if it is not already cached on the current machine. The local path to the dataset is then stored in `dataset_path`. I add a section name to the path to extract particular portions of the dataset. 

```
x_train = load_idx_images(dataset_path / "train-images-idx3-ubyte")
```

`load_idx_images()` is a function I wrote to parse the images from a given section of the database. It iterates through the file based on the IDX format specified in the [MNIST Github repository](https://github.com/sunsided/mnist).[^2] The files in `Fashion-MNIST` are in the same IDX format as the `MNIST` handwritten digit database. The image file header consists of four 32-bit integers representing the magic number, number of images, number of rows, and number of columns, respectively. The integers are stored in big-endian format, with the most significant bit first. A string, `">IIII"`, provides the layout to `struct.unpack()`. I do not use the magic number. 

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
`data.py` is also imported to ingest the `Fashion_MNIST` dataset as outlined in [Data Retrieval](#data-retrieval)
```
from data import load_fashion_mnist
(x_train, y_train), (x_test, y_test) = load_fashion_mnist()
```
### ii. Select the target variable (labels)

### iii. Prepare the data (e.g., flatten images, normalize pixel values)
### iv. Split the data into training and validation sets
### v. Initialize a logistic regression classifier
### vi. Train (fit) the model on the training data
### vii. Evaluate predictions on unseen (validation/test) data
## Results Analysis
### i. Generate a classification report (precision, recall, F1-score)
### ii. Create a confusion matrix to visualize model performance
### iii. Show correct predictions using example images
### iv. Show misclassified examples and explore potential causes
### v. Optionally, include “corrected” images or improvements based on your insights(e.g., better preprocessing, tuning, or visualization)
## Regularization in Logistic Regression
• Explain the concept of regularization (L1, L2) in logistic regression.
• Describe how regularization helps prevent overfitting.
• Discuss how you applied or could apply regularization in this modeling task and how
it affects model performance on Fashion-MNIST.
## Saving and Using the Trained Model
• Show how you save the trained model for future use
• Explain how the saved model can be used to make predictions on new, unseen data.
• (Optional) Provide a short code snippet showing how to load the model and predict
on a new input.

## References
[^1]:Zalando Research, “Fashion-MNIST,” Kaggle dataset. [Online]. Available: https://www.kaggle.com/datasets/zalando-research/fashionmnist. [Accessed: Sep. 10, 2026].

[^2]: M. Mayer (sunsided), “MNIST,” GitHub repository, containing specifications credited to Y. LeCun and C. Cortes. [Online]. Available: https://github.com/sunsided/mnist. [Accessed: Sep. 10, 2026].