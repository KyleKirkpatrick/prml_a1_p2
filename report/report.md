## Problem Introduction
This project implements a solution to train a machine-learning model to label images depicting various articles of clothing. It uses the publically accessible Fashion-MNIST dataset for training and testing. The solution is adapted from previous code examples using the MNIST database of handwritten digits, restructured to import the fashion database instead. 

By implementing this solution, I aim to explore the following questions:
- How is the Fashion-MNIST dataset structured? 
- How is data curated for ML training?
- How is logistic regression implemented in training a ML model?

## Dataset Description
- Describe the structure of the dataset: What do the rows and columns represent? What are the labels/classes?
- Highlight any important characteristics of the dataset that may influence your modeling.

## Justification for Using Logistic Regression
• Explain why logistic regression is an appropriate model for this classification task.
• Discuss its strengths and limitations in the context of Fashion-MNIST.
## Data Retrieval
• Show how you loaded the Fashion-MNIST dataset into your program.
• Include any code snippets for downloading or loading the data.
## Data Exploration
Demonstrate understanding and inspection of the dataset.
i. Show example images and labels
ii. Display the pixel matrix for an image
iii. Describe the data format, ranges, and any preprocessing needs
iv. Note any important patterns, class distribution, or anomalies
## Building the Logistic Regression Model
For each step, show and explain your code:
i. Load required Python libraries/packages
ii. Select the target variable (labels)
iii. Prepare the data (e.g., flatten images, normalize pixel values)
iv. Split the data into training and validation sets
v. Initialize a logistic regression classifier
vi. Train (fit) the model on the training data
vii. Evaluate predictions on unseen (validation/test) data
## Results Analysis
i. Generate a classification report (precision, recall, F1-score)
ii. Create a confusion matrix to visualize model performance
iii. Show correct predictions using example images
iv. Show misclassified examples and explore potential causes
v. Optionally, include “corrected” images or improvements based on your insights
(e.g., better preprocessing, tuning, or visualization)
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