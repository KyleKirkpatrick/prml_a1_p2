# University of Canberra

## Faculty of Science and Technology

- 11482 Pattern Recognition and Machine Learning
- 11512 Pattern Recognition and Machine Learning PG

# Assignment 1 – Part B: Logistic Regression on Fashion-MNIST

In this task, you will apply logistic regression to the Fashion-MNIST dataset, following the same process used for the MNIST dataset in Tutorial Weeks 3-4. For more details on logistic regression, refer to Lecture Week 3.

## Instructions

1. Study the provided code, data, and documentation from Week 3, where logistic regression was applied to the MNIST dataset.
2. Download the Fashion-MNIST dataset from:

   <https://www.kaggle.com/datasets/zalando-research/fashionmnist>

3. Adapt the MNIST logistic regression code to work with the Fashion-MNIST dataset.
4. Run your model and analyze the results.
5. Prepare a report that includes:
   - A clear explanation of the steps taken
   - Your understanding of the learning process
   - Analysis and discussion of the results

## Submission Options

You need to submit:

- A PDF report, include answers to all required questions, results, visualisations and discussions (max 5 pages).
- The corresponding Python Code, with clear comments and documentation.

Clear and concise academic writing is essential.

## Report Structure (address all points within 5 pages)

Your report must cover the following sections and address the associated rubric criteria.

### 1. Problem Introduction

- Clearly describe the problem you are solving using the Fashion-MNIST dataset.
- Identify and explain the key questions you aim to answer.

### 2. Dataset Description

- Describe the structure of the dataset: What do the rows and columns represent? What are the labels/classes?
- Highlight any important characteristics of the dataset that may influence your modeling.

### 3. Justification for Using Logistic Regression

- Explain why logistic regression is an appropriate model for this classification task.
- Discuss its strengths and limitations in the context of Fashion-MNIST.

### 4. Data Retrieval

- Show how you loaded the Fashion-MNIST dataset into your program.
- Include any code snippets for downloading or loading the data.

### 5. Data Exploration

Demonstrate understanding and inspection of the dataset.

i. Show example images and labels  
ii. Display the pixel matrix for an image  
iii. Describe the data format, ranges, and any preprocessing needs  
iv. Note any important patterns, class distribution, or anomalies

### 6. Building the Logistic Regression Model

For each step, show and explain your code:

i. Load required Python libraries/packages  
ii. Select the target variable (labels)  
iii. Prepare the data (e.g., flatten images, normalize pixel values)  
iv. Split the data into training and validation sets  
v. Initialize a logistic regression classifier  
vi. Train (fit) the model on the training data  
vii. Evaluate predictions on unseen (validation/test) data

### 7. Results Analysis

i. Generate a classification report (precision, recall, F1-score)  
ii. Create a confusion matrix to visualize model performance  
iii. Show correct predictions using example images  
iv. Show misclassified examples and explore potential causes  
v. Optionally, include “corrected” images or improvements based on your insights  
(e.g., better preprocessing, tuning, or visualization)

### 8. Regularization in Logistic Regression

- Explain the concept of regularization (L1, L2) in logistic regression.
- Describe how regularization helps prevent overfitting.
- Discuss how you applied or could apply regularization in this modeling task and how it affects model performance on Fashion-MNIST.

### 9. Saving and Using the Trained Model

- Show how you save the trained model for future use.
- Explain how the saved model can be used to make predictions on new, unseen data.
- (Optional) Provide a short code snippet showing how to load the model and predict on a new input.

## Tips for Success

- Use concise explanations, clean visuals, and well-commented code.
- Be selective - only include key plots, results, and images.
- Make sure your work is logically structured and easy to follow.
