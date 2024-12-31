# Apartment for Rent Classified - Predicting Rent Prices & Classifying Amenities

This project uses the "Apartment for Rent Classified" dataset from the UCI Machine Learning Repository. The dataset contains information about rental apartments, including various attributes that describe the apartment's features. The goal is to predict apartment prices based on the provided attributes and classify amenities using a machine learning approach.

## Dataset Overview

The "Apartment for Rent Classified" dataset contains **10,000 rows** and **22 columns**. The columns are as follows:

- **Id**: Integer
- **Category**: Categorical
- **Title**: Categorical
- **Body**: Categorical
- **Amenities**: Categorical
- **Bathrooms**: Float
- **Bedrooms**: Float
- **Currency**: Categorical
- **Fee**: Categorical
- **Has_photo**: Categorical
- **Pet_allowed**: Categorical
- **Price**: Integer (Target variable for regression)
- **Price_type**: Categorical
- **Price_display**: Categorical
- **Square_feet**: Integer
- **Address**: Categorical
- **Cityname**: Categorical
- **Latitude**: Float
- **Longitude**: Float
- **Source**: Categorical
- **Time**: Integer

### Objective

The project has two main tasks:

1. **Predict Rent Prices**: Predict the price of the apartment based on various features such as the number of bathrooms, bedrooms, amenities, and location (latitude/longitude).
   
2. **Classify Amenities**: Classify the amenities of the apartment (such as gym, parking, etc.) using attributes such as price, number of rooms, size, and location.

## Approach

### 1. Regression for Price Prediction

We use multiple machine learning algorithms for **regression** to predict the price of the apartments:

- **Linear Regression**: A basic model that assumes a linear relationship between the independent variables and the target price.
- **Random Forest Regression**: A tree-based method that creates multiple decision trees and averages their predictions to improve accuracy.
- **Decision Tree Regression**: A model that splits the data into subsets based on the input features to make predictions.
- **Gradient Boosting Regression**: An ensemble method that builds trees sequentially, where each tree corrects errors made by the previous one.

### 2. Classification for Amenities

We use multiple algorithms for **classification** to predict the amenities of the apartment:

- **Logistic Regression**: A linear model used for binary classification of amenities (e.g., gym: yes/no).
- **Decision Tree Classifier**: A tree-based model used to classify amenities based on input features.
- **K-Nearest Neighbors (KNN)**: A non-parametric method that classifies amenities based on the closest data points in the feature space.

### 3. Addressing Overfitting

Overfitting is a common issue in machine learning, especially when using complex models. To address overfitting, the following techniques are applied:

- **Hyperparameter Tuning**: The performance of the models is optimized by adjusting the hyperparameters (e.g., the depth of trees, the number of trees in Random Forest, learning rate in Gradient Boosting, etc.) using techniques like Grid Search and Randomized Search.
  
- **Feature Selection**: To reduce the risk of overfitting and improve model performance, unnecessary or irrelevant features are removed. This is done using methods like:
  - **Correlation Matrix**: Identifying and removing highly correlated features.
  - **Recursive Feature Elimination (RFE)**: Iteratively removing less important features based on model performance.

- **Cross-Validation**: Cross-validation techniques are used to ensure that the models generalize well to unseen data by splitting the data into multiple subsets for training and testing.

