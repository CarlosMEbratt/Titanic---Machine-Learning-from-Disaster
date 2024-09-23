# Titanic Survival Prediction using Machine Learning and Deep Learning

## Project Overview

This project aims to predict the survival of passengers aboard the Titanic using data from the well-known Titanic dataset, which is commonly featured in Kaggle competitions. The goal is to use machine learning and deep learning techniques to achieve high predictive accuracy. By analyzing various features such as passenger class, age, sex, and fare, the model predicts whether a passenger would survive or not.

## Approach

1. **Data Preprocessing**:
   - Handled missing values for key features (e.g., age, embarked).
   - Encoded categorical variables such as `sex`, `embarked`, and `pclass`.
   - Normalized numerical variables like `age` and `fare` to scale the input features.
   - Split the dataset into training and testing sets.

2. **Machine Learning Models**:
   - Initial models were built using classification algorithms like **XGBoost** and **Random Forest**, but these models showed moderate accuracy.

3. **Deep Learning Approach**:
   - A deep neural network (DNN) was implemented to improve predictive performance. This model includes:
     - Multiple fully connected layers (Dense layers) with **ReLU** activation.
     - Regularization techniques such as **L2 regularization** and **Dropout** to prevent overfitting.
   - **Hyperparameter tuning** was applied to optimize the network architecture, including the number of layers, neurons, and dropout rates.
   
4. **Model Evaluation**:
   - The deep learning model reached an accuracy of around **90%** on the training set.
   - However, further refinement is ongoing to improve the generalization performance on the test data, with efforts to minimize overfitting.

## Key Features

- **Data Preprocessing**: Addressing missing data, feature encoding, and scaling.
- **Machine Learning Models**: Initial models included XGBoost and Random Forest classifiers.
- **Deep Neural Network**: Implemented a DNN with regularization and dropout layers to avoid overfitting.
- **Hyperparameter Tuning**: Fine-tuned the model architecture and regularization to optimize performance.
- **Model Export**: Prepared predictions for submission to Kaggle as a CSV file.

## Technologies Used

- Python (with libraries: `pandas`, `numpy`, `sklearn`, `tensorflow`, `keras`)
- Jupyter Notebook
- TensorFlow and Keras for building deep learning models.
- Scikit-learn for data preprocessing and machine learning.

## Instructions for Use

1. Clone the repository.
2. Install required dependencies: `pip install -r requirements.txt`.
3. Run the Jupyter notebooks or Python scripts to preprocess data and train the model.
4. Use the `submission.csv` file to submit predictions to Kaggle.
