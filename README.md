# Competition-Project-African-Credit-Scoring-Challenge-Predicting-Loan-Default-with-Deep-Learning
[Verified on Zindi](https://zindi.africa/users/Md_Farman_Ali/competitions/certificate)

This repository presents a complete machine learning pipeline to predict loan default using customer financial data in the context of an African credit scoring challenge. The goal is to build a binary classifier that can predict whether a customer is likely to default on a loan, with a strong emphasis on precision and F1-score.

## Project Objectives

- Build a robust model to predict credit risk (default vs. non-default)
- Engineer high-quality interaction features from raw data
- Apply deep learning techniques on structured tabular data
- Tune the prediction threshold for optimal F1 performance

## Dataset

- **Train.csv** – labeled data with customer features and target variable
- **Test.csv** – unlabeled data to be used for prediction
- **Target** – binary indicator (`1` = will default, `0` = will not default)

## Workflow Overview

### 1. Data Preprocessing
- Loaded and inspected `Train.csv` and `Test.csv`
- Verified data types and absence of missing values
- Separated features and target labels

### 2. Feature Engineering
- Generated interaction features using `PolynomialFeatures` (degree 2, interaction-only)
- Selected top 30 most important features using `SelectKBest` with ANOVA F-statistic

### 3. Handling Class Imbalance
- Calculated class weights to address imbalance in the training labels
- Used `compute_class_weight` to feed into model training

### 4. Model Architecture (Deep Neural Network)
Constructed a deep feedforward neural network using `TensorFlow/Keras`:

- Input layer with 30 selected features
- Dense layers: [1024, 512, 256, 128, 64, 128]
- Regularization: Batch Normalization, Dropout, L2 penalties
- Output layer: Sigmoid activation for binary classification
- Optimizer: Adam (learning rate = 0.001)
- Loss function: Binary Crossentropy

### 5. Threshold Optimization
- Generated predicted probabilities on the validation set
- Tuned prediction threshold using `precision_recall_curve` and F1-score maximization
- Selected the best threshold dynamically for final classification

### 6. Evaluation
- Calculated confusion matrix, F1-score, and accuracy on validation set
- Visualized performance with a confusion matrix heatmap
- Printed class distribution and feature summary

### 7. Prediction & Submission
- Aligned test data features with training structure
- Applied preprocessing: polynomial features and feature selection
- Made predictions using the optimized threshold
- Exported predictions as `predictions.csv` for submission

## Future Improvements

1. **Model Optimization**
   - Use of ensemble models like LightGBM or XGBoost
   - Apply hyperparameter tuning with cross-validation
2. **Better Imbalance Handling**
   - Try SMOTE or ADASYN for synthetic minority oversampling
   - Combine class weights with resampling techniques
3. **Explainability**
   - Add SHAP/LIME for model interpretation
   - Identify key drivers of default
4. **Deployment**
   - Deploy the model using Flask, FastAPI, or Streamlit
   - Wrap the pipeline into a user-friendly interface


