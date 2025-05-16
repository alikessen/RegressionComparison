# RegressionComparison

This project compares the performance of Decision Tree and Linear Regression models using a structured dataset. It includes data visualisation, model training, evaluation metrics, cross-validation, and feature importance analysis using Python and scikit-learn.

## Table of Contents
- [Description](#description)
- [Features](#features)
- [Usage](#usage)
- [Requirements](#requirements)
- [License](#license)

## Description
The goal of this project is to demonstrate a complete machine learning workflow for regression analysis. Two models—Decision Tree Regressor and Linear Regression—are trained and evaluated on the same dataset. The script includes correlation analysis, visualisation, prediction performance comparison (MSE and R²), and 10-fold cross-validation. Feature importance from the Decision Tree is also analysed and visualised.

## Features
- Correlation heatmap and pair plot visualisation
- Train/test split for model evaluation
- Two regression models:
  - Decision Tree Regressor
  - Linear Regression
- Evaluation metrics: Mean Squared Error and R² Score
- 10-fold cross-validation
- Feature importance bar plot (Decision Tree)

## Usage
1. Clone the repository:
   ```bash
   git clone https://github.com/alikessen/RegressionComparison.git
   cd RegressionComparison
   
2. Make sure regression_data.csv is in the same directory.

3. Install required libraries:
   ```bash
   pip install pandas scikit-learn matplotlib seaborn numpy

4. Run the script:
   ```bash
   python regression_comparison.py


