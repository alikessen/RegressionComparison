import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score
import numpy as np

#DATA IMPORT

# Importing the data
data = pd.read_csv("regression_data.csv")

# Creating DataFrame
df = pd.DataFrame(data)



#DATA VISUALIZATION

# Correlation Heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt='.2f')
plt.title("Correlation Heatmap")
plt.show()

# Pair Plot
sns.pairplot(df)
plt.title("Pair Plot (Scatter Matrix)")
plt.show()



#SPLIT THE DATA INTO TRAINING AND TESTING SETS

X = df.drop('y', axis=1)  # Drop the 'y' column to use as input
y = df['y']  # The target variable

# 80% for training and 20% for testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Print the shapes of the resulting datasets
print("—————————————————————————————————————————————————————————————————————")
print(f"Training set shape: X_train: {X_train.shape}, y_train: {y_train.shape}")
print(f"Testing set shape: X_test: {X_test.shape}, y_test: {y_test.shape}")
print("—————————————————————————————————————————————————————————————————————")



#BUILDING MODELS AND GENERATING PREDICTIONS

# Decision Tree Regressor
dt_model = DecisionTreeRegressor(random_state=42)
dt_model.fit(X_train, y_train)

# Linear Regression
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

# Predictions on the testing set
y_pred_dt = dt_model.predict(X_test)
y_pred_lr = lr_model.predict(X_test)

# Evaluate models
mse_dt = mean_squared_error(y_test, y_pred_dt)
mse_lr = mean_squared_error(y_test, y_pred_lr)
r2_dt = r2_score(y_test, y_pred_dt)
r2_lr = r2_score(y_test, y_pred_lr)

# Print evaluation results
print(f"Decision Tree MSE: {mse_dt}, R2 Score: {r2_dt}")
print(f"Linear Regression MSE: {mse_lr}, R2 Score: {r2_lr}")
print("—————————————————————————————————————————————————————————————————————")



#ADDITIONALS

# Visualizing Decision Tree predictions vs actual values
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred_dt, color='blue', label='Decision Tree Predictions')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--', label='Ideal Fit')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Decision Tree Predictions vs Actual')
plt.legend()
plt.show()

# Visualizing Linear Regression predictions vs actual values
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred_lr, color='green', label='Linear Regression Predictions')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--', label='Ideal Fit')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Linear Regression Predictions vs Actual')
plt.legend()
plt.show()

#Cross-Validation

# Use cross-validation to validate the model performance across multiple data splits

# Cross-validation for Decision Tree
cv_scores_dt = cross_val_score(dt_model, X, y, cv=10)
print(f"Decision Tree Cross-Validation Scores: {cv_scores_dt}")
print(f"Mean Cross-Validation Score for Decision Tree: {cv_scores_dt.mean()}")

# Cross-validation for Linear Regression
cv_scores_lr = cross_val_score(lr_model, X, y, cv=10)
print(f"Linear Regression Cross-Validation Scores: {cv_scores_lr}")
print(f"Mean Cross-Validation Score for Linear Regression: {cv_scores_lr.mean()}")
print("———————————————————————————————————————————————————————————————————")



# Feature importance for the Decision Tree model

# Calculating feature importances to understand which input features contribute the most to predictions

# Retrieve feature importances from the Decision Tree model
feature_importances = dt_model.feature_importances_

# Create a DataFrame 
importance_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': feature_importances
})

# Sort the DataFrame by importance in descending order
importance_df = importance_df.sort_values(by='Importance', ascending=False)

# Display feature importance values
print("Feature Importances for Decision Tree Model:")
print(importance_df)

# Plot feature importance
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=importance_df, palette="viridis")
plt.title("Feature Importance in Decision Tree Model")
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.show()

