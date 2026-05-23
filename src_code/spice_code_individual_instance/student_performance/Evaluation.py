import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import joblib

# Load dataset
data1 = pd.read_csv('/Users/saikat/Desktop/Summer 2025/CFNet_AAAI_26/Student Performance Datastet/AISTATS Student Performance/student_train.csv')
data2 = pd.read_csv('/Users/saikat/Desktop/Summer 2025/CFNet_AAAI_26/Student Performance Datastet/AISTATS Student Performance/student_test.csv')

# Combine datasets
data = pd.concat([data1, data2], ignore_index=True)

# Display the combined dataset
print(data.head())
print(data.shape)

# Check the distribution of the target feature
target_distribution = data['target'].value_counts()

# Display the distribution
print(target_distribution)

# Label encoding the binary categorical features
binary_features = ['famsup', 'higher', 'internet', 'romantic']

for feature in binary_features:
    data[feature] = data[feature].map({'yes': 1, 'no': 0})

print(data.head())

# Define features and target
X = data.drop(columns=['target']).values
y = data['target'].values

# Split data
split_idx = int(0.85 * len(X))
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

# Standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train Logistic Regression model
lr_model = LogisticRegression(max_iter=1000, random_state=0)
lr_model.fit(X_train, y_train)

# Save the trained model
model_path = '/Users/saikat/Desktop/Summer 2025/CFNet_AAAI_26/Student Performance Datastet/AISTATS Student Performance/best_admission_lr_model.pkl'
joblib.dump(lr_model, model_path)

# Load the best model
lr_model = joblib.load(model_path)

# Evaluate model performance on the test set
y_pred_test = lr_model.predict(X_test)
accuracy_test = accuracy_score(y_test, y_pred_test)
conf_matrix_test = confusion_matrix(y_test, y_pred_test)
class_report_test = classification_report(y_test, y_pred_test)

print(f'Accuracy on Test Set: {accuracy_test:.2f}')
print('Confusion Matrix:')
print(conf_matrix_test)
print('Classification Report:')
print(class_report_test)
