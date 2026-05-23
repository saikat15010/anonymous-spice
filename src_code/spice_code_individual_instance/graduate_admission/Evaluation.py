import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load dataset
data1 = pd.read_csv('/Users/saikat/Desktop/Summer 2025/CFNet_AAAI_26/Graduate Admission Dataset/AISTATS Graduate Admission/admission_train.csv')
data2 = pd.read_csv('/Users/saikat/Desktop/Summer 2025/CFNet_AAAI_26/Graduate Admission Dataset/AISTATS Graduate Admission/admission_test.csv')

# Combine datasets
data = pd.concat([data1, data2], ignore_index=True)

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

# Train Random Forest Classifier
rf_model = RandomForestClassifier(n_estimators=100, random_state=0)
rf_model.fit(X_train, y_train)

# Save the trained model
model_path = '/Users/saikat/Desktop/Summer 2025/CFNet_AAAI_26/Graduate Admission Dataset/AISTATS Graduate Admission/best_admission_rf_model.pkl'
joblib.dump(rf_model, model_path)

# Load the best model
rf_model = joblib.load(model_path)

# Evaluate model performance on the test set
y_pred_test = rf_model.predict(X_test)
accuracy_test = accuracy_score(y_test, y_pred_test)
conf_matrix_test = confusion_matrix(y_test, y_pred_test)
class_report_test = classification_report(y_test, y_pred_test)

print(f'Accuracy on Test Set: {accuracy_test:.2f}')
print('Confusion Matrix:')
print(conf_matrix_test)
print('Classification Report:')
print(class_report_test)

