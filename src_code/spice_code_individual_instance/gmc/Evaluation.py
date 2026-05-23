import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
import joblib
from xgboost import XGBClassifier


combined_df = pd.read_csv('/Users/saikat/Desktop/Summer 2025/CFNet_AAAI_26/GMC Dataset/AISTATS GMC Dataset/preprocessed_credit_data.csv')

# Check for missing values in target variable
print(f"Missing values in target variable: {combined_df['SeriousDlqin2yrs'].isnull().sum()}")
print(f"Unique values in target: {combined_df['SeriousDlqin2yrs'].unique()}")

# Remove rows with missing target values
print(f"Original dataset shape: {combined_df.shape}")
combined_df_clean = combined_df.dropna(subset=['SeriousDlqin2yrs'])
print(f"Dataset shape after removing missing targets: {combined_df_clean.shape}")

# Define features and target (following your exact format)
X = combined_df_clean.drop(columns=['SeriousDlqin2yrs', 'ID']).values
y = combined_df_clean['SeriousDlqin2yrs'].values

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=0)

# Further split the training data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.10, random_state=0)

# Standardize features (optional for Gradient Boosting, but keeping for consistency)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

# Initialize Gradient Boosting Classifier
gb_model = XGBClassifier(n_estimators=50, random_state=0)


print("Training XGBoost model...")

# Train the model
gb_model.fit(X_train, y_train)

# Save the trained model
model_path = '/Users/saikat/Desktop/Summer 2025/CFNet_AAAI_26/GMC Dataset/AISTATS GMC Dataset/best_credit_xgb_model.pkl'
joblib.dump(gb_model, model_path)
print(f"Model saved to: {model_path}")

# Load the model (simulating loading the best model)
loaded_model = joblib.load(model_path)

# Make predictions on test set
y_pred_test = loaded_model.predict(X_test)
y_pred_proba_test = loaded_model.predict_proba(X_test)[:, 1]  # Probabilities for positive class

# Evaluate model performance
accuracy_test = accuracy_score(y_test, y_pred_test)
conf_matrix_test = confusion_matrix(y_test, y_pred_test)
class_report_test = classification_report(y_test, y_pred_test)

print(f'\nAccuracy on Test Set: {accuracy_test:.2f}')
print('Confusion Matrix:')
print(conf_matrix_test)
print('Classification Report:')
print(class_report_test)
