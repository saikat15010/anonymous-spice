import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
import joblib

# Load dataset
data = pd.read_csv('/Users/saikat/Desktop/Summer 2025/CFNet_AAAI_26/Adult Income Dataset/AISTATS Adult Income /processed_adult.csv')

# Define features and target
X = data.drop(columns=['target']).values
y = data['target'].values

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
gb_model = GradientBoostingClassifier(
    n_estimators=100,           # Number of trees
    max_depth=6,                # Maximum depth of trees
    learning_rate=0.1,          # Learning rate
    subsample=0.8,              # Subsample ratio
    random_state=0,             # For reproducibility
    validation_fraction=0.1,    # Fraction of training data for early stopping
    n_iter_no_change=10,        # Early stopping rounds
    tol=1e-4                    # Tolerance for early stopping
)

print("Training Gradient Boosting model...")

# Train the model
gb_model.fit(X_train, y_train)

# Save the trained model
model_path = '/Users/saikat/Desktop/Summer 2025/CFNet_AAAI_26/Adult Income Dataset/AISTATS Adult Income /best_adult_income_gb_model.pkl'
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
