import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
import joblib

# Load dataset
data = pd.read_csv('/Users/saikat/Desktop/Summer 2025/CFNet_AAAI_26/HELOC Dataset/AISTATS HELOC Dataset/heloc_dataset_v1.csv')

# Preprocess the target variable
data['target'] = data['target'].map({'Bad': 0, 'Good': 1})

# Define features (X) and target (y)
X = data.drop(columns=['target']).values
y = data['target'].values

# Split data
split_idx = int(0.85 * len(X))
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]


# Standardize features - This is important for Neural Networks
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Initialize the Multilayer Perceptron (MLP) Classifier
mlp_model = MLPClassifier(
    hidden_layer_sizes=(100, 75), # Two hidden layers with 100 and 50 neurons
    activation='relu',            # ReLU activation function
    solver='adam',                # Adam optimizer
    max_iter=500,                 # Maximum number of iterations
    random_state=0,               # For reproducibility
    early_stopping=True,          # Enable early stopping
    validation_fraction=0.1,      # Proportion of training data to set aside for validation
    n_iter_no_change=10,          # Number of iterations with no improvement to wait before stopping
    tol=1e-4                      # Tolerance for the optimization
)

print("Training Multilayer Perceptron (MLP) model...")

# Train the model
# Note: MLPClassifier uses its own internal validation set for early stopping,
# so we don't need to pass X_val, y_val to the fit method.
mlp_model.fit(X_train, y_train)

# Save the trained model
model_path = 'best_heloc_mlp_model.pkl'
joblib.dump(mlp_model, model_path)
print(f"Model saved to: {model_path}")

# Load the model (simulating loading for a separate prediction task)
loaded_model = joblib.load(model_path)

# Make predictions on the test set
y_pred_test = loaded_model.predict(X_test)
y_pred_proba_test = loaded_model.predict_proba(X_test)[:, 1]  # Probabilities for the positive class

# Evaluate model performance
accuracy_test = accuracy_score(y_test, y_pred_test)
conf_matrix_test = confusion_matrix(y_test, y_pred_test)
class_report_test = classification_report(y_test, y_pred_test)

print(f'\nAccuracy on Test Set: {accuracy_test:.4f}')
print('\nConfusion Matrix on Test Set:')
print(conf_matrix_test)
print('\nClassification Report on Test Set:')
print(class_report_test)




