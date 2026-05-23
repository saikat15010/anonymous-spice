import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
import joblib
import shap
import warnings
warnings.filterwarnings('ignore')

# Load dataset
data = pd.read_csv('/Users/saikat/Desktop/Summer 2025/CFNet_AAAI_26/HELOC Dataset/heloc_dataset_v1.csv')

# Preprocess the target variable
data['target'] = data['target'].map({'Bad': 0, 'Good': 1})

# Get feature names for SHAP analysis
feature_names = data.drop(columns=['target']).columns.tolist()

# Define features (X) and target (y)
X = data.drop(columns=['target']).values
y = data['target'].values

# Split data into training, validation, and test sets
# 75% training, 10% validation, 15% testing
X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=0.15, random_state=0)
X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=0.10, random_state=0)

# Standardize features - This is important for Neural Networks
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

# Initialize the Multilayer Perceptron (MLP) Classifier
mlp_model = MLPClassifier(
    hidden_layer_sizes=(100, 50), # Two hidden layers with 100 and 50 neurons
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

# ====== SHAP Analysis ======
print("\n" + "="*60)
print("SHAP FEATURE IMPORTANCE ANALYSIS")
print("="*60)

# Create a function that returns predictions for SHAP
def model_predict(X):
    """Wrapper function for model prediction"""
    return loaded_model.predict_proba(X)[:, 1]

# Use a subset of training data as background for SHAP
# Using 100 samples for computational efficiency
background_sample = X_train[:100]

# Initialize SHAP explainer
print("Initializing SHAP explainer...")
explainer = shap.KernelExplainer(model_predict, background_sample)

# Calculate SHAP values for test set (using subset for efficiency)
# Using 100 test samples for demonstration
test_sample = X_test[:100]
print("Calculating SHAP values...")
shap_values = explainer.shap_values(test_sample)

# Calculate feature importance scores
feature_importance = np.abs(shap_values).mean(axis=0)

# Create DataFrame for better visualization
importance_df = pd.DataFrame({
    'Feature': feature_names,
    'SHAP_Importance': feature_importance
})

# Sort by importance
importance_df = importance_df.sort_values('SHAP_Importance', ascending=False)

print("\nFeature Importance Ranking (SHAP Values):")
print("-" * 50)
for i, (_, row) in enumerate(importance_df.iterrows(), 1):
    print(f"{i:2d}. {row['Feature']:<25} | Score: {row['SHAP_Importance']:.6f}")

print(f"\nTop 10 Most Important Features:")
print("-" * 40)
top_10 = importance_df.head(10)
for i, (_, row) in enumerate(top_10.iterrows(), 1):
    print(f"{i:2d}. {row['Feature']:<25} | Score: {row['SHAP_Importance']:.6f}")


# Additional SHAP visualizations
print("\nGenerating additional SHAP visualizations...")

# Calculate and display summary statistics
print("\nSHAP Analysis Summary:")
print("-" * 30)
print(f"Total number of features: {len(feature_names)}")
print(f"Mean SHAP importance: {feature_importance.mean():.6f}")
print(f"Std SHAP importance: {feature_importance.std():.6f}")
print(f"Max SHAP importance: {feature_importance.max():.6f}")
print(f"Min SHAP importance: {feature_importance.min():.6f}")

# Calculate percentage contribution of top features
total_importance = feature_importance.sum()
top_5_contribution = importance_df.head(5)['SHAP_Importance'].sum()
top_10_contribution = importance_df.head(10)['SHAP_Importance'].sum()

print(f"\nFeature Contribution Analysis:")
print(f"Top 5 features contribute: {(top_5_contribution/total_importance)*100:.2f}% of total importance")
print(f"Top 10 features contribute: {(top_10_contribution/total_importance)*100:.2f}% of total importance")

# Save results to CSV
importance_df.to_csv('shap_feature_importance.csv', index=False)
print(f"\nFeature importance rankings saved to: shap_feature_importance.csv")

print("\n" + "="*60)
print("SHAP ANALYSIS COMPLETE")
print("="*60)