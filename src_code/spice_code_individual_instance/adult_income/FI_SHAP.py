import numpy as np
import pandas as pd
import shap
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# =============================================================================
# LOAD DATA AND PREPARE FOR SHAP ANALYSIS
# =============================================================================

# Load dataset
data = pd.read_csv('/Users/saikat/Desktop/Summer 2025/CFNet_AAAI_26/Adult Income Dataset/processed_adult.csv')

# Get feature names
feature_names = data.drop(columns=['target']).columns.tolist()

# Define features and target
X = data.drop(columns=['target']).values
y = data['target'].values

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=0)

# Further split the training data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.10, random_state=0)

# Standardize features (same as training script)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# =============================================================================
# LOAD TRAINED MODEL
# =============================================================================

model_path = '/Users/saikat/Desktop/Summer 2025/CFNet_AAAI_26/Adult Income Dataset/best_adult_income_gb_model.pkl'
rf_model = joblib.load(model_path)

print("✓ Model loaded successfully!")
print(f"✓ Features: {feature_names}")

# =============================================================================
# SHAP ANALYSIS - FIND MOST IMPORTANT FEATURE
# =============================================================================

print("\nCalculating SHAP values...")

# Create SHAP explainer
explainer = shap.TreeExplainer(rf_model)

# Calculate SHAP values for test set
shap_values = explainer.shap_values(X_test_scaled)

print(f"SHAP values type: {type(shap_values)}")
print(f"SHAP values shape: {shap_values.shape}")

# For binary classification, SHAP values have shape (n_samples, n_features, n_classes)
# We need to select the positive class (class 1 - admission)
if len(shap_values.shape) == 3:
    shap_values_class1 = shap_values[:, :, 1]  # Select positive class
    print(f"Selected positive class SHAP values shape: {shap_values_class1.shape}")
else:
    shap_values_class1 = shap_values

# Calculate mean absolute SHAP values for feature importance
mean_shap_values = np.abs(shap_values_class1).mean(axis=0)
print(f"Mean SHAP values shape: {mean_shap_values.shape}")
print(f"Feature names length: {len(feature_names)}")

# Create feature importance ranking
feature_importance = pd.DataFrame({
    'Feature': feature_names,
    'SHAP_Importance': mean_shap_values
}).sort_values('SHAP_Importance', ascending=False)

# =============================================================================
# RESULTS
# =============================================================================

print("\n" + "="*60)
print("FEATURE IMPORTANCE RANKING (SHAP Analysis)")
print("="*60)

for i, (idx, row) in enumerate(feature_importance.iterrows(), 1):
    print(f"{i:2d}. {row['Feature']:25s} | SHAP Score: {row['SHAP_Importance']:.6f}")

print("\n" + "="*60)
print("MOST IMPORTANT FEATURE")
print("="*60)

most_important_feature = feature_importance.iloc[0]['Feature']
highest_shap_score = feature_importance.iloc[0]['SHAP_Importance']

print(f"Feature Name: {most_important_feature}")
print(f"SHAP Score:   {highest_shap_score:.6f}")

# Additional insights
print(f"\nThis feature is {highest_shap_score/feature_importance.iloc[1]['SHAP_Importance']:.2f}x more important than the 2nd most important feature.")

print("\n" + "="*60)
print("TOP 3 FEATURES SUMMARY")
print("="*60)

for i in range(min(3, len(feature_importance))):
    feature = feature_importance.iloc[i]['Feature']
    score = feature_importance.iloc[i]['SHAP_Importance']
    percentage = (score / mean_shap_values.sum()) * 100
    print(f"{i+1}. {feature}: {score:.6f} ({percentage:.1f}% of total importance)")

print(f"\nAnalysis complete! The most important feature is: {most_important_feature}")