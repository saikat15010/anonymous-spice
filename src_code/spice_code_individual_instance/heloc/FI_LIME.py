import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
import joblib
import lime
import lime.lime_tabular
import warnings
warnings.filterwarnings('ignore')

# Load dataset
data = pd.read_csv('/Users/saikat/Desktop/Summer 2025/CFNet_AAAI_26/HELOC Dataset/heloc_dataset_v1.csv')

# Preprocess the target variable
data['target'] = data['target'].map({'Bad': 0, 'Good': 1})

# Get feature names for LIME analysis
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

# ====== LIME Analysis ======
print("\n" + "="*60)
print("LIME FEATURE IMPORTANCE ANALYSIS")
print("="*60)

# Initialize LIME explainer
print("Initializing LIME explainer...")
lime_explainer = lime.lime_tabular.LimeTabularExplainer(
    X_train,
    feature_names=feature_names,
    class_names=['Bad', 'Good'],
    mode='classification',
    discretize_continuous=True,
    random_state=0
)

# Function to get prediction probabilities for LIME
def lime_predict_proba(X):
    """Wrapper function for LIME prediction"""
    return loaded_model.predict_proba(X)

# Analyze multiple instances and collect feature importance
print("Calculating LIME explanations for multiple instances...")
n_instances = 100  # Number of test instances to analyze
lime_importance_scores = {}

# Initialize dictionary to store importance scores for each feature
for feature in feature_names:
    lime_importance_scores[feature] = []

# Generate LIME explanations for multiple instances
for i in range(min(n_instances, len(X_test))):
    if (i + 1) % 20 == 0:
        print(f"Processing instance {i + 1}/{n_instances}...")
    
    # Get LIME explanation for instance i
    lime_explanation = lime_explainer.explain_instance(
        X_test[i], 
        lime_predict_proba, 
        num_features=len(feature_names),
        num_samples=1000  # Number of samples for LIME
    )
    
    # Extract feature importance scores
    lime_features = lime_explanation.as_list()
    
    # Create a dictionary for this instance
    instance_scores = {}
    for feature_desc, importance in lime_features:
        # Parse feature name from LIME description
        # LIME returns descriptions like "feature_name <= value" or "feature_name > value"
        feature_name = feature_desc.split(' ')[0]
        if feature_name in feature_names:
            instance_scores[feature_name] = abs(importance)
    
    # Add scores to the collection
    for feature in feature_names:
        if feature in instance_scores:
            lime_importance_scores[feature].append(instance_scores[feature])
        else:
            lime_importance_scores[feature].append(0.0)

# Calculate average importance scores across all instances
lime_avg_importance = {}
lime_std_importance = {}
lime_max_importance = {}
lime_min_importance = {}

for feature in feature_names:
    scores = lime_importance_scores[feature]
    lime_avg_importance[feature] = np.mean(scores)
    lime_std_importance[feature] = np.std(scores)
    lime_max_importance[feature] = np.max(scores)
    lime_min_importance[feature] = np.min(scores)

# Create DataFrame for LIME results
lime_importance_df = pd.DataFrame({
    'Feature': feature_names,
    'LIME_Avg_Importance': [lime_avg_importance[f] for f in feature_names],
    'LIME_Std_Importance': [lime_std_importance[f] for f in feature_names],
    'LIME_Max_Importance': [lime_max_importance[f] for f in feature_names],
    'LIME_Min_Importance': [lime_min_importance[f] for f in feature_names]
})

# Sort by average importance
lime_importance_df = lime_importance_df.sort_values('LIME_Avg_Importance', ascending=False)

print(f"\nLIME Feature Importance Ranking (Average over {n_instances} instances):")
print("-" * 85)
print(f"{'Rank':<4} {'Feature':<25} {'Avg Score':<12} {'Std Dev':<12} {'Max Score':<12} {'Min Score':<12}")
print("-" * 85)

for i, (_, row) in enumerate(lime_importance_df.iterrows(), 1):
    print(f"{i:2d}. {row['Feature']:<25} {row['LIME_Avg_Importance']:<12.6f} {row['LIME_Std_Importance']:<12.6f} {row['LIME_Max_Importance']:<12.6f} {row['LIME_Min_Importance']:<12.6f}")

print(f"\nTop 10 Most Important Features (LIME):")
print("-" * 85)
print(f"{'Rank':<4} {'Feature':<25} {'Avg Score':<12} {'Std Dev':<12} {'Max Score':<12} {'Min Score':<12}")
print("-" * 85)

top_10_lime = lime_importance_df.head(10)
for i, (_, row) in enumerate(top_10_lime.iterrows(), 1):
    print(f"{i:2d}. {row['Feature']:<25} {row['LIME_Avg_Importance']:<12.6f} {row['LIME_Std_Importance']:<12.6f} {row['LIME_Max_Importance']:<12.6f} {row['LIME_Min_Importance']:<12.6f}")

# Calculate LIME analysis statistics
lime_scores = lime_importance_df['LIME_Avg_Importance'].values
print(f"\nLIME Analysis Summary:")
print("-" * 40)
print(f"Total number of features analyzed: {len(feature_names)}")
print(f"Number of instances analyzed: {n_instances}")
print(f"Mean LIME importance: {lime_scores.mean():.6f}")
print(f"Std LIME importance: {lime_scores.std():.6f}")
print(f"Max LIME importance: {lime_scores.max():.6f}")
print(f"Min LIME importance: {lime_scores.min():.6f}")

# Feature contribution analysis for LIME
total_lime_importance = lime_scores.sum()
top_5_lime_contribution = lime_importance_df.head(5)['LIME_Avg_Importance'].sum()
top_10_lime_contribution = lime_importance_df.head(10)['LIME_Avg_Importance'].sum()

print(f"\nLIME Feature Contribution Analysis:")
print(f"Top 5 features contribute: {(top_5_lime_contribution/total_lime_importance)*100:.2f}% of total importance")
print(f"Top 10 features contribute: {(top_10_lime_contribution/total_lime_importance)*100:.2f}% of total importance")

# Detailed individual instance explanations
print(f"\nDetailed Individual Instance Explanations:")
print("-" * 60)

# Analyze first 5 instances in detail
for instance_idx in range(min(5, len(X_test))):
    lime_exp = lime_explainer.explain_instance(
        X_test[instance_idx], 
        lime_predict_proba, 
        num_features=10  # Show top 10 features
    )
    
    # Get prediction for this instance
    prediction = loaded_model.predict([X_test[instance_idx]])[0]
    probability = loaded_model.predict_proba([X_test[instance_idx]])[0]
    actual_label = y_test[instance_idx]
    
    print(f"\nInstance {instance_idx + 1}:")
    print(f"Actual Label: {'Good' if actual_label == 1 else 'Bad'}")
    print(f"Predicted Label: {'Good' if prediction == 1 else 'Bad'}")
    print(f"Prediction Probability: {probability[1]:.4f} (Good), {probability[0]:.4f} (Bad)")
    print(f"Correct Prediction: {'Yes' if prediction == actual_label else 'No'}")
    
    # Get feature explanations
    feature_explanations = lime_exp.as_list()
    print(f"Top 10 Feature Contributions:")
    for j, (feature_desc, importance) in enumerate(feature_explanations[:10], 1):
        direction = "towards Good" if importance > 0 else "towards Bad"
        print(f"  {j:2d}. {feature_desc:<35} | Score: {importance:8.4f} ({direction})")

# Calculate feature consistency across instances
print(f"\nFeature Consistency Analysis:")
print("-" * 50)
print("Features with highest consistency (low std deviation relative to mean):")

# Calculate coefficient of variation (std/mean) for non-zero means
cv_scores = []
for _, row in lime_importance_df.iterrows():
    if row['LIME_Avg_Importance'] > 0:
        cv = row['LIME_Std_Importance'] / row['LIME_Avg_Importance']
        cv_scores.append((row['Feature'], cv, row['LIME_Avg_Importance']))

# Sort by coefficient of variation (ascending = more consistent)
cv_scores.sort(key=lambda x: x[1])

print(f"Most Consistent Features (Top 10):")
for i, (feature, cv, avg_importance) in enumerate(cv_scores[:10], 1):
    print(f"{i:2d}. {feature:<25} | CV: {cv:.4f} | Avg Importance: {avg_importance:.6f}")

print(f"\nLeast Consistent Features (Top 10):")
for i, (feature, cv, avg_importance) in enumerate(cv_scores[-10:], 1):
    print(f"{i:2d}. {feature:<25} | CV: {cv:.4f} | Avg Importance: {avg_importance:.6f}")

# Save LIME results to CSV
lime_importance_df.to_csv('lime_feature_importance.csv', index=False)
print(f"\nLIME feature importance rankings saved to: lime_feature_importance.csv")

# Create a summary report
summary_report = {
    'Total_Features': len(feature_names),
    'Instances_Analyzed': n_instances,
    'Mean_Importance': lime_scores.mean(),
    'Std_Importance': lime_scores.std(),
    'Max_Importance': lime_scores.max(),
    'Min_Importance': lime_scores.min(),
    'Top_5_Contribution_Percent': (top_5_lime_contribution/total_lime_importance)*100,
    'Top_10_Contribution_Percent': (top_10_lime_contribution/total_lime_importance)*100
}

# Save summary report
import json
with open('lime_analysis_summary.json', 'w') as f:
    json.dump(summary_report, f, indent=2)

print(f"LIME analysis summary saved to: lime_analysis_summary.json")

print("\n" + "="*60)
print("LIME ANALYSIS COMPLETE")
print("="*60)