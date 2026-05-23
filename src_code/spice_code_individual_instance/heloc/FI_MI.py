import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import KBinsDiscretizer
import joblib
import lime
import lime.lime_tabular
import warnings
import json
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, spearmanr
warnings.filterwarnings('ignore')

# Load dataset
data = pd.read_csv('/Users/saikat/Desktop/Summer 2025/CFNet_AAAI_26/HELOC Dataset/heloc_dataset_v1.csv')

# Preprocess the target variable
data['target'] = data['target'].map({'Bad': 0, 'Good': 1})

# Get feature names for analysis
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
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

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
mlp_model.fit(X_train_scaled, y_train)

# Save the trained model
model_path = 'best_heloc_mlp_model.pkl'
joblib.dump(mlp_model, model_path)
print(f"Model saved to: {model_path}")

# Load the model (simulating loading for a separate prediction task)
loaded_model = joblib.load(model_path)

# Make predictions on the test set
y_pred_test = loaded_model.predict(X_test_scaled)
y_pred_proba_test = loaded_model.predict_proba(X_test_scaled)[:, 1]  # Probabilities for the positive class

# Evaluate model performance
accuracy_test = accuracy_score(y_test, y_pred_test)
conf_matrix_test = confusion_matrix(y_test, y_pred_test)
class_report_test = classification_report(y_test, y_pred_test)

print(f'\nAccuracy on Test Set: {accuracy_test:.4f}')
print('\nConfusion Matrix on Test Set:')
print(conf_matrix_test)
print('\nClassification Report on Test Set:')
print(class_report_test)

# ====== MUTUAL INFORMATION ANALYSIS ======
print("\n" + "="*60)
print("MUTUAL INFORMATION FEATURE IMPORTANCE ANALYSIS")
print("="*60)

# Calculate mutual information scores
print("Calculating mutual information scores...")

# Method 1: Direct mutual information on original features
mi_scores_original = mutual_info_classif(X_train, y_train, random_state=0)

# Method 2: Mutual information on standardized features
mi_scores_scaled = mutual_info_classif(X_train_scaled, y_train, random_state=0)

# Method 3: Discretized mutual information (for better handling of continuous features)
print("Discretizing features for robust mutual information calculation...")
discretizer = KBinsDiscretizer(n_bins=10, encode='ordinal', strategy='quantile')
X_train_discretized = discretizer.fit_transform(X_train)
mi_scores_discretized = mutual_info_classif(X_train_discretized, y_train, random_state=0)

# Create DataFrame for mutual information results
mi_results_df = pd.DataFrame({
    'Feature': feature_names,
    'MI_Original': mi_scores_original,
    'MI_Scaled': mi_scores_scaled,
    'MI_Discretized': mi_scores_discretized
})

# Sort by discretized MI scores (generally most reliable for continuous features)
mi_results_df = mi_results_df.sort_values('MI_Discretized', ascending=False)

print(f"\nMutual Information Feature Importance Ranking:")
print("-" * 100)
print(f"{'Rank':<4} {'Feature':<25} {'MI Original':<12} {'MI Scaled':<12} {'MI Discretized':<15}")
print("-" * 100)

for i, (_, row) in enumerate(mi_results_df.iterrows(), 1):
    print(f"{i:2d}. {row['Feature']:<25} {row['MI_Original']:<12.6f} {row['MI_Scaled']:<12.6f} {row['MI_Discretized']:<15.6f}")

print(f"\nTop 10 Most Important Features (Mutual Information - Discretized):")
print("-" * 100)
print(f"{'Rank':<4} {'Feature':<25} {'MI Original':<12} {'MI Scaled':<12} {'MI Discretized':<15}")
print("-" * 100)

top_10_mi = mi_results_df.head(10)
for i, (_, row) in enumerate(top_10_mi.iterrows(), 1):
    print(f"{i:2d}. {row['Feature']:<25} {row['MI_Original']:<12.6f} {row['MI_Scaled']:<12.6f} {row['MI_Discretized']:<15.6f}")

# Calculate mutual information statistics
mi_scores = mi_results_df['MI_Discretized'].values
print(f"\nMutual Information Analysis Summary:")
print("-" * 50)
print(f"Total number of features analyzed: {len(feature_names)}")
print(f"Mean MI score: {mi_scores.mean():.6f}")
print(f"Std MI score: {mi_scores.std():.6f}")
print(f"Max MI score: {mi_scores.max():.6f}")
print(f"Min MI score: {mi_scores.min():.6f}")

# Feature contribution analysis for MI
total_mi_importance = mi_scores.sum()
top_5_mi_contribution = mi_results_df.head(5)['MI_Discretized'].sum()
top_10_mi_contribution = mi_results_df.head(10)['MI_Discretized'].sum()

print(f"\nMutual Information Feature Contribution Analysis:")
print(f"Top 5 features contribute: {(top_5_mi_contribution/total_mi_importance)*100:.2f}% of total MI score")
print(f"Top 10 features contribute: {(top_10_mi_contribution/total_mi_importance)*100:.2f}% of total MI score")

# ====== LIME ANALYSIS ======
print("\n" + "="*60)
print("LIME FEATURE IMPORTANCE ANALYSIS")
print("="*60)

# Initialize LIME explainer
print("Initializing LIME explainer...")
lime_explainer = lime.lime_tabular.LimeTabularExplainer(
    X_train_scaled,
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
for i in range(min(n_instances, len(X_test_scaled))):
    if (i + 1) % 20 == 0:
        print(f"Processing instance {i + 1}/{n_instances}...")
    
    # Get LIME explanation for instance i
    lime_explanation = lime_explainer.explain_instance(
        X_test_scaled[i], 
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
for feature in feature_names:
    scores = lime_importance_scores[feature]
    lime_avg_importance[feature] = np.mean(scores)

# Create DataFrame for LIME results
lime_importance_df = pd.DataFrame({
    'Feature': feature_names,
    'LIME_Avg_Importance': [lime_avg_importance[f] for f in feature_names]
})

# Sort by average importance
lime_importance_df = lime_importance_df.sort_values('LIME_Avg_Importance', ascending=False)

print(f"\nLIME Feature Importance Ranking (Average over {n_instances} instances):")
print("-" * 50)
print(f"{'Rank':<4} {'Feature':<25} {'LIME Score':<12}")
print("-" * 50)

for i, (_, row) in enumerate(lime_importance_df.iterrows(), 1):
    print(f"{i:2d}. {row['Feature']:<25} {row['LIME_Avg_Importance']:<12.6f}")

# ====== COMPARATIVE ANALYSIS ======
print("\n" + "="*60)
print("COMPARATIVE ANALYSIS: MUTUAL INFORMATION vs LIME")
print("="*60)

# Merge MI and LIME results
comparison_df = pd.merge(
    mi_results_df[['Feature', 'MI_Discretized']], 
    lime_importance_df[['Feature', 'LIME_Avg_Importance']], 
    on='Feature'
)

# Calculate normalized scores for comparison
comparison_df['MI_Normalized'] = comparison_df['MI_Discretized'] / comparison_df['MI_Discretized'].max()
comparison_df['LIME_Normalized'] = comparison_df['LIME_Avg_Importance'] / comparison_df['LIME_Avg_Importance'].max()

# Calculate correlation between MI and LIME scores
mi_lime_corr_pearson, mi_lime_p_pearson = pearsonr(comparison_df['MI_Discretized'], comparison_df['LIME_Avg_Importance'])
mi_lime_corr_spearman, mi_lime_p_spearman = spearmanr(comparison_df['MI_Discretized'], comparison_df['LIME_Avg_Importance'])

print(f"Correlation between Mutual Information and LIME scores:")
print(f"Pearson correlation: {mi_lime_corr_pearson:.4f} (p-value: {mi_lime_p_pearson:.4f})")
print(f"Spearman correlation: {mi_lime_corr_spearman:.4f} (p-value: {mi_lime_p_spearman:.4f})")

# Sort by combined normalized score
comparison_df['Combined_Score'] = (comparison_df['MI_Normalized'] + comparison_df['LIME_Normalized']) / 2
comparison_df = comparison_df.sort_values('Combined_Score', ascending=False)

print(f"\nTop 15 Features - Combined Ranking (MI + LIME):")
print("-" * 120)
print(f"{'Rank':<4} {'Feature':<25} {'MI Score':<12} {'LIME Score':<12} {'MI Norm':<10} {'LIME Norm':<10} {'Combined':<10}")
print("-" * 120)

for i, (_, row) in enumerate(comparison_df.head(15).iterrows(), 1):
    print(f"{i:2d}. {row['Feature']:<25} {row['MI_Discretized']:<12.6f} {row['LIME_Avg_Importance']:<12.6f} "
          f"{row['MI_Normalized']:<10.4f} {row['LIME_Normalized']:<10.4f} {row['Combined_Score']:<10.4f}")

# Identify features that rank high in both methods
print(f"\nFeatures ranking in Top 10 for both MI and LIME:")
print("-" * 60)

# Get top 10 features for each method
top_10_mi_features = set(mi_results_df.head(10)['Feature'].values)
top_10_lime_features = set(lime_importance_df.head(10)['Feature'].values)

# Find intersection
common_top_features = top_10_mi_features.intersection(top_10_lime_features)

if common_top_features:
    print(f"Features in both top 10 lists ({len(common_top_features)} features):")
    for feature in sorted(common_top_features):
        mi_rank = mi_results_df[mi_results_df['Feature'] == feature].index[0] + 1
        lime_rank = lime_importance_df[lime_importance_df['Feature'] == feature].index[0] + 1
        print(f"  - {feature:<25} | MI Rank: {mi_rank:2d} | LIME Rank: {lime_rank:2d}")
else:
    print("No features appear in both top 10 lists!")

# Identify features with large ranking differences
print(f"\nFeatures with largest ranking differences between MI and LIME:")
print("-" * 80)

# Calculate ranking differences
ranking_diff_analysis = []
for feature in feature_names:
    mi_rank = mi_results_df[mi_results_df['Feature'] == feature].index[0] + 1
    lime_rank = lime_importance_df[lime_importance_df['Feature'] == feature].index[0] + 1
    rank_diff = abs(mi_rank - lime_rank)
    ranking_diff_analysis.append((feature, mi_rank, lime_rank, rank_diff))

# Sort by ranking difference
ranking_diff_analysis.sort(key=lambda x: x[3], reverse=True)

print(f"{'Feature':<25} {'MI Rank':<8} {'LIME Rank':<10} {'Difference':<10}")
print("-" * 80)
for feature, mi_rank, lime_rank, diff in ranking_diff_analysis[:10]:
    print(f"{feature:<25} {mi_rank:<8} {lime_rank:<10} {diff:<10}")

# Feature stability analysis
print(f"\nFeature Stability Analysis:")
print("-" * 50)
print("Features with consistent high importance across methods:")

# Define "high importance" as top 20% of features
n_features = len(feature_names)
top_20_percent = int(0.2 * n_features)

top_mi_features = set(mi_results_df.head(top_20_percent)['Feature'].values)
top_lime_features = set(lime_importance_df.head(top_20_percent)['Feature'].values)
stable_features = top_mi_features.intersection(top_lime_features)

if stable_features:
    print(f"Stable high-importance features ({len(stable_features)} features):")
    for feature in sorted(stable_features):
        mi_score = mi_results_df[mi_results_df['Feature'] == feature]['MI_Discretized'].values[0]
        lime_score = lime_importance_df[lime_importance_df['Feature'] == feature]['LIME_Avg_Importance'].values[0]
        print(f"  - {feature:<25} | MI: {mi_score:.6f} | LIME: {lime_score:.6f}")
else:
    print("No features consistently rank in top 20% for both methods!")

# Save comprehensive results
print(f"\nSaving comprehensive analysis results...")

# Save detailed comparison
comparison_df.to_csv('mutual_info_lime_comparison.csv', index=False)

# Save mutual information results
mi_results_df.to_csv('mutual_information_analysis.csv', index=False)

# Create comprehensive summary
comprehensive_summary = {
    'Model_Performance': {
        'Test_Accuracy': float(accuracy_test),
        'Confusion_Matrix': conf_matrix_test.tolist()
    },
    'Mutual_Information_Analysis': {
        'Total_Features': len(feature_names),
        'Mean_MI_Score': float(mi_scores.mean()),
        'Std_MI_Score': float(mi_scores.std()),
        'Max_MI_Score': float(mi_scores.max()),
        'Min_MI_Score': float(mi_scores.min()),
        'Top_5_MI_Contribution_Percent': float((top_5_mi_contribution/total_mi_importance)*100),
        'Top_10_MI_Contribution_Percent': float((top_10_mi_contribution/total_mi_importance)*100),
        'Top_10_MI_Features': mi_results_df.head(10)['Feature'].tolist()
    },
    'LIME_Analysis': {
        'Instances_Analyzed': n_instances,
        'Top_10_LIME_Features': lime_importance_df.head(10)['Feature'].tolist()
    },
    'Comparative_Analysis': {
        'MI_LIME_Pearson_Correlation': float(mi_lime_corr_pearson),
        'MI_LIME_Spearman_Correlation': float(mi_lime_corr_spearman),
        'Features_In_Both_Top10': list(common_top_features),
        'Stable_High_Importance_Features': list(stable_features),
        'Top_10_Combined_Features': comparison_df.head(10)['Feature'].tolist()
    }
}

# Save comprehensive summary
with open('comprehensive_feature_analysis.json', 'w') as f:
    json.dump(comprehensive_summary, f, indent=2)

print(f"Analysis complete! Files saved:")
print(f"- mutual_info_lime_comparison.csv")
print(f"- mutual_information_analysis.csv") 
print(f"- comprehensive_feature_analysis.json")
print(f"- best_heloc_mlp_model.pkl")

print("\n" + "="*60)
print("COMPREHENSIVE FEATURE ANALYSIS COMPLETE")
print("="*60)