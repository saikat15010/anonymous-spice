import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from scipy.stats import f_oneway
import joblib
import warnings
warnings.filterwarnings('ignore')

# Load dataset
data = pd.read_csv('/Users/saikat/Desktop/Summer 2025/CFNet/New Experiment/HELOC Dataset/heloc_dataset_v1.csv')

# Preprocess the target variable
data['target'] = data['target'].map({'Bad': 0, 'Good': 1})

# Get feature names for ANOVA analysis
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

# ====== ANOVA Analysis ======
print("\n" + "="*60)
print("ANOVA FEATURE IMPORTANCE ANALYSIS")
print("="*60)

# Get original unscaled data for ANOVA analysis
X_original = data.drop(columns=['target']).values
y_original = data['target'].values

print("Performing ANOVA F-test for feature selection...")

# Perform ANOVA F-test using SelectKBest
selector = SelectKBest(score_func=f_classif, k='all')
X_selected = selector.fit_transform(X_original, y_original)

# Get F-scores and p-values
f_scores = selector.scores_
p_values = selector.pvalues_

# Create ANOVA results DataFrame
anova_results = pd.DataFrame({
    'Feature': feature_names,
    'F_Score': f_scores,
    'P_Value': p_values,
    'Significant': p_values < 0.05,
    'Highly_Significant': p_values < 0.01,
    'Very_Highly_Significant': p_values < 0.001
})

# Sort by F-score (descending)
anova_results = anova_results.sort_values('F_Score', ascending=False)

print(f"\nANOVA Feature Importance Ranking (F-scores):")
print("-" * 90)
print(f"{'Rank':<4} {'Feature':<25} {'F-Score':<12} {'P-Value':<12} {'Significant':<12} {'Highly Sig':<12}")
print("-" * 90)

for i, (_, row) in enumerate(anova_results.iterrows(), 1):
    sig_symbol = "***" if row['Very_Highly_Significant'] else "**" if row['Highly_Significant'] else "*" if row['Significant'] else ""
    print(f"{i:2d}. {row['Feature']:<25} {row['F_Score']:<12.4f} {row['P_Value']:<12.6f} {str(row['Significant']):<12} {sig_symbol:<12}")

print(f"\nTop 10 Most Important Features (ANOVA F-test):")
print("-" * 90)
print(f"{'Rank':<4} {'Feature':<25} {'F-Score':<12} {'P-Value':<12} {'Significant':<12} {'Highly Sig':<12}")
print("-" * 90)

top_10_anova = anova_results.head(10)
for i, (_, row) in enumerate(top_10_anova.iterrows(), 1):
    sig_symbol = "***" if row['Very_Highly_Significant'] else "**" if row['Highly_Significant'] else "*" if row['Significant'] else ""
    print(f"{i:2d}. {row['Feature']:<25} {row['F_Score']:<12.4f} {row['P_Value']:<12.6f} {str(row['Significant']):<12} {sig_symbol:<12}")

# Manual ANOVA calculation for verification
print(f"\nManual ANOVA Verification (Top 5 features):")
print("-" * 70)
print(f"{'Feature':<25} {'F-Statistic':<15} {'P-Value':<15} {'Status':<15}")
print("-" * 70)

for i, (_, row) in enumerate(top_10_anova.head(5).iterrows(), 1):
    feature_idx = feature_names.index(row['Feature'])
    feature_data = X_original[:, feature_idx]
    
    # Split feature data by class
    class_0_data = feature_data[y_original == 0]  # Bad
    class_1_data = feature_data[y_original == 1]  # Good
    
    # Perform manual ANOVA F-test
    f_stat, p_val = f_oneway(class_0_data, class_1_data)
    
    status = "Highly Significant" if p_val < 0.001 else "Significant" if p_val < 0.05 else "Not Significant"
    print(f"{row['Feature']:<25} {f_stat:<15.4f} {p_val:<15.6f} {status:<15}")

# Statistical summary of ANOVA results
print(f"\nANOVA Analysis Summary:")
print("-" * 40)
print(f"Total number of features: {len(feature_names)}")
print(f"Significant features (p < 0.05): {sum(anova_results['Significant'])}")
print(f"Highly significant features (p < 0.01): {sum(anova_results['Highly_Significant'])}")
print(f"Very highly significant features (p < 0.001): {sum(anova_results['Very_Highly_Significant'])}")

# F-score statistics
f_score_stats = {
    'Mean F-Score': f_scores.mean(),
    'Std F-Score': f_scores.std(),
    'Max F-Score': f_scores.max(),
    'Min F-Score': f_scores.min(),
    'Median F-Score': np.median(f_scores)
}

print(f"\nF-Score Statistics:")
for key, value in f_score_stats.items():
    print(f"{key}: {value:.4f}")

# P-value statistics
p_value_stats = {
    'Mean P-Value': p_values.mean(),
    'Std P-Value': p_values.std(),
    'Max P-Value': p_values.max(),
    'Min P-Value': p_values.min(),
    'Median P-Value': np.median(p_values)
}

print(f"\nP-Value Statistics:")
for key, value in p_value_stats.items():
    print(f"{key}: {value:.6f}")

# Feature contribution analysis
total_f_score = f_scores.sum()
top_5_contribution = anova_results.head(5)['F_Score'].sum()
top_10_contribution = anova_results.head(10)['F_Score'].sum()

print(f"\nFeature Contribution Analysis:")
print(f"Top 5 features contribute: {(top_5_contribution/total_f_score)*100:.2f}% of total F-score")
print(f"Top 10 features contribute: {(top_10_contribution/total_f_score)*100:.2f}% of total F-score")

# Detailed class-wise statistics for top features
print(f"\nDetailed Class-wise Statistics for Top 10 Features:")
print("-" * 100)
print(f"{'Feature':<25} {'Class 0 Mean':<12} {'Class 1 Mean':<12} {'Class 0 Std':<12} {'Class 1 Std':<12} {'Mean Diff':<12}")
print("-" * 100)

for _, row in top_10_anova.iterrows():
    feature_idx = feature_names.index(row['Feature'])
    feature_data = X_original[:, feature_idx]
    
    class_0_data = feature_data[y_original == 0]
    class_1_data = feature_data[y_original == 1]
    
    class_0_mean = np.mean(class_0_data)
    class_1_mean = np.mean(class_1_data)
    class_0_std = np.std(class_0_data)
    class_1_std = np.std(class_1_data)
    mean_diff = abs(class_0_mean - class_1_mean)
    
    print(f"{row['Feature']:<25} {class_0_mean:<12.4f} {class_1_mean:<12.4f} {class_0_std:<12.4f} {class_1_std:<12.4f} {mean_diff:<12.4f}")

# Effect size calculation (Cohen's d) for top features
print(f"\nEffect Size Analysis (Cohen's d) for Top 10 Features:")
print("-" * 60)
print(f"{'Feature':<25} {'Cohen\'s d':<12} {'Effect Size':<20}")
print("-" * 60)

for _, row in top_10_anova.iterrows():
    feature_idx = feature_names.index(row['Feature'])
    feature_data = X_original[:, feature_idx]
    
    class_0_data = feature_data[y_original == 0]
    class_1_data = feature_data[y_original == 1]
    
    # Calculate Cohen's d
    mean_diff = np.mean(class_1_data) - np.mean(class_0_data)
    pooled_std = np.sqrt(((len(class_0_data) - 1) * np.var(class_0_data, ddof=1) + 
                         (len(class_1_data) - 1) * np.var(class_1_data, ddof=1)) / 
                        (len(class_0_data) + len(class_1_data) - 2))
    
    cohens_d = mean_diff / pooled_std
    
    # Interpret effect size
    if abs(cohens_d) < 0.2:
        effect_size = "Small"
    elif abs(cohens_d) < 0.5:
        effect_size = "Small to Medium"
    elif abs(cohens_d) < 0.8:
        effect_size = "Medium to Large"
    else:
        effect_size = "Large"
    
    print(f"{row['Feature']:<25} {cohens_d:<12.4f} {effect_size:<20}")

# Feature selection based on different significance levels
print(f"\nFeature Selection Based on Significance Levels:")
print("-" * 50)

significance_levels = [0.001, 0.01, 0.05, 0.1]
for alpha in significance_levels:
    selected_features = anova_results[anova_results['P_Value'] < alpha]['Feature'].tolist()
    print(f"Features with p-value < {alpha}: {len(selected_features)}")
    if len(selected_features) <= 10:
        print(f"  Selected features: {', '.join(selected_features[:10])}")
    else:
        print(f"  Top 10 selected features: {', '.join(selected_features[:10])}")

# Bonferroni correction for multiple comparisons
bonferroni_alpha = 0.05 / len(feature_names)
bonferroni_significant = anova_results[anova_results['P_Value'] < bonferroni_alpha]

print(f"\nBonferroni Correction Analysis:")
print(f"Adjusted significance level: {bonferroni_alpha:.6f}")
print(f"Features significant after Bonferroni correction: {len(bonferroni_significant)}")

if len(bonferroni_significant) > 0:
    print(f"Bonferroni-corrected significant features:")
    for i, (_, row) in enumerate(bonferroni_significant.iterrows(), 1):
        print(f"  {i:2d}. {row['Feature']:<25} | F-Score: {row['F_Score']:.4f} | P-Value: {row['P_Value']:.6f}")

# Save ANOVA results
anova_results.to_csv('anova_feature_importance.csv', index=False)
print(f"\nANOVA feature importance rankings saved to: anova_feature_importance.csv")

# Create comprehensive summary
anova_summary = {
    'Total_Features': len(feature_names),
    'Significant_Features_005': sum(anova_results['Significant']),
    'Highly_Significant_Features_001': sum(anova_results['Highly_Significant']),
    'Very_Highly_Significant_Features_0001': sum(anova_results['Very_Highly_Significant']),
    'Bonferroni_Significant_Features': len(bonferroni_significant),
    'Bonferroni_Alpha': bonferroni_alpha,
    'Mean_F_Score': f_scores.mean(),
    'Std_F_Score': f_scores.std(),
    'Max_F_Score': f_scores.max(),
    'Min_F_Score': f_scores.min(),
    'Top_5_Contribution_Percent': (top_5_contribution/total_f_score)*100,
    'Top_10_Contribution_Percent': (top_10_contribution/total_f_score)*100
}

# Save summary
import json
with open('anova_analysis_summary.json', 'w') as f:
    json.dump(anova_summary, f, indent=2)

print(f"ANOVA analysis summary saved to: anova_analysis_summary.json")

print("\n" + "="*60)
print("ANOVA ANALYSIS COMPLETE")
print("="*60)