import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import f_classif
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import warnings
from scipy import stats
warnings.filterwarnings('ignore')

# =============================================================================
# LOAD DATA AND PREPARE FOR ANOVA F-TEST ANALYSIS
# =============================================================================

# Load dataset
data1 = pd.read_csv('/Users/saikat/Desktop/Summer 2025/CFNet_AAAI_26/Graduate Admission Dataset/admission_train.csv')
data2 = pd.read_csv('/Users/saikat/Desktop/Summer 2025/CFNet_AAAI_26/Graduate Admission Dataset/admission_test.csv')

# Combine datasets
data = pd.concat([data1, data2], ignore_index=True)

# Get feature names
feature_names = data.drop(columns=['target']).columns.tolist()

# Define features and target
X = data.drop(columns=['target']).values
y = data['target'].values

# Split data (same as training script)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=0)

# Standardize features (same as training script)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# =============================================================================
# LOAD TRAINED MODEL FOR COMPARISON
# =============================================================================

model_path = '/Users/saikat/Desktop/Summer 2025/CFNet_AAAI_26/Graduate Admission Dataset/best_admission_rf_model.pkl'
rf_model = joblib.load(model_path)

print("✓ Model loaded successfully!")
print(f"✓ Features: {feature_names}")
print(f"✓ Training set size: {X_train_scaled.shape[0]} samples")
print(f"✓ Test set size: {X_test_scaled.shape[0]} samples")

# Quick model performance check
y_pred = rf_model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
print(f"✓ Model accuracy on test set: {accuracy:.4f}")

# =============================================================================
# ANOVA F-TEST ANALYSIS - FIND MOST IMPORTANT FEATURE
# =============================================================================

print("\nCalculating ANOVA F-test scores...")

# Calculate ANOVA F-test for both scaled and original data
print("  Computing F-scores on scaled training data...")
f_scores_scaled, p_values_scaled = f_classif(X_train_scaled, y_train)

print("  Computing F-scores on original training data...")
f_scores_original, p_values_original = f_classif(X_train, y_train)

print("✓ ANOVA F-test calculation completed!")

# =============================================================================
# FEATURE IMPORTANCE RANKING
# =============================================================================

# Create DataFrames for both scaled and original data results
feature_importance_scaled = pd.DataFrame({
    'Feature': feature_names,
    'F_Score_Scaled': f_scores_scaled,
    'P_Value_Scaled': p_values_scaled
}).sort_values('F_Score_Scaled', ascending=False)

feature_importance_original = pd.DataFrame({
    'Feature': feature_names,
    'F_Score_Original': f_scores_original,
    'P_Value_Original': p_values_original
}).sort_values('F_Score_Original', ascending=False)

# Combine both results for comparison
feature_importance_combined = pd.merge(
    feature_importance_scaled, 
    feature_importance_original[['Feature', 'F_Score_Original', 'P_Value_Original']], 
    on='Feature'
)

# Calculate average F-score
feature_importance_combined['F_Score_Average'] = (
    feature_importance_combined['F_Score_Scaled'] + 
    feature_importance_combined['F_Score_Original']
) / 2

# Calculate average p-value
feature_importance_combined['P_Value_Average'] = (
    feature_importance_combined['P_Value_Scaled'] + 
    feature_importance_combined['P_Value_Original']
) / 2

# Sort by average F-score (higher is better)
feature_importance_final = feature_importance_combined.sort_values('F_Score_Average', ascending=False)

# =============================================================================
# RESULTS
# =============================================================================

print("\n" + "="*100)
print("FEATURE IMPORTANCE RANKING (ANOVA F-TEST Analysis)")
print("="*100)
print(f"{'Rank':<4} {'Feature':<25} {'Scaled F':<12} {'Original F':<12} {'Average F':<12} {'Avg P-Value':<12}")
print("-" * 100)

for i, (idx, row) in enumerate(feature_importance_final.iterrows(), 1):
    print(f"{i:<4} {row['Feature']:<25} {row['F_Score_Scaled']:<12.4f} "
          f"{row['F_Score_Original']:<12.4f} {row['F_Score_Average']:<12.4f} {row['P_Value_Average']:<12.6f}")

print("\n" + "="*100)
print("MOST IMPORTANT FEATURE")
print("="*100)

most_important_feature = feature_importance_final.iloc[0]['Feature']
highest_f_score_scaled = feature_importance_final.iloc[0]['F_Score_Scaled']
highest_f_score_original = feature_importance_final.iloc[0]['F_Score_Original']
highest_f_score_average = feature_importance_final.iloc[0]['F_Score_Average']
lowest_p_value_average = feature_importance_final.iloc[0]['P_Value_Average']

print(f"Feature Name:         {most_important_feature}")
print(f"F-Score (Scaled):     {highest_f_score_scaled:.4f}")
print(f"F-Score (Original):   {highest_f_score_original:.4f}")
print(f"F-Score (Average):    {highest_f_score_average:.4f}")
print(f"P-Value (Average):    {lowest_p_value_average:.2e}")

# Statistical significance check
alpha = 0.05
if lowest_p_value_average < alpha:
    print(f"✓ STATISTICALLY SIGNIFICANT at α = {alpha} level")
else:
    print(f"✗ NOT statistically significant at α = {alpha} level")

# Additional insights
if len(feature_importance_final) > 1:
    second_best_score = feature_importance_final.iloc[1]['F_Score_Average']
    if second_best_score > 0:
        ratio = highest_f_score_average / second_best_score
        print(f"\nThis feature's F-score is {ratio:.2f}x higher than the 2nd most important feature.")

print("\n" + "="*100)
print("TOP 5 FEATURES DETAILED ANALYSIS")
print("="*100)

top_5 = feature_importance_final.head(5)
total_f_score = feature_importance_final['F_Score_Average'].sum()

for i, (idx, row) in enumerate(top_5.iterrows(), 1):
    feature = row['Feature']
    score_avg = row['F_Score_Average']
    score_scaled = row['F_Score_Scaled']
    score_original = row['F_Score_Original']
    p_value_avg = row['P_Value_Average']
    
    if total_f_score > 0:
        percentage = (score_avg / total_f_score) * 100
        significance = "***" if p_value_avg < 0.001 else "**" if p_value_avg < 0.01 else "*" if p_value_avg < 0.05 else "ns"
        
        print(f"{i}. {feature}")
        print(f"   Average F-Score:  {score_avg:.4f} ({percentage:.1f}% of total)")
        print(f"   Scaled F-Score:   {score_scaled:.4f}")
        print(f"   Original F-Score: {score_original:.4f}")
        print(f"   P-Value:          {p_value_avg:.2e} ({significance})")
        print()

print("Significance levels: *** p<0.001, ** p<0.01, * p<0.05, ns = not significant")

# =============================================================================
# STATISTICAL INSIGHTS
# =============================================================================

print("=" * 100)
print("ANOVA F-TEST INSIGHTS")
print("=" * 100)

print("ANOVA F-test measures the linear relationship between each feature and the target:")
print("• Higher F-scores indicate stronger linear relationships")
print("• F-scores are always non-negative")
print("• P-values indicate statistical significance (lower is better)")
print("• F-test assumes normality and homogeneity of variances")
print("• Unlike MI, F-test specifically captures linear relationships")

print(f"\nDataset Statistics:")
print(f"• Total features analyzed: {len(feature_names)}")
print(f"• Highest F-score: {highest_f_score_average:.4f}")
print(f"• Lowest F-score: {feature_importance_final.iloc[-1]['F_Score_Average']:.4f}")
print(f"• Average F-score across all features: {feature_importance_final['F_Score_Average'].mean():.4f}")
print(f"• Standard deviation of F-scores: {feature_importance_final['F_Score_Average'].std():.4f}")

# Count statistically significant features
significant_features = feature_importance_final[feature_importance_final['P_Value_Average'] < 0.05]
highly_significant_features = feature_importance_final[feature_importance_final['P_Value_Average'] < 0.001]

print(f"• Statistically significant features (p < 0.05): {len(significant_features)}")
print(f"• Highly significant features (p < 0.001): {len(highly_significant_features)}")

# Check for non-significant features
non_significant_features = feature_importance_final[feature_importance_final['P_Value_Average'] >= 0.05]
if len(non_significant_features) > 0:
    print(f"\nFeatures that are NOT statistically significant (p ≥ 0.05):")
    for _, row in non_significant_features.iterrows():
        print(f"  • {row['Feature']}: F-score = {row['F_Score_Average']:.4f}, p-value = {row['P_Value_Average']:.4f}")

# =============================================================================
# COMPARISON WITH RANDOM FOREST FEATURE IMPORTANCE
# =============================================================================

print("\n" + "="*100)
print("COMPARISON: ANOVA F-TEST vs RANDOM FOREST IMPORTANCE")
print("="*100)

# Get Random Forest feature importance
rf_importance = pd.DataFrame({
    'Feature': feature_names,
    'RF_Importance': rf_model.feature_importances_
}).sort_values('RF_Importance', ascending=False)

# Merge for comparison
comparison = pd.merge(
    feature_importance_final[['Feature', 'F_Score_Average', 'P_Value_Average']], 
    rf_importance, 
    on='Feature'
)

print(f"{'Feature':<25} {'F-Test Rank':<12} {'RF Rank':<8} {'F-Score':<12} {'P-Value':<12} {'RF Score':<12}")
print("-" * 100)

for _, row in comparison.iterrows():
    feature = row['Feature']
    f_score = row['F_Score_Average']
    p_value = row['P_Value_Average']
    rf_score = row['RF_Importance']
    
    # Find ranks
    f_rank = feature_importance_final[feature_importance_final['Feature'] == feature].index[0] + 1
    rf_rank = rf_importance[rf_importance['Feature'] == feature].index[0] + 1
    
    print(f"{feature:<25} {f_rank:<12} {rf_rank:<8} {f_score:<12.4f} {p_value:<12.2e} {rf_score:<12.6f}")

# Calculate correlation between F-test and RF importance
correlation = np.corrcoef(comparison['F_Score_Average'], comparison['RF_Importance'])[0, 1]
print(f"\nCorrelation between F-test scores and RF importance: {correlation:.4f}")

# =============================================================================
# FEATURE SELECTION RECOMMENDATIONS
# =============================================================================

print("\n" + "="*100)
print("FEATURE SELECTION RECOMMENDATIONS")
print("="*100)

# Recommend features based on statistical significance and F-scores
print("Recommended feature selection strategies:")

print("\n1. CONSERVATIVE APPROACH (High confidence):")
high_conf_features = feature_importance_final[
    (feature_importance_final['P_Value_Average'] < 0.001) & 
    (feature_importance_final['F_Score_Average'] > feature_importance_final['F_Score_Average'].median())
]
print(f"   Select {len(high_conf_features)} features with p < 0.001 and F-score > median:")
for _, row in high_conf_features.iterrows():
    print(f"   • {row['Feature']} (F-score: {row['F_Score_Average']:.4f})")

print("\n2. MODERATE APPROACH (Good balance):")
moderate_features = feature_importance_final[feature_importance_final['P_Value_Average'] < 0.01]
print(f"   Select {len(moderate_features)} features with p < 0.01:")
for _, row in moderate_features.head(8).iterrows():  # Show top 8
    print(f"   • {row['Feature']} (F-score: {row['F_Score_Average']:.4f})")
if len(moderate_features) > 8:
    print(f"   ... and {len(moderate_features) - 8} more features")

print("\n3. LIBERAL APPROACH (Include more features):")
liberal_features = feature_importance_final[feature_importance_final['P_Value_Average'] < 0.05]
print(f"   Select all {len(liberal_features)} statistically significant features (p < 0.05)")

print("\n" + "="*100)
print("ANALYSIS COMPLETE!")
print("="*100)
print(f"The most important feature according to ANOVA F-test is: {most_important_feature}")
print(f"This feature shows the strongest LINEAR relationship with the admission decision.")
print(f"F-Score: {highest_f_score_average:.4f}, P-Value: {lowest_p_value_average:.2e}")

if lowest_p_value_average < 0.001:
    print("This result is HIGHLY STATISTICALLY SIGNIFICANT (p < 0.001)")
elif lowest_p_value_average < 0.01:
    print("This result is STATISTICALLY SIGNIFICANT (p < 0.01)")
elif lowest_p_value_average < 0.05:
    print("This result is STATISTICALLY SIGNIFICANT (p < 0.05)")
else:
    print("WARNING: This result is NOT statistically significant (p ≥ 0.05)")