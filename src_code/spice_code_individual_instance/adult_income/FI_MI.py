import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# LOAD DATA AND PREPARE FOR MUTUAL INFORMATION ANALYSIS
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
# LOAD TRAINED MODEL FOR COMPARISON
# =============================================================================

model_path = '/Users/saikat/Desktop/Summer 2025/CFNet/New Experiment/Adult Income Dataset/best_adult_income_gb_model.pkl'
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
# MUTUAL INFORMATION ANALYSIS - FIND MOST IMPORTANT FEATURE
# =============================================================================

print("\nCalculating Mutual Information scores...")

# Calculate mutual information for both scaled and original data
print("  Computing MI on scaled training data...")
mi_scores_scaled = mutual_info_classif(
    X_train_scaled, 
    y_train,
    discrete_features=False,  # All features are continuous
    n_neighbors=5,            # Number of neighbors for MI estimation
    random_state=42
)

print("  Computing MI on original training data...")
mi_scores_original = mutual_info_classif(
    X_train, 
    y_train,
    discrete_features=False,
    n_neighbors=5,
    random_state=42
)

print("✓ Mutual Information calculation completed!")

# =============================================================================
# FEATURE IMPORTANCE RANKING
# =============================================================================

# Create DataFrames for both scaled and original data results
feature_importance_scaled = pd.DataFrame({
    'Feature': feature_names,
    'MI_Score_Scaled': mi_scores_scaled
}).sort_values('MI_Score_Scaled', ascending=False)

feature_importance_original = pd.DataFrame({
    'Feature': feature_names,
    'MI_Score_Original': mi_scores_original
}).sort_values('MI_Score_Original', ascending=False)

# Combine both results for comparison
feature_importance_combined = pd.merge(
    feature_importance_scaled, 
    feature_importance_original[['Feature', 'MI_Score_Original']], 
    on='Feature'
)

# Calculate average MI score
feature_importance_combined['MI_Score_Average'] = (
    feature_importance_combined['MI_Score_Scaled'] + 
    feature_importance_combined['MI_Score_Original']
) / 2

# Sort by average score
feature_importance_final = feature_importance_combined.sort_values('MI_Score_Average', ascending=False)

# =============================================================================
# RESULTS
# =============================================================================

print("\n" + "="*80)
print("FEATURE IMPORTANCE RANKING (MUTUAL INFORMATION Analysis)")
print("="*80)
print(f"{'Rank':<4} {'Feature':<25} {'Scaled MI':<12} {'Original MI':<12} {'Average MI':<12}")
print("-" * 80)

for i, (idx, row) in enumerate(feature_importance_final.iterrows(), 1):
    print(f"{i:<4} {row['Feature']:<25} {row['MI_Score_Scaled']:<12.6f} "
          f"{row['MI_Score_Original']:<12.6f} {row['MI_Score_Average']:<12.6f}")

print("\n" + "="*80)
print("MOST IMPORTANT FEATURE")
print("="*80)

most_important_feature = feature_importance_final.iloc[0]['Feature']
highest_mi_score_scaled = feature_importance_final.iloc[0]['MI_Score_Scaled']
highest_mi_score_original = feature_importance_final.iloc[0]['MI_Score_Original']
highest_mi_score_average = feature_importance_final.iloc[0]['MI_Score_Average']

print(f"Feature Name:        {most_important_feature}")
print(f"MI Score (Scaled):   {highest_mi_score_scaled:.6f}")
print(f"MI Score (Original): {highest_mi_score_original:.6f}")
print(f"MI Score (Average):  {highest_mi_score_average:.6f}")

# Additional insights
if len(feature_importance_final) > 1:
    second_best_score = feature_importance_final.iloc[1]['MI_Score_Average']
    if second_best_score > 0:
        ratio = highest_mi_score_average / second_best_score
        print(f"\nThis feature is {ratio:.2f}x more important than the 2nd most important feature.")

print("\n" + "="*80)
print("TOP 5 FEATURES DETAILED ANALYSIS")
print("="*80)

top_5 = feature_importance_final.head(5)
total_mi = feature_importance_final['MI_Score_Average'].sum()

for i, (idx, row) in enumerate(top_5.iterrows(), 1):
    feature = row['Feature']
    score_avg = row['MI_Score_Average']
    score_scaled = row['MI_Score_Scaled']
    score_original = row['MI_Score_Original']
    
    if total_mi > 0:
        percentage = (score_avg / total_mi) * 100
        print(f"{i}. {feature}")
        print(f"   Average MI:  {score_avg:.6f} ({percentage:.1f}% of total)")
        print(f"   Scaled MI:   {score_scaled:.6f}")
        print(f"   Original MI: {score_original:.6f}")
        print()

# =============================================================================
# STATISTICAL INSIGHTS
# =============================================================================

print("=" * 80)
print("MUTUAL INFORMATION INSIGHTS")
print("=" * 80)

print("Mutual Information measures the dependency between each feature and the target:")
print("• Higher MI scores indicate stronger statistical dependency")
print("• MI = 0 means features are independent of the target")
print("• MI values are always non-negative")
print("• Unlike correlation, MI captures both linear and non-linear relationships")

print(f"\nDataset Statistics:")
print(f"• Total features analyzed: {len(feature_names)}")
print(f"• Highest MI score: {highest_mi_score_average:.6f}")
print(f"• Lowest MI score: {feature_importance_final.iloc[-1]['MI_Score_Average']:.6f}")
print(f"• Average MI across all features: {feature_importance_final['MI_Score_Average'].mean():.6f}")
print(f"• Standard deviation of MI scores: {feature_importance_final['MI_Score_Average'].std():.6f}")

# Check for feature independence
low_mi_features = feature_importance_final[feature_importance_final['MI_Score_Average'] < 0.01]
if len(low_mi_features) > 0:
    print(f"\nFeatures with very low MI (< 0.01) - potentially less informative:")
    for _, row in low_mi_features.iterrows():
        print(f"  • {row['Feature']}: {row['MI_Score_Average']:.6f}")

# =============================================================================
# COMPARISON WITH RANDOM FOREST FEATURE IMPORTANCE
# =============================================================================

print("\n" + "="*80)
print("COMPARISON: MUTUAL INFORMATION vs RANDOM FOREST IMPORTANCE")
print("="*80)

# Get Random Forest feature importance
rf_importance = pd.DataFrame({
    'Feature': feature_names,
    'RF_Importance': rf_model.feature_importances_
}).sort_values('RF_Importance', ascending=False)

# Merge for comparison
comparison = pd.merge(
    feature_importance_final[['Feature', 'MI_Score_Average']], 
    rf_importance, 
    on='Feature'
)

print(f"{'Feature':<25} {'MI Rank':<8} {'RF Rank':<8} {'MI Score':<12} {'RF Score':<12}")
print("-" * 80)

for _, row in comparison.iterrows():
    feature = row['Feature']
    mi_score = row['MI_Score_Average']
    rf_score = row['RF_Importance']
    
    # Find ranks
    mi_rank = feature_importance_final[feature_importance_final['Feature'] == feature].index[0] + 1
    rf_rank = rf_importance[rf_importance['Feature'] == feature].index[0] + 1
    
    print(f"{feature:<25} {mi_rank:<8} {rf_rank:<8} {mi_score:<12.6f} {rf_score:<12.6f}")

# Calculate correlation between MI and RF importance
correlation = np.corrcoef(comparison['MI_Score_Average'], comparison['RF_Importance'])[0, 1]
print(f"\nCorrelation between MI and RF importance: {correlation:.4f}")

print("\n" + "="*80)
print("ANALYSIS COMPLETE!")
print("="*80)
print(f"The most important feature according to Mutual Information is: {most_important_feature}")
print(f"This feature has the strongest statistical dependency with the admission decision.")