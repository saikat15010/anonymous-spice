import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from lime import lime_tabular # type: ignore
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# LOAD DATA AND PREPARE FOR LIME ANALYSIS
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
# LOAD TRAINED MODEL
# =============================================================================

model_path = '/Users/saikat/Desktop/Summer 2025/CFNet_AAAI_26/Graduate Admission Dataset/best_admission_rf_model.pkl'
rf_model = joblib.load(model_path)

print("✓ Model loaded successfully!")
print(f"✓ Features: {feature_names}")
print(f"✓ Test set size: {X_test_scaled.shape[0]} samples")

# =============================================================================
# LIME ANALYSIS - FIND MOST IMPORTANT FEATURE
# =============================================================================

print("\nSetting up LIME explainer...")

# Create LIME explainer
explainer = lime_tabular.LimeTabularExplainer(
    X_train_scaled,
    feature_names=feature_names,
    class_names=['Rejected', 'Admitted'],
    mode='classification',
    discretize_continuous=True,
    random_state=42
)

print("Calculating LIME explanations for test samples...")

# Store feature importance scores from all explanations
all_feature_scores = []
sample_count = 0
max_samples = min(200, len(X_test_scaled))  # Analyze up to 200 samples for efficiency

for i in range(max_samples):
    try:
        # Get explanation for this sample
        explanation = explainer.explain_instance(
            X_test_scaled[i], 
            rf_model.predict_proba,
            num_features=len(feature_names),
            num_samples=1000
        )
        
        # Extract feature importance scores
        feature_scores = {}
        for feature_idx, score in explanation.as_list():
            # Parse feature name from LIME output (removes value ranges)
            feature_name = feature_idx.split(' ')[0] if ' ' in feature_idx else feature_idx
            # Map back to original feature names
            for orig_name in feature_names:
                if orig_name.replace(' ', '') in feature_name.replace(' ', '') or feature_name.replace(' ', '') in orig_name.replace(' ', ''):
                    feature_scores[orig_name] = abs(score)  # Use absolute value
                    break
        
        # If parsing failed, use index-based mapping
        if len(feature_scores) != len(feature_names):
            feature_scores = {}
            explanation_list = explanation.as_list()
            for j, (_, score) in enumerate(explanation_list):
                if j < len(feature_names):
                    feature_scores[feature_names[j]] = abs(score)
        
        all_feature_scores.append(feature_scores)
        sample_count += 1
        
        if (i + 1) % 10 == 0:
            print(f"  Processed {i + 1}/{max_samples} samples...")
            
    except Exception as e:
        print(f"  Warning: Skipped sample {i} due to error: {str(e)}")
        continue

print(f"✓ Successfully processed {sample_count} samples")

# =============================================================================
# AGGREGATE RESULTS ACROSS ALL SAMPLES
# =============================================================================

print("\nCalculating average feature importance across all samples...")

# Calculate mean importance for each feature
feature_importance_sums = {feature: 0.0 for feature in feature_names}
feature_counts = {feature: 0 for feature in feature_names}

for scores in all_feature_scores:
    for feature, score in scores.items():
        if feature in feature_importance_sums:
            feature_importance_sums[feature] += score
            feature_counts[feature] += 1

# Calculate average importance
avg_feature_importance = {}
for feature in feature_names:
    if feature_counts[feature] > 0:
        avg_feature_importance[feature] = feature_importance_sums[feature] / feature_counts[feature]
    else:
        avg_feature_importance[feature] = 0.0

# Create feature importance DataFrame
feature_importance = pd.DataFrame({
    'Feature': list(avg_feature_importance.keys()),
    'LIME_Importance': list(avg_feature_importance.values())
}).sort_values('LIME_Importance', ascending=False)

# =============================================================================
# RESULTS
# =============================================================================

print("\n" + "="*60)
print("FEATURE IMPORTANCE RANKING (LIME Analysis)")
print("="*60)

for i, (idx, row) in enumerate(feature_importance.iterrows(), 1):
    print(f"{i:2d}. {row['Feature']:25s} | LIME Score: {row['LIME_Importance']:.6f}")

print("\n" + "="*60)
print("MOST IMPORTANT FEATURE")
print("="*60)

most_important_feature = feature_importance.iloc[0]['Feature']
highest_lime_score = feature_importance.iloc[0]['LIME_Importance']

print(f"Feature Name: {most_important_feature}")
print(f"LIME Score:   {highest_lime_score:.6f}")

# Additional insights
if len(feature_importance) > 1:
    second_best_score = feature_importance.iloc[1]['LIME_Importance']
    if second_best_score > 0:
        ratio = highest_lime_score / second_best_score
        print(f"\nThis feature is {ratio:.2f}x more important than the 2nd most important feature.")

print("\n" + "="*60)
print("TOP 3 FEATURES SUMMARY")
print("="*60)

total_importance = feature_importance['LIME_Importance'].sum()
for i in range(min(3, len(feature_importance))):
    feature = feature_importance.iloc[i]['Feature']
    score = feature_importance.iloc[i]['LIME_Importance']
    if total_importance > 0:
        percentage = (score / total_importance) * 100
        print(f"{i+1}. {feature}: {score:.6f} ({percentage:.1f}% of total importance)")
    else:
        print(f"{i+1}. {feature}: {score:.6f}")

print("\n" + "="*60)
print("ANALYSIS SUMMARY")
print("="*60)
print(f"• Analyzed {sample_count} test samples")
print(f"• Most important feature: {most_important_feature}")
print(f"• LIME provides local explanations averaged across samples")
print(f"• Higher scores indicate greater influence on predictions")

print(f"\nAnalysis complete! The most important feature according to LIME is: {most_important_feature}")