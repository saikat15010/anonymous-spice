import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import f_classif
import joblib

# Load dataset
data1 = pd.read_csv('/Users/saikat/Desktop/Summer 2025/CFNet_AAAI_26/Student Performance Datastet/student_train.csv')
data2 = pd.read_csv('/Users/saikat/Desktop/Summer 2025/CFNet_AAAI_26/Student Performance Datastet/student_test.csv')

data = pd.concat([data1, data2], ignore_index=True)

# Binary encoding
binary_features = ['famsup', 'higher', 'internet', 'romantic']
for feature in binary_features:
    data[feature] = data[feature].map({'yes': 1, 'no': 0})

# Feature-target split
feature_names = data.drop(columns=['target']).columns.tolist()
X = data.drop(columns=['target']).values
y = data['target'].values

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=0)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Load trained logistic regression model
model_path = '/Users/saikat/Desktop/Summer 2025/CFNet_AAAI_26/Student Performance Datastet/best_admission_lr_model.pkl'
lr_model = joblib.load(model_path)

# Evaluate model
y_pred = lr_model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
print(f"✓ Logistic Regression Accuracy: {accuracy:.4f}")
print(f"✓ Features: {feature_names}")

# ANOVA F-test
print("\nCalculating ANOVA F-test scores...")
f_scaled, p_scaled = f_classif(X_train_scaled, y_train)
f_original, p_original = f_classif(X_train, y_train)
print("✓ F-test calculation complete.")

# Combine F-test results
anova_df = pd.DataFrame({
    'Feature': feature_names,
    'F_Score_Scaled': f_scaled,
    'P_Value_Scaled': p_scaled,
    'F_Score_Original': f_original,
    'P_Value_Original': p_original
})
anova_df['F_Score_Average'] = (anova_df['F_Score_Scaled'] + anova_df['F_Score_Original']) / 2
anova_df['P_Value_Average'] = (anova_df['P_Value_Scaled'] + anova_df['P_Value_Original']) / 2
anova_df = anova_df.sort_values('F_Score_Average', ascending=False).reset_index(drop=True)

# Logistic Regression coefficient importance
lr_importance = pd.DataFrame({
    'Feature': feature_names,
    'LR_Importance': np.abs(lr_model.coef_).flatten()
}).sort_values('LR_Importance', ascending=False).reset_index(drop=True)

# Merge for comparison
comparison = pd.merge(anova_df[['Feature', 'F_Score_Average', 'P_Value_Average']], lr_importance, on='Feature')
comparison = comparison.sort_values('F_Score_Average', ascending=False).reset_index(drop=True)

# Display top 5
print("\nTop 5 features by F-score and Logistic Coefficients:")
print(f"{'Feature':<20} {'F-Score':<12} {'P-Value':<12} {'LR Coef':<12}")
print("-" * 60)
for i, row in comparison.head(5).iterrows():
    print(f"{row['Feature']:<20} {row['F_Score_Average']:<12.4f} {row['P_Value_Average']:<12.2e} {row['LR_Importance']:<12.6f}")

# Feature ranks
print("\nComplete Comparison: ANOVA F-Test vs Logistic Coefficients")
print(f"{'Feature':<20} {'F-Test Rank':<12} {'LR Rank':<10} {'F-Score':<12} {'LR Coef':<12}")
print("-" * 70)
for i, row in comparison.iterrows():
    f_rank = i + 1
    lr_rank = lr_importance[lr_importance['Feature'] == row['Feature']].index[0] + 1
    print(f"{row['Feature']:<20} {f_rank:<12} {lr_rank:<10} {row['F_Score_Average']:<12.4f} {row['LR_Importance']:<12.6f}")

# Correlation
correlation = np.corrcoef(comparison['F_Score_Average'], comparison['LR_Importance'])[0, 1]
print(f"\nCorrelation between F-test scores and Logistic Regression coefficients: {correlation:.4f}")

# Summary
most_important = comparison.iloc[0]
print("\n" + "="*60)
print(f"Most Important Feature by F-Test: {most_important['Feature']}")
print(f"F-Score: {most_important['F_Score_Average']:.4f}")
print(f"P-Value: {most_important['P_Value_Average']:.2e}")
print(f"Logistic Coefficient (abs): {most_important['LR_Importance']:.6f}")
if most_important['P_Value_Average'] < 0.001:
    print("✓ HIGHLY statistically significant (p < 0.001)")
elif most_important['P_Value_Average'] < 0.01:
    print("✓ Statistically significant (p < 0.01)")
elif most_important['P_Value_Average'] < 0.05:
    print("✓ Statistically significant (p < 0.05)")
else:
    print("✗ NOT statistically significant (p ≥ 0.05)")
print("="*60)
