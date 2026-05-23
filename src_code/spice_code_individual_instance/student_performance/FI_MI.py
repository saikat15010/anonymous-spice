import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import mutual_info_classif
import joblib

# Load dataset
data1 = pd.read_csv('/Users/saikat/Desktop/Summer 2025/CFNet_AAAI_26/Student Performance Datastet/student_train.csv')
data2 = pd.read_csv('/Users/saikat/Desktop/Summer 2025/CFNet_AAAI_26/Student Performance Datastet/student_test.csv')


data = pd.concat([data1, data2], ignore_index=True)

# Binary encoding
binary_features = ['famsup', 'higher', 'internet', 'romantic']
for feature in binary_features:
    data[feature] = data[feature].map({'yes': 1, 'no': 0})

feature_names = data.drop(columns=['target']).columns.tolist()
X = data.drop(columns=['target']).values
y = data['target'].values

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=0)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Load logistic regression model
model_path = '/Users/saikat/Desktop/Summer 2025/CFNet_AAAI_26/Student Performance Datastet/best_admission_lr_model.pkl'
lr_model = joblib.load(model_path)

# Evaluate accuracy
y_pred = lr_model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
print(f"✓ Logistic Regression Test Accuracy: {accuracy:.4f}")

# Mutual Information
print("\nCalculating Mutual Information scores...")
mi_scores_scaled = mutual_info_classif(X_train_scaled, y_train, discrete_features=False, random_state=42)
mi_scores_original = mutual_info_classif(X_train, y_train, discrete_features=False, random_state=42)

# Combine MI scores
mi_df = pd.DataFrame({
    'Feature': feature_names,
    'MI_Scaled': mi_scores_scaled,
    'MI_Original': mi_scores_original
})
mi_df['MI_Average'] = (mi_df['MI_Scaled'] + mi_df['MI_Original']) / 2
mi_df = mi_df.sort_values('MI_Average', ascending=False)

# Logistic Regression importance
coef_importance = pd.DataFrame({
    'Feature': feature_names,
    'LR_Importance': np.abs(lr_model.coef_).flatten()
}).sort_values('LR_Importance', ascending=False)

# Merge for comparison
comparison_df = pd.merge(mi_df, coef_importance, on='Feature')
comparison_df = comparison_df.sort_values('MI_Average', ascending=False)

# Print top 5
print("\nTop 5 features by MI Average:")
print(comparison_df[['Feature', 'MI_Average', 'LR_Importance']].head(5))

# Print complete comparison with ranking
print("\nComparison: Mutual Information vs Logistic Regression Coefficient Importance")
print(f"{'Feature':<25} {'MI Rank':<8} {'LR Rank':<8} {'MI Avg':<12} {'LR Coef':<12}")
print("-" * 70)
for _, row in comparison_df.iterrows():
    f = row['Feature']
    mi_rank = mi_df[mi_df['Feature'] == f].index[0] + 1
    lr_rank = coef_importance[coef_importance['Feature'] == f].index[0] + 1
    print(f"{f:<25} {mi_rank:<8} {lr_rank:<8} {row['MI_Average']:<12.6f} {row['LR_Importance']:<12.6f}")

# Correlation
corr = np.corrcoef(comparison_df['MI_Average'], comparison_df['LR_Importance'])[0, 1]
print(f"\nPearson correlation between MI and LR importance: {corr:.4f}")
