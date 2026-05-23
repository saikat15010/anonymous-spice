import numpy as np
import pandas as pd
import dice_ml
import joblib
import warnings
from itertools import combinations
from scipy.spatial.distance import hamming
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

# --- Configuration & Paths ---
DATA_PATH = '/Users/saikat/Desktop/Summer 2025/CFNet_AAAI_26/GMC Dataset/AISTATS GMC Dataset/preprocessed_credit_data.csv'
MODEL_PATH = '/Users/saikat/Desktop/Summer 2025/CFNet_AAAI_26/GMC Dataset/AISTATS GMC Dataset/best_credit_xgb_model.pkl'

# Define feature columns matching your SPICE GMC script
FEATURE_COLUMNS = [
    'RevolvingUtilizationOfUnsecuredLines', 'age', 
    'NumberOfTime30-59DaysPastDueNotWorse', 'DebtRatio', 'MonthlyIncome', 
    'NumberOfOpenCreditLinesAndLoans', 'NumberOfTimes90DaysLate', 
    'NumberRealEstateLoansOrLines', 'NumberOfTime60-89DaysPastDueNotWorse', 
    'NumberOfDependents'
]
CATEGORICAL_FEATS = ['NumberOfDependents', 'NumberOfTime30-59DaysPastDueNotWorse', 'NumberRealEstateLoansOrLines']
NUMERICAL_FEATS   = [f for f in FEATURE_COLUMNS if f not in CATEGORICAL_FEATS]
IMMUTABLE_FEATS   = ['age'] # Standard constraint for GMC
TARGET_COL        = 'SeriousDlqin2yrs'

NUM_QUERIES = 100
RANDOM_SEED = 42

def calculate_heom(q, c, cat_idx, num_idx, ranges):
    dist = 0.0
    for i in cat_idx: dist += 1.0 if q[i] != c[i] else 0.0
    for i in num_idx:
        rng = ranges[i] if ranges[i] != 0 else 1
        dist += abs(q[i] - c[i]) / rng
    return dist

def calculate_diversity(cf_list):
    if len(cf_list) < 2: return 0.0
    pairs = list(combinations(cf_list, 2))
    dists = [hamming(p[0], p[1]) for p in pairs]
    return np.mean(dists) * 100.0

def main():
    # 1. Load and Clean Data
    df = pd.read_csv(DATA_PATH).dropna(subset=[TARGET_COL])
    
    X_raw = df[FEATURE_COLUMNS].values
    y_raw = df[TARGET_COL].values
    
    # Split matches SPICE protocol: 85% Train, 15% Test
    split_idx = int(0.85 * len(X_raw))
    X_train_raw, X_test_raw = X_raw[:split_idx], X_raw[split_idx:]
    
    scaler = StandardScaler().fit(X_train_raw)
    loaded_model = joblib.load(MODEL_PATH)

    # Plausibility Model
    isf = IsolationForest(contamination=0.10, random_state=RANDOM_SEED)
    isf.fit(scaler.transform(X_train_raw))
    
    ranges = np.array([df[f].max() - df[f].min() if f in NUMERICAL_FEATS else 0 for f in FEATURE_COLUMNS])
    cat_idx = [FEATURE_COLUMNS.index(f) for f in CATEGORICAL_FEATS]
    num_idx = [FEATURE_COLUMNS.index(f) for f in NUMERICAL_FEATS]

    # Wrapper to handle XGBoost + Scaling
    class ModelWrapper:
        def __init__(self, model, scaler):
            self.model, self.scaler = model, scaler
        def predict_proba(self, data):
            return self.model.predict_proba(self.scaler.transform(data.values))
        def predict(self, data):
            return self.model.predict(self.scaler.transform(data.values))

    wrapped_model = ModelWrapper(loaded_model, scaler)
    d_data = dice_ml.Data(dataframe=df[FEATURE_COLUMNS + [TARGET_COL]], 
                          continuous_features=NUMERICAL_FEATS, outcome_name=TARGET_COL)
    d_model = dice_ml.Model(model=wrapped_model, backend="sklearn")
    exp = dice_ml.Dice(d_data, d_model, method="genetic") 

    # Select queries from test pool where model predicts 0 (No delinquency)
    test_X_scaled = scaler.transform(X_test_raw)
    preds = loaded_model.predict(test_X_scaled)
    query_pool_indices = np.where(preds == 0)[0] + split_idx
    
    selected_indices = np.random.default_rng(RANDOM_SEED).choice(query_pool_indices, size=NUM_QUERIES, replace=False)

    results = []
    print(f"Generating DiCE counterfactuals for {len(selected_indices)} queries...")

    for idx in selected_indices:
        query_instance = df.iloc[[idx]][FEATURE_COLUMNS]
        q_array = X_raw[idx]
        try:
            dice_exp = exp.generate_counterfactuals(
                query_instance, total_CFs=10, desired_class="opposite",
                features_to_vary=[f for f in FEATURE_COLUMNS if f not in IMMUTABLE_FEATS]
            )
            cf_df = dice_exp.cf_examples_list[0].final_cfs_df
            if cf_df is None or cf_df.empty: continue
            
            # Ensure target is dropped if returned
            cf_values = cf_df[FEATURE_COLUMNS].values
            
            proxs = [calculate_heom(q_array, cf, cat_idx, num_idx, ranges) for cf in cf_values]
            spars = [np.sum(q_array != cf)/len(q_array) for cf in cf_values]
            plaus_preds = isf.predict(scaler.transform(cf_values))
            
            results.append({
                'best_sparsity': min(spars), 'best_proximity': min(proxs),
                'avg_sparsity': np.mean(spars), 'avg_proximity': np.mean(proxs),
                'diversity': calculate_diversity(cf_values),
                'plausibility': np.mean(plaus_preds == 1) * 100.0
            })
        except: continue

    res_df = pd.DataFrame(results)
    print("\n" + "="*65)
    print("DICE (GENETIC) BASELINE - GMC (100 QUERIES)")
    print("="*65)
    for m in ['best_sparsity', 'best_proximity', 'avg_sparsity', 'avg_proximity', 'diversity', 'plausibility']:
        print(f"{m:<15} {res_df[m].mean():>12.4f} +/- {res_df[m].std():.4f}")
    
    res_df.to_csv('table5_dice_GMC.csv', index=False)

if __name__ == "__main__":
    main()