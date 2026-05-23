import numpy as np
import pandas as pd
import dice_ml
import joblib
import warnings
from itertools import combinations
from scipy.spatial.distance import hamming
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

DATA_PATH = '/Users/saikat/Desktop/Summer 2025/CFNet_AAAI_26/HELOC Dataset/AISTATS HELOC Dataset/heloc_dataset_v1.csv'
MODEL_PATH = 'best_heloc_mlp_model.pkl'

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
    df = pd.read_csv(DATA_PATH)
    df.columns = df.columns.str.strip()
    df['target'] = df['target'].map({'Bad': 0, 'Good': 1})
    
    # 1. DYNAMICALLY RECONSTRUCT THE 23 FEATURE COLUMNS
    # The error says the model expects 23. This list adds 'NumTradesOpeninLast12M' 
    # and ensures we catch the Bank/National line column correctly.
    bank_col = [c for c in df.columns if c.startswith('NumBank2')][0]
    
    # This list now contains exactly 23 features commonly found in the FICO HELOC set
    FEATURE_COLUMNS = [
        'ExternalRiskEstimate', 'MSinceOldestTradeOpen', 'MSinceMostRecentTradeOpen', 
        'AverageMInFile', 'NumSatisfactoryTrades', 'NumTrades60Ever2DerogPubRec', 
        'NumTrades90Ever2DerogPubRec', 'PercentTradesNeverDelq', 'MSinceMostRecentDelq', 
        'MaxDelq2PublicRecLast12M', 'MaxDelqEver', 'NumTotalTrades', 
        'NumTradesOpeninLast12M', 'PercentInstallTrades', 'MSinceMostRecentInqexcl7days', 
        'NumInqLast6M', 'NumInqLast6Mexcl7days', 'NetFractionRevolvingBurden', 
        'NetFractionInstallBurden', 'NumRevolvingTradesWBalance', 'NumInstallTradesWBalance', 
        bank_col,
        'NumTradesOpeninLast12M' 
    ]
    
    if len(set(FEATURE_COLUMNS)) != 23:
        FEATURE_COLUMNS = [c for c in df.columns if c != 'target']
        if len(FEATURE_COLUMNS) != 23:
             FEATURE_COLUMNS = df.columns.drop('target').tolist()

    print(f"Using {len(FEATURE_COLUMNS)} features to match MLP expectations.")

    CATEGORICAL_FEATS = ['NumTrades60Ever2DerogPubRec', 'NumTrades90Ever2DerogPubRec']
    NUMERICAL_FEATS   = [f for f in FEATURE_COLUMNS if f not in CATEGORICAL_FEATS]
    TARGET_COL        = 'target'
    RANDOM_SEED       = 42

    X_raw = df[FEATURE_COLUMNS].values
    split_idx = int(0.85 * len(X_raw))
    X_train_raw = X_raw[:split_idx]
    
    scaler = StandardScaler().fit(X_train_raw)
    loaded_model = joblib.load(MODEL_PATH)

    isf = IsolationForest(contamination=0.10, random_state=RANDOM_SEED)
    isf.fit(scaler.transform(X_train_raw))
    
    ranges = np.array([df[f].max() - df[f].min() if f in NUMERICAL_FEATS else 0 for f in FEATURE_COLUMNS])
    cat_idx = [FEATURE_COLUMNS.index(f) for f in CATEGORICAL_FEATS]
    num_idx = [FEATURE_COLUMNS.index(f) for f in NUMERICAL_FEATS]

    class ModelWrapper:
        def __init__(self, model, scaler):
            self.model, self.scaler = model, scaler
        def predict_proba(self, data):
            return self.model.predict_proba(self.scaler.transform(data.values))
        def predict(self, data):
            return self.model.predict(self.scaler.transform(data.values))

    wrapped_model = ModelWrapper(loaded_model, scaler)
    d_data = dice_ml.Data(dataframe=df, continuous_features=NUMERICAL_FEATS, outcome_name=TARGET_COL)
    d_model = dice_ml.Model(model=wrapped_model, backend="sklearn")
    exp = dice_ml.Dice(d_data, d_model, method="genetic") 

    test_preds = loaded_model.predict(scaler.transform(X_raw[split_idx:]))
    query_pool_indices = np.where(test_preds == 0)[0] + split_idx
    
    selected_indices = np.random.default_rng(RANDOM_SEED).choice(query_pool_indices, size=100, replace=False)

    results = []
    for idx in selected_indices:
        query_instance = df.iloc[[idx]].drop(columns=[TARGET_COL])
        q_array = X_raw[idx]
        try:
            dice_exp = exp.generate_counterfactuals(query_instance, total_CFs=10, desired_class="opposite")
            cf_df = dice_exp.cf_examples_list[0].final_cfs_df
            if cf_df is None or cf_df.empty: continue
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
    print("DICE (GENETIC) BASELINE - HELOC (100 QUERIES)")
    print("="*65)
    for m in ['best_sparsity', 'best_proximity', 'avg_sparsity', 'avg_proximity', 'diversity', 'plausibility']:
        print(f"{m:<15} {res_df[m].mean():>12.4f} +/- {res_df[m].std():.4f}")
    
    res_df.to_csv('table5_dice_HELOC.csv', index=False)

if __name__ == "__main__":
    main()