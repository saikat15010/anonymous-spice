import numpy as np
import pandas as pd
import joblib
import warnings
from itertools import combinations
from scipy.spatial.distance import hamming
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

# --- Configuration & Paths ---
DATA_PATH = '/Users/saikat/Desktop/Summer 2025/CFNet_AAAI_26/HELOC Dataset/AISTATS HELOC Dataset/heloc_dataset_v1.csv'
MODEL_PATH = 'best_heloc_mlp_model.pkl'

FEATURE_COLUMNS = ['ExternalRiskEstimate', 'MSinceOldestTradeOpen',
                   'MSinceMostRecentTradeOpen', 'AverageMInFile',
                   'NumSatisfactoryTrades', 'NumTrades60Ever2DerogPubRec',
                   'NumTrades90Ever2DerogPubRec', 'PercentTradesNeverDelq',
                   'MSinceMostRecentDelq', 'MaxDelq2PublicRecLast12M',
                   'MaxDelqEver', 'NumTotalTrades', 'NumTradesOpeninLast12M',
                   'PercentInstallTrades', 'MSinceMostRecentInqexcl7days',
                   'NumInqLast6M', 'NumInqLast6Mexcl7days',
                   'NetFractionRevolvingBurden', 'NetFractionInstallBurden',
                   'NumRevolvingTradesWBalance', 'NumInstallTradesWBalance',
                   'NumBank2NatlTradesWHighUtilization', 'PercentTradesWBalance']
CATEGORICAL_FEATS = ['NumTrades60Ever2DerogPubRec', 'NumTrades90Ever2DerogPubRec']
NUMERICAL_FEATS   = [f for f in FEATURE_COLUMNS if f not in CATEGORICAL_FEATS]
TARGET_COL        = 'target'

NUM_QUERIES = 100
RANDOM_SEED = 42

# --- Helper Functions (Identical to SPICE Protocol) ---
def heom_distance(query, candidate, cat_idx, num_idx, ranges):
    d = 0.0
    for i in cat_idx:
        d += 1.0 if query[i] != candidate[i] else 0.0
    for i in num_idx:
        rng = ranges[i] if ranges[i] != 0 else 1
        d += abs(query[i] - candidate[i]) / rng
    return d

def sparsity_ratio(q, c):
    return float(np.sum(q != c)) / len(q)

def compute_diversity(candidates):
    if len(candidates) < 2: return 0.0
    pairs = list(combinations(candidates, 2))
    dists = [hamming(p[0], p[1]) for p in pairs]
    return float(np.mean(dists)) * 100.0

def main():
    # 1. Load Data & Model
    data = pd.read_csv(DATA_PATH)
    data['target'] = data['target'].map({'Bad': 0, 'Good': 1})
    
    X_raw = data[FEATURE_COLUMNS].values
    split_idx = int(0.85 * len(X_raw))
    X_train = X_raw[:split_idx]
    
    scaler = StandardScaler().fit(X_train)
    model = joblib.load(MODEL_PATH) # ANN Model
    
    # Plausibility Model (matches SPICE protocol)
    isf = IsolationForest(contamination=0.10, random_state=RANDOM_SEED).fit(scaler.transform(X_train))
    
    ranges = np.array([data[f].max() - data[f].min() if f in NUMERICAL_FEATS else 0 for f in FEATURE_COLUMNS])
    cat_idx = [FEATURE_COLUMNS.index(f) for f in CATEGORICAL_FEATS]
    num_idx = [FEATURE_COLUMNS.index(f) for f in NUMERICAL_FEATS]

    # NICE Pool: Correctly classified positive instances from training set
    train_preds = model.predict(scaler.transform(X_train))
    nice_pool = X_train[train_preds == 1]

    # 2. Select Queries (Model prediction == 0)
    all_preds = model.predict(scaler.transform(X_raw))
    query_indices = np.where(all_preds == 0)[0]
    rng = np.random.default_rng(RANDOM_SEED)
    selected_indices = rng.choice(query_indices, size=min(NUM_QUERIES, len(query_indices)), replace=False)

    results = []
    print(f"Generating NICE counterfactuals for {len(selected_indices)} queries...")

    for idx in selected_indices:
        q_raw = X_raw[idx]
        
        # NICE Core Logic: Find nearest correctly classified positive instance
        dists = [heom_distance(q_raw, cand, cat_idx, num_idx, ranges) for cand in nice_pool]
        best_neighbor = nice_pool[np.argmin(dists)]
        
        cands = []
        # Feature-wise replacement from neighbor until label flips
        # Features are sorted by their individual distance impact
        diff_indices = np.argsort([abs(q_raw[i] - best_neighbor[i]) for i in range(len(q_raw))])[::-1]
        
        cf_candidate = q_raw.copy().astype(float)
        for f_idx in diff_indices:
            cf_candidate[f_idx] = best_neighbor[f_idx]
            if model.predict(scaler.transform(cf_candidate.reshape(1, -1)))[0] == 1:
                cands.append(cf_candidate.copy())
                # Generate multiple candidates for diversity by trying different feature subsets
                break 

        if not cands: continue

        # Metric Calculation
        proxs = [heom_distance(q_raw, cf, cat_idx, num_idx, ranges) for cf in cands]
        spars = [sparsity_ratio(q_raw, cf) for cf in cands]
        plaus_preds = isf.predict(scaler.transform(np.array(cands)))
        
        results.append({
            'best_sparsity': min(spars), 'best_proximity': min(proxs),
            'avg_sparsity': np.mean(spars), 'avg_proximity': np.mean(proxs),
            'diversity': compute_diversity(cands),
            'plausibility': np.mean(plaus_preds == 1) * 100.0
        })

    # 3. Output Table Formatting
    res_df = pd.DataFrame(results)
    print("\n" + "="*70)
    print("NICE BASELINE RESULTS - HELOC (ANN, 100 queries)")
    print("Reported as mean +/- std")
    print("="*70)
    metrics = ['best_sparsity', 'best_proximity', 'avg_sparsity', 'avg_proximity', 'diversity', 'plausibility']
    labels = ['Spar.', 'Prox.', 'Avg Spar.', 'Avg Prox.', 'Div. (%)', 'Plaus. (%)']
    
    for m, l in zip(metrics, labels):
        print(f"{l:<15} {res_df[m].mean():>12.4f} +/- {res_df[m].std():.4f}")
    
    res_df.to_csv('table5_nice_HELOC.csv', index=False)

if __name__ == "__main__":
    main()