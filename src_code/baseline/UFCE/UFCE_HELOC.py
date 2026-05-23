"""
UFCE Baseline Evaluation — HELOC Dataset
Suffian et al. (2024). User Feedback-Based Counterfactual Explanations.
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import joblib
from itertools import combinations
from scipy.spatial.distance import hamming
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

DATA_PATH  = '/Users/saikat/Desktop/Summer 2025/CFNet_AAAI_26/HELOC Dataset/AISTATS HELOC Dataset/heloc_dataset_v1.csv'
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
IMMUTABLE_FEATS   = []
TARGET_COL        = 'target'

NUM_QUERIES       = 100
RANDOM_SEED       = 42
ISO_CONTAMINATION = 0.10

K_NEIGHBORS = 5
ALPHA_STEPS = np.linspace(0.1, 1.0, 10)


def build_ranges(df):
    ranges = np.zeros(len(FEATURE_COLUMNS))
    for i, f in enumerate(FEATURE_COLUMNS):
        if f in NUMERICAL_FEATS:
            ranges[i] = df[f].max() - df[f].min()
    return ranges


def heom_distance(query, candidate, cat_idx, num_idx, ranges):
    d = 0.0
    for i in cat_idx:
        d += 1.0 if query[i] != candidate[i] else 0.0
    for i in num_idx:
        rng = ranges[i] if ranges[i] != 0 else 1.0
        d += abs(query[i] - candidate[i]) / rng
    return d


def sparsity_ratio(q, c):
    return float(np.sum(np.array(q) != np.array(c))) / len(q)


def _filter_for_diversity(arr, q_raw, nun, cat_idx, immutable_idx):
    keep = []
    for i in range(len(arr)):
        if i in immutable_idx:
            continue
        if i in cat_idx and q_raw[i] == nun[i]:
            continue
        keep.append(arr[i])
    return np.array(keep)


def compute_diversity(candidates, q_raw, nun, cat_idx, immutable_idx):
    if len(candidates) < 2:
        return 0.0
    filtered = [_filter_for_diversity(cf, q_raw, nun, cat_idx, immutable_idx)
                for cf in candidates]
    filtered = [f for f in filtered if len(f) > 0]
    if len(filtered) < 2:
        return 0.0
    pairs = list(combinations(filtered, 2))
    dists = [hamming(p[0], p[1]) for p in pairs]
    return float(np.mean(dists)) * 100.0


def compute_plausibility(candidates, isf, scaler):
    if len(candidates) == 0:
        return 0.0
    arr = np.array(candidates)
    preds = isf.predict(scaler.transform(arr))
    return float(np.mean(preds == 1)) * 100.0


def ufce_generate(q_raw, feasible_neighbors, model, scaler,
                  cat_idx, num_idx, mutable_idx, ranges):
    orig_pred = model.predict(scaler.transform(q_raw.reshape(1, -1)))[0]
    cat_set = set(cat_idx)
    mutable_set = set(mutable_idx)
    candidates = []
    for neighbor in feasible_neighbors:
        for alpha in ALPHA_STEPS:
            cf = q_raw.copy().astype(float)
            for f in mutable_set:
                if f in cat_set:
                    if alpha > 0:
                        cf[f] = neighbor[f]
                else:
                    cf[f] = q_raw[f] + alpha * (neighbor[f] - q_raw[f])
            pred = model.predict(scaler.transform(cf.reshape(1, -1)))[0]
            if pred != orig_pred:
                candidates.append(cf.copy())
    return candidates


def dedup(candidates):
    seen = set()
    out = []
    for cf in candidates:
        key = tuple(np.round(cf, 6).tolist())
        if key not in seen:
            seen.add(key)
            out.append(cf)
    return out


def main():
    print("Loading data, model, and IsolationForest...")
    data = pd.read_csv(DATA_PATH)
    data['target'] = data['target'].map({'Bad': 0, 'Good': 1})

    X_all = data[FEATURE_COLUMNS].values.astype(float)
    split_idx = int(0.85 * len(X_all))
    X_train = X_all[:split_idx]

    scaler = StandardScaler()
    scaler.fit(X_train)
    model = joblib.load(MODEL_PATH)
    isf = IsolationForest(contamination=ISO_CONTAMINATION,
                          random_state=RANDOM_SEED)
    isf.fit(scaler.transform(X_train))

    cat_idx       = [FEATURE_COLUMNS.index(f) for f in CATEGORICAL_FEATS]
    num_idx       = [FEATURE_COLUMNS.index(f) for f in NUMERICAL_FEATS]
    immutable_idx = [FEATURE_COLUMNS.index(f) for f in IMMUTABLE_FEATS]
    mutable_idx   = [i for i in range(len(FEATURE_COLUMNS))
                     if i not in immutable_idx]
    ranges = build_ranges(data)

    all_X_sc = scaler.transform(X_all)
    all_preds = model.predict(all_X_sc)
    pos_mask = (all_preds == 1)
    neg_mask = (all_preds == 0)

    print(f"Dataset size              : {len(X_all)}")
    print(f"Model predicts positive   : {pos_mask.sum()} ({100*pos_mask.mean():.1f}%)")
    print(f"Model predicts negative   : {neg_mask.sum()} ({100*neg_mask.mean():.1f}%)\n")

    pos_pool_raw    = X_all[pos_mask]
    pos_pool_scaled = all_X_sc[pos_mask]
    neg_pool_raw    = X_all[neg_mask]

    if len(pos_pool_raw) == 0:
        print("ERROR: no model-predicted positives.")
        return

    k_actual = min(K_NEIGHBORS, len(pos_pool_raw))
    nbrs = NearestNeighbors(n_neighbors=k_actual, metric='euclidean')
    nbrs.fit(pos_pool_scaled)
    print(f"k-NN feasible region index built  (k={k_actual}, pool size={len(pos_pool_raw)})")

    rng = np.random.default_rng(RANDOM_SEED)
    n_take = min(NUM_QUERIES, len(neg_pool_raw))
    q_idx = rng.choice(len(neg_pool_raw), size=n_take, replace=False)
    queries_raw = neg_pool_raw[q_idx]

    print(f"\nRunning UFCE on {n_take} queries (k={k_actual}, {len(ALPHA_STEPS)} alpha steps)...")
    print("NOTE: HELOC has 23 features and MLP predict is slow; "
          "expect 20-40 min.\n")

    rows, n_no_cf = [], 0
    for qi in range(n_take):
        if qi % 10 == 0 and qi > 0:
            print(f"  ...{qi}/{n_take}")
        q_raw = queries_raw[qi]
        q_scaled = scaler.transform(q_raw.reshape(1, -1))
        _, nn_idx = nbrs.kneighbors(q_scaled, n_neighbors=k_actual)
        feasible_neighbors = pos_pool_raw[nn_idx[0]]

        cands = ufce_generate(q_raw, feasible_neighbors, model, scaler,
                              cat_idx, num_idx, mutable_idx, ranges)
        if not cands:
            n_no_cf += 1
            continue
        cands = dedup(cands)

        proxs = [heom_distance(q_raw, cf, cat_idx, num_idx, ranges) for cf in cands]
        spars = [sparsity_ratio(q_raw, cf) for cf in cands]
        best_prox = float(min(proxs))
        best_spar = float(min(spars))
        avg_prox  = float(np.mean(proxs))
        avg_spar  = float(np.mean(spars))

        nun  = feasible_neighbors[0]
        div  = compute_diversity(cands, q_raw, nun, cat_idx, immutable_idx)
        plau = compute_plausibility(cands, isf, scaler)

        rows.append({
            'query_idx': qi,
            'n_candidates': len(cands),
            'best_sparsity': best_spar,
            'best_proximity': best_prox,
            'avg_sparsity': avg_spar,
            'avg_proximity': avg_prox,
            'diversity_pct': div,
            'plausibility_pct': plau,
        })

    df = pd.DataFrame(rows)
    coverage = len(df) / n_take * 100.0
    print(f"\nQueries with valid CFs     : {len(df)}/{n_take} (Coverage = {coverage:.1f}%)")
    print(f"Queries with no valid CFs  : {n_no_cf}/{n_take}")
    if len(df) > 0:
        print(f"Avg candidates per query   : {df['n_candidates'].mean():.1f}\n")
    if len(df) == 0:
        return

    print("=" * 70)
    print("UFCE Results — HELOC (100 queries)")
    print(f"Reported as mean +/- std  |  Coverage = {coverage:.1f}%")
    print("=" * 70)
    metrics = [
        ('best_sparsity',    'Spar.'),
        ('best_proximity',   'Prox.'),
        ('avg_sparsity',     'Avg Spar.'),
        ('avg_proximity',    'Avg Prox.'),
        ('diversity_pct',    'Div. (%)'),
        ('plausibility_pct', 'Plaus. (%)'),
    ]
    print(f"\n{'Metric':<15} {'Mean':>10} {'Std':>10}   {'Formatted':>20}")
    print("-" * 60)
    for col, label in metrics:
        m, s = df[col].mean(), df[col].std()
        print(f"{label:<15} {m:>10.4f} {s:>10.4f}   {m:.2f} +/- {s:.2f}")

    df.to_csv('table5_ufce_HELOC.csv', index=False)
    print(f"\nPer-query results saved to: table5_ufce_HELOC.csv")


if __name__ == "__main__":
    main()