"""
NICE Evaluation for Table 5 (Main Results Table)
Dataset: GMC

NICE Algorithm Reference:
  Brughmans, D., Leyman, P., & Martens, D. (2024).
  NICE: an algorithm for nearest instance counterfactual explanations.
  Data Mining and Knowledge Discovery, 38(5), 2665-2703.

NICE Strategy Used: PROXIMITY
  - Finds the Nearest Unlike Neighbor (NUN) as the starting counterfactual.
  - Greedily reverts features back toward the query (least HEOM-contributing
    features first) as long as the prediction flip is preserved.
  - Returns one counterfactual per query.

Protocol (mirrors SPICE_GMC.py exactly):
  - 100 random queries (seed=42)
  - Same ANN index (Falconn++ style LSH) as SPICE for NUN search
  - Same HEOM distance metric
  - Same IsolationForest (contamination=0.10, trained on X_train)
  - Same train/test split logic
  - Same query pool (model-predicted negatives)
  - Same NUN pool  (model-predicted positives)
  - NICE returns ONE counterfactual per query:
      Avg Sparsity / Avg Proximity / Diversity are reported as "-"
  - Reported as mean +/- std over valid queries
"""

import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import normalize, StandardScaler
from sklearn.model_selection import train_test_split

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_PATH  = '/Users/saikat/Desktop/Summer 2025/CFNet_AAAI_26/GMC Dataset/AISTATS GMC Dataset/preprocessed_credit_data.csv'
MODEL_PATH = '/Users/saikat/Desktop/Summer 2025/CFNet_AAAI_26/GMC Dataset/AISTATS GMC Dataset/best_credit_xgb_model.pkl'
TARGET_COL = 'SeriousDlqin2yrs'

# ── Feature schema (identical to SPICE_GMC.py) ───────────────────────────────
SEMANTIC_WEIGHTS = np.array([0.168549, 0.080057, 0.064094, 0.032998, 0.018477,
                              0.012226, 0.014331, 0.006502, 0.006534, 0.002307])

FEATURE_COLUMNS   = ['RevolvingUtilizationOfUnsecuredLines', 'age',
                     'NumberOfTime30-59DaysPastDueNotWorse', 'DebtRatio',
                     'MonthlyIncome', 'NumberOfOpenCreditLinesAndLoans',
                     'NumberOfTimes90DaysLate', 'NumberRealEstateLoansOrLines',
                     'NumberOfTime60-89DaysPastDueNotWorse', 'NumberOfDependents']
CATEGORICAL_FEATS = ['NumberOfDependents', 'NumberOfTime30-59DaysPastDueNotWorse',
                     'NumberRealEstateLoansOrLines']
NUMERICAL_FEATS   = ['RevolvingUtilizationOfUnsecuredLines', 'age', 'DebtRatio',
                     'MonthlyIncome', 'NumberOfOpenCreditLinesAndLoans',
                     'NumberOfTimes90DaysLate',
                     'NumberOfTime60-89DaysPastDueNotWorse']
IMMUTABLE_FEATS = ['age', 'DebtRatio']

# ── ANN hyper-parameters (identical to SPICE_GMC.py) ─────────────────────────
NUM_PROJECTIONS = 15
NUM_TABLES      = 10
ALPHA           = 0.25
I_PROBES        = 1
Q_PROBES        = 6

# ── Evaluation settings (identical to SPICE_GMC.py) ──────────────────────────
NUM_QUERIES       = 100
RANDOM_SEED       = 42
ISO_CONTAMINATION = 0.10


# ═════════════════════════════════════════════════════════════════════════════
# Distance / metric helpers
# ═════════════════════════════════════════════════════════════════════════════

def heom_distance(query, candidate, cat_idx, num_idx, ranges):
    """Heterogeneous Euclidean-Overlap Metric (identical to SPICE_GMC.py)."""
    d = 0.0
    for i in cat_idx:
        d += 1.0 if query[i] != candidate[i] else 0.0
    for i in num_idx:
        rng = ranges[i] if ranges[i] != 0 else 1
        d += abs(query[i] - candidate[i]) / rng
    return d


def sparsity_ratio(q, c):
    return float(np.sum(q != c)) / len(q)


# ═════════════════════════════════════════════════════════════════════════════
# Data / model loading (identical to SPICE_GMC.py)
# ═════════════════════════════════════════════════════════════════════════════

def load_everything():
    data = pd.read_csv(DATA_PATH).dropna(subset=[TARGET_COL])
    X = data[FEATURE_COLUMNS].values
    y = data[TARGET_COL].values

    # Replicate the exact two-stage split from SPICE_GMC.py
    X_train, _, _, _ = train_test_split(X, y, test_size=0.15, random_state=0)
    X_train, _, _, _ = train_test_split(X_train, y[:len(X_train)],
                                        test_size=0.10, random_state=0)

    scaler = StandardScaler()
    scaler.fit(X_train)

    model = joblib.load(MODEL_PATH)

    isf = IsolationForest(contamination=ISO_CONTAMINATION,
                          random_state=RANDOM_SEED)
    isf.fit(scaler.transform(X_train))

    cat_idx       = [FEATURE_COLUMNS.index(f) for f in CATEGORICAL_FEATS]
    num_idx       = [FEATURE_COLUMNS.index(f) for f in NUMERICAL_FEATS]
    immutable_idx = [FEATURE_COLUMNS.index(f) for f in IMMUTABLE_FEATS]

    ranges = np.zeros(len(FEATURE_COLUMNS))
    for i, f in enumerate(FEATURE_COLUMNS):
        if f in NUMERICAL_FEATS:
            ranges[i] = data[f].max() - data[f].min()

    return data, model, scaler, isf, cat_idx, num_idx, immutable_idx, ranges


# ═════════════════════════════════════════════════════════════════════════════
# ANN index (identical to SPICE_GMC.py)
# ═════════════════════════════════════════════════════════════════════════════

def build_lsh_index(weighted_neighbors, dim):
    np.random.seed(RANDOM_SEED)
    projections = [np.random.randn(NUM_PROJECTIONS, dim)
                   for _ in range(NUM_TABLES)]
    tables = [{} for _ in range(NUM_TABLES)]
    for t_idx in range(NUM_TABLES):
        proj = projections[t_idx]
        for p_idx, point in enumerate(weighted_neighbors):
            proj_vals  = np.dot(proj, point)
            sorted_idx = np.argsort(-np.abs(proj_vals))
            for i in range(I_PROBES):
                h = int(sorted_idx[i])
                tables[t_idx].setdefault(h, []).append(p_idx)
    return projections, tables


def query_lsh(query_norm, projections, tables, weighted_neighbors):
    candidates = set()
    for t_idx, table in enumerate(tables):
        proj_vectors = projections[t_idx]
        proj_query   = np.dot(proj_vectors, query_norm)
        sorted_idx_q = np.argsort(-np.abs(proj_query))
        for h in sorted_idx_q[:1 + Q_PROBES]:
            bucket = table.get(int(h), [])
            if not bucket:
                continue
            scored = []
            for idx in bucket:
                cand_vec  = weighted_neighbors[idx]
                proj_cand = np.dot(proj_vectors, cand_vec)
                scored.append((idx, np.max(np.abs(proj_cand))))
            scored.sort(key=lambda x: x[1], reverse=True)
            num_keep = max(1, int(ALPHA * len(scored)))
            candidates.update([s[0] for s in scored[:num_keep]])
    return list(candidates)


def find_weighted_nun(q_raw, neighbors_raw, weighted_neighbors, q_norm,
                     projections, tables, cat_idx, num_idx, ranges):
    """Return the feature-weighted NUN using the same ANN as SPICE."""
    cand_idx = query_lsh(q_norm, projections, tables, weighted_neighbors)
    if not cand_idx:
        return None
    best_idx, best_dist = None, float('inf')
    for i in cand_idx:
        d = heom_distance(q_raw, neighbors_raw[i], cat_idx, num_idx, ranges)
        if d < best_dist:
            best_dist, best_idx = d, i
    return neighbors_raw[best_idx] if best_idx is not None else None


# ═════════════════════════════════════════════════════════════════════════════
# NICE core
# ═════════════════════════════════════════════════════════════════════════════

def nice_counterfactual(q_raw, nun, model, scaler, cat_idx, num_idx,
                        immutable_idx, ranges):
    """
    NICE (Proximity strategy):
      1. Start from the NUN — always a valid counterfactual by definition.
      2. Rank mutable features by their individual HEOM contribution
         in ascending order (least-different features first).
      3. Greedily revert each feature back to the query value.
         Accept the reversion only if the prediction flip is preserved.
      4. Return the resulting sparse, proximate counterfactual.

    Ranking by HEOM contribution (ascending) means we first try to revert
    features that cost the least distance, naturally minimising sparsity
    while maintaining proximity — the NICE proximity strategy.
    """
    orig_pred = model.predict(
        scaler.transform(q_raw.reshape(1, -1)))[0]

    cf = nun.copy().astype(float)

    mutable_idx = [i for i in range(len(FEATURE_COLUMNS))
                   if i not in immutable_idx]

    # Compute per-feature HEOM contribution using actual ranges
    def heom_contrib(i):
        if i in cat_idx:
            return 1.0 if q_raw[i] != nun[i] else 0.0
        rng = ranges[i] if ranges[i] != 0 else 1.0
        return abs(float(q_raw[i]) - float(nun[i])) / rng

    # Sort ascending: revert cheapest features first
    ranked = sorted(mutable_idx, key=heom_contrib)

    for feat_idx in ranked:
        candidate = cf.copy()
        candidate[feat_idx] = q_raw[feat_idx]      # revert to query value
        pred = model.predict(
            scaler.transform(candidate.reshape(1, -1)))[0]
        if pred != orig_pred:                       # flip still holds
            cf = candidate

    return cf


def compute_plausibility_single(cf, isf, scaler):
    arr_scaled = scaler.transform(cf.reshape(1, -1))
    pred = isf.predict(arr_scaled)[0]
    return 1.0 if pred == 1 else 0.0


# ═════════════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════════════

def main():
    (data, model, scaler, isf,
     cat_idx, num_idx, immutable_idx, ranges) = load_everything()

    all_X        = data[FEATURE_COLUMNS].values.astype(float)
    all_X_scaled = scaler.transform(all_X)
    all_preds    = model.predict(all_X_scaled)

    pos_mask = (all_preds == 1)
    neg_mask = (all_preds == 0)

    print(f"Dataset size: {len(data)}")
    print(f"Model predicts as positive: {pos_mask.sum()} "
          f"({100*pos_mask.mean():.1f}%)")
    print(f"Model predicts as negative: {neg_mask.sum()} "
          f"({100*neg_mask.mean():.1f}%)")
    print(f"(Ground-truth: {(data[TARGET_COL]==1).sum()} positive, "
          f"{(data[TARGET_COL]==0).sum()} negative)\n")

    # ── Build ANN index on positive (NUN) pool ────────────────────────────────
    neighbors_raw      = all_X[pos_mask]
    norm_neighbors     = normalize(neighbors_raw)
    weighted_neighbors = norm_neighbors * SEMANTIC_WEIGHTS
    projections, tables = build_lsh_index(weighted_neighbors,
                                          len(FEATURE_COLUMNS))

    # ── Sample query pool (identical to SPICE_GMC.py) ────────────────────────
    query_pool  = all_X[neg_mask]
    n_take      = min(NUM_QUERIES, len(query_pool))
    rng         = np.random.default_rng(RANDOM_SEED)
    query_idx   = rng.choice(len(query_pool), size=n_take, replace=False)
    queries_raw = query_pool[query_idx]
    queries_norm_for_hash = normalize(queries_raw)

    print(f"Running NICE on {n_take} queries (Proximity strategy)...")

    rows      = []
    n_no_cand = 0

    for q_idx in range(n_take):
        if q_idx % 10 == 0 and q_idx > 0:
            print(f"  ...{q_idx}/{n_take}")

        q_raw  = queries_raw[q_idx]
        q_hash = queries_norm_for_hash[q_idx]

        # Step 1: find weighted NUN via same ANN as SPICE
        nun = find_weighted_nun(q_raw, neighbors_raw, weighted_neighbors,
                                q_hash, projections, tables,
                                cat_idx, num_idx, ranges)
        if nun is None:
            n_no_cand += 1
            continue

        # Step 2: NICE greedy reversion toward query
        cf = nice_counterfactual(q_raw, nun, model, scaler,
                                 cat_idx, num_idx, immutable_idx, ranges)

        # Step 3: verify validity — fall back to raw NUN if broken
        orig_pred = model.predict(
            scaler.transform(q_raw.reshape(1, -1)))[0]
        cf_pred   = model.predict(
            scaler.transform(cf.reshape(1, -1)))[0]
        if cf_pred == orig_pred:
            cf = nun.copy().astype(float)

        best_spar = sparsity_ratio(q_raw, cf)
        best_prox = heom_distance(q_raw, cf, cat_idx, num_idx, ranges)
        plau      = compute_plausibility_single(cf, isf, scaler) * 100.0

        rows.append({
            'query_idx':        q_idx,
            'best_sparsity':    best_spar,
            'best_proximity':   best_prox,
            'plausibility_pct': plau,
        })

    df = pd.DataFrame(rows)
    print(f"\nQueries with valid counterfactuals: {len(df)}/{n_take}")
    print(f"Queries with no NUN found:          {n_no_cand}/{n_take}")

    if len(df) == 0:
        print("ERROR: no queries produced counterfactuals.")
        return

    print()
    print("=" * 70)
    print("NICE Results - GMC (Proximity strategy, 100 queries)")
    print("Reported as mean +/- std")
    print("Note: NICE generates ONE counterfactual per query.")
    print("      Avg Sparsity / Avg Proximity / Diversity = '-' (N/A)")
    print("=" * 70)

    cols   = ['best_sparsity', 'best_proximity', 'plausibility_pct']
    labels = ['Spar.', 'Prox.', 'Plaus. (%)']

    print(f"{'Metric':<15} {'Mean':>12} {'Std':>12} {'Formatted':>20}")
    print("-" * 62)
    for c, l in zip(cols, labels):
        m = df[c].mean()
        s = df[c].std()
        formatted = f"{m:.2f} +/- {s:.2f}"
        print(f"{l:<15} {m:>12.4f} {s:>12.4f} {formatted:>20}")

    print(f"\nAvg Spar.  : -")
    print(f"Avg Prox.  : -")
    print(f"Div. (%)   : -")

    df.to_csv('table5_nice_GMC.csv', index=False)
    print(f"\nPer-query results saved to: table5_nice_GMC.csv")


if __name__ == "__main__":
    main()