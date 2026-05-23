"""
NICE Evaluation for Table 5 (Main Results Table)
Dataset: Graduate Admission

NICE Algorithm Reference:
  Brughmans, D., Leyman, P., & Martens, D. (2024).
  NICE: an algorithm for nearest instance counterfactual explanations.
  Data Mining and Knowledge Discovery, 38(5), 2665-2703.

Key differences from SPICE:
  - NICE uses HEOM-cost-ordered feature substitution (cheapest feature first),
    not feature-importance ordering.
  - NICE produces a single best counterfactual per query, not a candidate set.
  - NICE does NOT apply the iterative numerical perturbation step that SPICE uses.
  - All other infrastructure (ANN, HEOM, IsolationForest, LSH index) is
    identical to SPICE_GraduateAdmission.py for a fair comparison.

Protocol:
  - 100 random queries (seed=42), same query pool as SPICE
  - Same NUN pool (model-predicted positive instances)
  - Same LSH index parameters: L=10 tables, p=15 projections, alpha=0.50
  - Isolation Forest: trained on X_train, contamination=0.10
  - Reported as mean +/- std over queries with a valid counterfactual
"""

import numpy as np
import pandas as pd
import joblib
from itertools import combinations
from scipy.spatial.distance import hamming
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import normalize, StandardScaler

# ── Paths ──────────────────────────────────────────────────────────────────────
TRAIN_PATH = '/Users/saikat/Desktop/Summer 2025/CFNet_AAAI_26/Graduate Admission Dataset/admission_train.csv'
TEST_PATH  = '/Users/saikat/Desktop/Summer 2025/CFNet_AAAI_26/Graduate Admission Dataset/admission_test.csv'
MODEL_PATH = '/Users/saikat/Desktop/Summer 2025/CFNet_AAAI_26/Graduate Admission Dataset/AISTATS Graduate Admission/best_admission_rf_model.pkl'

# ── Feature schema (must match SPICE exactly) ──────────────────────────────────
SEMANTIC_WEIGHTS = np.array([0.220184, 0.149150, 0.088845, 0.049897,
                              0.040006, 0.043933, 0.006985])

FEATURE_COLUMNS   = ['GRE Score', 'TOEFL Score', 'University Rating',
                     'SOP', 'LOR', 'CGPA', 'Research']
CATEGORICAL_FEATS = ['University Rating', 'Research']
NUMERICAL_FEATS   = ['GRE Score', 'TOEFL Score', 'SOP', 'LOR', 'CGPA']
IMMUTABLE_FEATS   = []

# ── ANN hyper-parameters (identical to SPICE) ──────────────────────────────────
NUM_PROJECTIONS = 15
NUM_TABLES      = 10
ALPHA           = 0.50
I_PROBES        = 1
Q_PROBES        = 6

# ── Evaluation protocol ────────────────────────────────────────────────────────
NUM_QUERIES       = 100
RANDOM_SEED       = 42
ISO_CONTAMINATION = 0.10


# ══════════════════════════════════════════════════════════════════════════════
# Shared utilities  (identical to SPICE so results are directly comparable)
# ══════════════════════════════════════════════════════════════════════════════

def heom_distance(query, candidate, cat_idx, num_idx, ranges):
    """Heterogeneous Euclidean Overlap Metric (HEOM)."""
    d = 0.0
    for i in cat_idx:
        d += 0.0 if query[i] == candidate[i] else 1.0
    for i in num_idx:
        rng = ranges[i] if ranges[i] != 0 else 1.0
        d += abs(query[i] - candidate[i]) / rng
    return d


def per_feature_heom(query, candidate, cat_idx, num_idx, ranges):
    """
    Return a vector of per-feature HEOM costs.
    Used by NICE to order features by substitution cost (cheapest first).
    """
    costs = np.zeros(len(query))
    for i in cat_idx:
        costs[i] = 0.0 if query[i] == candidate[i] else 1.0
    for i in num_idx:
        rng = ranges[i] if ranges[i] != 0 else 1.0
        costs[i] = abs(query[i] - candidate[i]) / rng
    return costs


def sparsity_ratio(q, c):
    return float(np.sum(q != c)) / len(q)


def compute_plausibility(candidates, isf, scaler):
    if not candidates:
        return 0.0
    arr = np.array(candidates)
    arr_scaled = scaler.transform(arr)
    preds = isf.predict(arr_scaled)
    return float(np.mean(preds == 1)) * 100.0


# ══════════════════════════════════════════════════════════════════════════════
# Data & model loading
# ══════════════════════════════════════════════════════════════════════════════

def load_everything():
    train    = pd.read_csv(TRAIN_PATH)
    test     = pd.read_csv(TEST_PATH)
    combined = pd.concat([train, test], ignore_index=True)

    split_idx  = int(0.85 * len(combined))
    X_all      = combined.drop(columns=['target']).values
    X_train    = X_all[:split_idx]

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
            ranges[i] = combined[f].max() - combined[f].min()

    return combined, model, scaler, isf, cat_idx, num_idx, immutable_idx, ranges


# ══════════════════════════════════════════════════════════════════════════════
# ANN index  (identical to SPICE)
# ══════════════════════════════════════════════════════════════════════════════

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
        proj_vectors  = projections[t_idx]
        proj_query    = np.dot(proj_vectors, query_norm)
        sorted_idx_q  = np.argsort(-np.abs(proj_query))
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
    """
    Identical to SPICE: use the feature-weighted ANN to find the NUN.
    Returns the raw (unweighted, unscaled) NUN feature vector.
    """
    cand_idx = query_lsh(q_norm, projections, tables, weighted_neighbors)
    if not cand_idx:
        return None
    best_idx, best_dist = None, float('inf')
    for i in cand_idx:
        d = heom_distance(q_raw, neighbors_raw[i], cat_idx, num_idx, ranges)
        if d < best_dist:
            best_dist, best_idx = d, i
    return neighbors_raw[best_idx] if best_idx is not None else None


# ══════════════════════════════════════════════════════════════════════════════
# NICE counterfactual generation
# ══════════════════════════════════════════════════════════════════════════════

def nice_generate(q_raw, nun, model, scaler, cat_idx, num_idx,
                  immutable_idx, ranges):
    """
    Core NICE algorithm.

    Strategy
    --------
    1. Compute the per-feature HEOM cost between the query and the NUN.
    2. Rank mutable features in ascending order of cost
       (cheapest substitution first — the defining characteristic of NICE).
    3. Greedily substitute features from the NUN into a working copy of the
       query, one at a time, and check for a prediction flip after each step.
    4. Return the first counterfactual that achieves a prediction flip, or
       the full NUN as a guaranteed fallback (100 % coverage).

    Note: NICE does NOT apply the numerical perturbation step used in SPICE.
    It returns exactly one counterfactual per query.
    """
    orig_pred     = model.predict(scaler.transform(q_raw.reshape(1, -1)))[0]
    mutable_idx   = [i for i in range(len(FEATURE_COLUMNS))
                     if i not in immutable_idx]

    # Per-feature costs between query and NUN
    costs         = per_feature_heom(q_raw, nun, cat_idx, num_idx, ranges)

    # Sort mutable features by ascending substitution cost
    feature_order = sorted(mutable_idx, key=lambda i: costs[i])

    cf = q_raw.copy().astype(float)

    for feat_idx in feature_order:
        # Skip features where query already matches NUN (zero-cost, no change)
        if cf[feat_idx] == nun[feat_idx]:
            continue

        cf[feat_idx] = nun[feat_idx]

        pred = model.predict(scaler.transform(cf.reshape(1, -1)))[0]
        if pred != orig_pred:
            return cf   # earliest (sparsest) flip found

    # Fallback: full NUN guarantees coverage
    return nun.copy().astype(float)


# ══════════════════════════════════════════════════════════════════════════════
# Main evaluation loop
# ══════════════════════════════════════════════════════════════════════════════

def main():
    (combined, model, scaler, isf,
     cat_idx, num_idx, immutable_idx, ranges) = load_everything()

    # ── Build NUN pool and query pool from model predictions (same as SPICE) ──
    all_X        = combined[FEATURE_COLUMNS].values.astype(float)
    all_X_scaled = scaler.transform(all_X)
    all_preds    = model.predict(all_X_scaled)

    pos_mask = (all_preds == 1)
    neg_mask = (all_preds == 0)

    print(f"Dataset size                : {len(combined)}")
    print(f"Model predicts as positive  : {pos_mask.sum()} "
          f"({100*pos_mask.mean():.1f}%)")
    print(f"Model predicts as negative  : {neg_mask.sum()} "
          f"({100*neg_mask.mean():.1f}%)")
    print(f"(Ground-truth: {(combined['target']==1).sum()} positive, "
          f"{(combined['target']==0).sum()} negative)\n")

    # ── Build the ANN index on feature-weighted, L2-normalised NUN pool ───────
    neighbors_raw      = all_X[pos_mask]
    norm_neighbors     = normalize(neighbors_raw)
    weighted_neighbors = norm_neighbors * SEMANTIC_WEIGHTS
    projections, tables = build_lsh_index(weighted_neighbors,
                                          len(FEATURE_COLUMNS))

    # ── Sample 100 queries (identical seed / pool as SPICE) ──────────────────
    query_pool           = all_X[neg_mask]
    n_take               = min(NUM_QUERIES, len(query_pool))
    rng                  = np.random.default_rng(RANDOM_SEED)
    query_idx            = rng.choice(len(query_pool), size=n_take, replace=False)
    queries_raw          = query_pool[query_idx]
    queries_norm_for_hash = normalize(queries_raw)

    print(f"Running NICE on {n_take} queries...")

    rows      = []
    n_no_cand = 0

    for q_idx in range(n_take):
        if q_idx % 20 == 0 and q_idx > 0:
            print(f"  ...{q_idx}/{n_take}")

        q_raw  = queries_raw[q_idx]
        q_hash = queries_norm_for_hash[q_idx]

        # Step 1: find the feature-weighted NUN via ANN (same as SPICE)
        nun = find_weighted_nun(q_raw, neighbors_raw, weighted_neighbors,
                                q_hash, projections, tables,
                                cat_idx, num_idx, ranges)
        if nun is None:
            n_no_cand += 1
            continue

        # Step 2: NICE greedy substitution
        cf = nice_generate(q_raw, nun, model, scaler,
                           cat_idx, num_idx, immutable_idx, ranges)

        # Step 3: verify validity (should always hold due to NUN fallback)
        orig_pred = model.predict(scaler.transform(q_raw.reshape(1, -1)))[0]
        cf_pred   = model.predict(scaler.transform(cf.reshape(1, -1)))[0]
        valid     = int(cf_pred != orig_pred)

        if not valid:
            # Extremely rare given NUN fallback; record and skip
            n_no_cand += 1
            continue

        prox  = heom_distance(q_raw, cf, cat_idx, num_idx, ranges)
        spar  = sparsity_ratio(q_raw, cf)
        plau  = compute_plausibility([cf], isf, scaler)

        rows.append({
            'query_idx'       : q_idx,
            'best_sparsity'   : spar,
            'best_proximity'  : prox,
            'plausibility_pct': plau,
            'valid'           : valid,
        })

    df = pd.DataFrame(rows)
    print(f"\nQueries with valid counterfactual : {len(df)}/{n_take}")
    print(f"Queries with no counterfactual    : {n_no_cand}/{n_take}\n")

    if len(df) == 0:
        print("ERROR: no queries produced a counterfactual.")
        return

    # ── Print results table ───────────────────────────────────────────────────
    print("=" * 70)
    print("NICE Results - Graduate Admission (100 queries)")
    print("Reported as mean +/- std")
    print("=" * 70)
    print(f"{'Metric':<20} {'Mean':>10} {'Std':>10} {'Formatted':>22}")
    print("-" * 65)

    metrics = [
        ('best_sparsity',    'Spar.'),
        ('best_proximity',   'Prox.'),
        ('plausibility_pct', 'Plaus. (%)'),
    ]
    for col, label in metrics:
        m = df[col].mean()
        s = df[col].std()
        print(f"{label:<20} {m:>10.4f} {s:>10.4f}   {m:.2f} +/- {s:.2f}")

    print("\nNote: NICE produces one counterfactual per query.")
    print("      Avg. Sparsity, Avg. Proximity, and Diversity are N/A by design.")

    # ── Coverage ──────────────────────────────────────────────────────────────
    coverage = 100.0 * len(df) / n_take
    print(f"\nCoverage: {coverage:.1f}%  ({len(df)}/{n_take} queries)")

    df.to_csv('table5_nice_GraduateAdmission.csv', index=False)
    print(f"\nPer-query results saved to: table5_nice_GraduateAdmission.csv")


if __name__ == "__main__":
    main()