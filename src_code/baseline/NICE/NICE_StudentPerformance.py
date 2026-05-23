"""
NICE Baseline Evaluation — Student Performance Dataset
=======================================================
NICE: Nearest Instance Counterfactual Explanations
Brughmans et al. (2024). Data Mining and Knowledge Discovery.

Algorithm:
  Step 1 — Find the exhaustive Nearest Unlike Neighbor (NUN): the closest
            instance from the opposite class using HEOM distance.
  Step 2 — Start from cf = NUN (guaranteed valid counterfactual).
  Step 3 — Greedily restore features back toward the query one at a time.
            At each step, pick the feature whose restoration:
              (a) keeps f(cf) != f(query)  [validity preserved], AND
              (b) reduces HEOM(query, cf) the most  [proximity improves].
            Continue until no further restoration is possible without
            invalidating the counterfactual.
  Step 4 — Return the final cf (single counterfactual per query).

Protocol mirrors SPICE exactly:
  - 100 random queries drawn from model-predicted negatives (seed=42)
  - Same 85/15 split, model (Logistic Regression), and IsolationForest
  - Same HEOM distance, sparsity ratio, and plausibility metric
  - Categorical yes/no features encoded as 0/1 (same as SPICE)
  - Immutable feature 'age' is never restored or substituted
  - Reported as mean +/- std across queries that produced a valid CF
  - Coverage = fraction of queries returning a valid counterfactual

Note: NICE produces ONE counterfactual per query, so diversity and
average proximity/sparsity are not reported (marked N/A), consistent
with how NICE is treated in Table 3 of the SPICE paper.
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_PATH  = '/Users/saikat/Desktop/Summer 2025/CFNet_AAAI_26/Student Performance Datastet/processed_student.csv'
MODEL_PATH = '/Users/saikat/Desktop/Summer 2025/CFNet_AAAI_26/Student Performance Datastet/AISTATS Student Performance/best_admission_lr_model.pkl'

# ── Feature schema (must match SPICE exactly) ─────────────────────────────────
FEATURE_COLUMNS   = ['age', 'Medu', 'Fedu', 'studytime', 'famsup', 'higher',
                     'internet', 'romantic', 'freetime', 'goout', 'health',
                     'absences', 'G1', 'G2']
CATEGORICAL_FEATS = ['famsup', 'higher', 'internet', 'romantic']  # binary 0/1
NUMERICAL_FEATS   = ['age', 'Medu', 'Fedu', 'studytime', 'freetime', 'goout',
                     'health', 'absences', 'G1', 'G2']
IMMUTABLE_FEATS   = ['age', 'higher', 'romantic', 'Fedu', 'studytime', 'G1']
TARGET_COL        = 'target'

# ── Evaluation hyper-parameters ───────────────────────────────────────────────
NUM_QUERIES       = 100
RANDOM_SEED       = 42
ISO_CONTAMINATION = 0.10      # identical to SPICE


# ─────────────────────────────────────────────────────────────────────────────
# Metric helpers  (identical implementations to SPICE)
# ─────────────────────────────────────────────────────────────────────────────

def build_ranges(data):
    """Per-feature numeric ranges for HEOM normalisation."""
    ranges = np.zeros(len(FEATURE_COLUMNS))
    for i, f in enumerate(FEATURE_COLUMNS):
        if f in NUMERICAL_FEATS:
            ranges[i] = data[f].max() - data[f].min()
    return ranges


def heom_distance(query, candidate, cat_idx, num_idx, ranges):
    """Heterogeneous Euclidean-Overlap Metric — identical to SPICE."""
    d = 0.0
    for i in cat_idx:
        d += 1.0 if query[i] != candidate[i] else 0.0
    for i in num_idx:
        rng = ranges[i] if ranges[i] != 0 else 1.0
        d += abs(query[i] - candidate[i]) / rng
    return d


def sparsity_ratio(q, c):
    """Proportion of features that differ — identical to SPICE."""
    return float(np.sum(np.array(q) != np.array(c))) / len(q)


def compute_plausibility(candidate, isf, scaler):
    """
    1.0 if CF is deemed an inlier by IsolationForest, 0.0 otherwise.
    NICE returns a single CF so this is binary per query; the mean
    across queries equals the fraction of plausible CFs.
    """
    arr        = np.array(candidate).reshape(1, -1)
    arr_scaled = scaler.transform(arr)
    pred       = isf.predict(arr_scaled)[0]
    return 1.0 if pred == 1 else 0.0


# ─────────────────────────────────────────────────────────────────────────────
# NICE core
# ─────────────────────────────────────────────────────────────────────────────

def find_nun_exhaustive(q_raw, pool_opposite, cat_idx, num_idx, ranges):
    """
    Exhaustive nearest unlike neighbor search — O(n * m).
    NICE guarantees 100% coverage because the NUN is itself a valid CF
    and always exists when the opposite-class pool is non-empty.
    """
    best_idx, best_dist = None, float('inf')
    for i, nb in enumerate(pool_opposite):
        d = heom_distance(q_raw, nb, cat_idx, num_idx, ranges)
        if d < best_dist:
            best_dist, best_idx = d, i
    return pool_opposite[best_idx] if best_idx is not None else None


def nice_generate(q_raw, nun, model, scaler, cat_idx, num_idx,
                  immutable_idx, ranges):
    """
    NICE greedy feature-restoration algorithm.

    Start from cf = NUN (valid by definition).
    Each round: try restoring every not-yet-restored mutable feature
    back to the query value. Accept the one that:
      1. Preserves validity — f(cf) != f(query), AND
      2. Gives the greatest reduction in HEOM(query, cf).
    Repeat until no valid restoration remains.

    Special handling for Student Performance:
      - 'age' (immutable) is never touched.
      - Binary categoricals (famsup, higher, internet, romantic) are
        restored to exactly the query's 0/1 value — no fractional risk
        since both query and NUN are already 0 or 1.

    Returns the final counterfactual (single instance).
    """
    orig_pred = model.predict(scaler.transform(q_raw.reshape(1, -1)))[0]
    cf        = nun.copy().astype(float)
    mutable   = [i for i in range(len(FEATURE_COLUMNS))
                 if i not in immutable_idx]

    not_restored = set(mutable)   # features not yet restored to query value

    improved = True
    while improved and not_restored:
        improved       = False
        best_feat      = None
        best_dist_gain = -np.inf

        current_dist = heom_distance(q_raw, cf, cat_idx, num_idx, ranges)

        for feat_idx in not_restored:
            # Tentatively restore this feature to query value
            trial            = cf.copy()
            trial[feat_idx]  = q_raw[feat_idx]

            # If already equal, restoring is a no-op — skip
            if trial[feat_idx] == cf[feat_idx]:
                continue

            # Check validity after restoration
            pred = model.predict(
                scaler.transform(trial.reshape(1, -1)))[0]
            if pred == orig_pred:
                # Restoring this feature flips prediction back — not allowed
                continue

            # Measure proximity improvement
            new_dist  = heom_distance(q_raw, trial, cat_idx, num_idx, ranges)
            dist_gain = current_dist - new_dist   # positive = closer to query

            if dist_gain > best_dist_gain:
                best_dist_gain = dist_gain
                best_feat      = feat_idx

        if best_feat is not None:
            cf[best_feat] = q_raw[best_feat]
            not_restored.discard(best_feat)
            improved = True

    return cf


# ─────────────────────────────────────────────────────────────────────────────
# Data & model loading
# ─────────────────────────────────────────────────────────────────────────────

def load_everything():
    data = pd.read_csv(DATA_PATH)

    # Encode categorical yes/no → 0/1 (same as SPICE)
    for f in CATEGORICAL_FEATS:
        if data[f].dtype == object:
            data[f] = data[f].map({'yes': 1, 'no': 0})

    X = data[FEATURE_COLUMNS].values.astype(float)

    # Simple 85/15 split — identical to SPICE Student Performance
    split_idx = int(0.85 * len(X))
    X_train   = X[:split_idx]

    scaler = StandardScaler()
    scaler.fit(X_train)

    model = joblib.load(MODEL_PATH)

    isf = IsolationForest(contamination=ISO_CONTAMINATION,
                          random_state=RANDOM_SEED)
    isf.fit(scaler.transform(X_train))

    cat_idx       = [FEATURE_COLUMNS.index(f) for f in CATEGORICAL_FEATS]
    num_idx       = [FEATURE_COLUMNS.index(f) for f in NUMERICAL_FEATS]
    immutable_idx = [FEATURE_COLUMNS.index(f) for f in IMMUTABLE_FEATS]
    ranges        = build_ranges(data)

    return data, model, scaler, isf, cat_idx, num_idx, immutable_idx, ranges


# ─────────────────────────────────────────────────────────────────────────────
# Main evaluation loop
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("Loading data, model, and IsolationForest...")
    (data, model, scaler, isf,
     cat_idx, num_idx, immutable_idx, ranges) = load_everything()

    # ── Feature matrix and model predictions ──────────────────────────────────
    all_X     = data[FEATURE_COLUMNS].values.astype(float)
    all_X_sc  = scaler.transform(all_X)
    all_preds = model.predict(all_X_sc)

    pos_mask = (all_preds == 1)
    neg_mask = (all_preds == 0)

    print(f"Dataset size              : {len(data)}")
    print(f"Model predicts positive   : {pos_mask.sum()} "
          f"({100*pos_mask.mean():.1f}%)")
    print(f"Model predicts negative   : {neg_mask.sum()} "
          f"({100*neg_mask.mean():.1f}%)")
    print(f"Ground-truth positives    : {(data[TARGET_COL]==1).sum()}")
    print(f"Ground-truth negatives    : {(data[TARGET_COL]==0).sum()}\n")

    # Opposite-class pool (NUN source) and query pool
    pos_pool_raw = all_X[pos_mask]   # flipped-to target
    neg_pool_raw = all_X[neg_mask]   # queries drawn from here

    if len(pos_pool_raw) == 0:
        print("ERROR: no model-predicted positives — cannot find NUN.")
        return

    # ── 100 random queries (same seed as SPICE) ───────────────────────────────
    rng    = np.random.default_rng(RANDOM_SEED)
    n_take = min(NUM_QUERIES, len(neg_pool_raw))
    q_idx  = rng.choice(len(neg_pool_raw), size=n_take, replace=False)
    queries_raw = neg_pool_raw[q_idx]

    print(f"Running NICE on {n_take} queries (exhaustive NUN + greedy "
          f"restoration)...\n")

    rows                     = []
    n_no_cf                  = 0
    nun_same_label_failures  = 0

    for qi in range(n_take):
        if qi % 20 == 0 and qi > 0:
            print(f"  ...{qi}/{n_take}")

        q_raw     = queries_raw[qi]
        orig_pred = model.predict(
            scaler.transform(q_raw.reshape(1, -1)))[0]

        # ── Step 1: exhaustive NUN search ─────────────────────────────────
        nun = find_nun_exhaustive(
            q_raw, pos_pool_raw, cat_idx, num_idx, ranges)
        if nun is None:
            n_no_cf += 1
            continue

        # Sanity-check: NUN must have opposite prediction
        nun_pred = model.predict(
            scaler.transform(nun.reshape(1, -1)))[0]
        if nun_pred == orig_pred:
            # Rare boundary artefact — skip
            nun_same_label_failures += 1
            n_no_cf += 1
            continue

        # ── Steps 2 & 3: greedy restoration ──────────────────────────────
        cf = nice_generate(q_raw, nun, model, scaler,
                           cat_idx, num_idx, immutable_idx, ranges)

        # Final validity guard — fall back to NUN if restoration overshot
        cf_pred = model.predict(
            scaler.transform(cf.reshape(1, -1)))[0]
        used_fallback = False
        if cf_pred == orig_pred:
            cf            = nun.copy()
            used_fallback = True

        # ── Metrics ───────────────────────────────────────────────────────
        prox = heom_distance(q_raw, cf, cat_idx, num_idx, ranges)
        spar = sparsity_ratio(q_raw, cf)
        plau = compute_plausibility(cf, isf, scaler)

        rows.append({
            'query_idx'        : qi,
            'best_sparsity'    : spar,
            'best_proximity'   : prox,
            'plausibility_pct' : plau * 100.0,
            'used_nun_fallback': int(used_fallback),
        })

    # ── Summary ───────────────────────────────────────────────────────────────
    df       = pd.DataFrame(rows)
    coverage = len(df) / n_take * 100.0

    print(f"\nQueries with valid CF      : {len(df)}/{n_take} "
          f"(Coverage = {coverage:.1f}%)")
    print(f"Queries with no valid CF   : {n_no_cf}/{n_take}")
    if nun_same_label_failures > 0:
        print(f"NUN same-label edge cases  : {nun_same_label_failures}")
    if len(df) > 0:
        nun_fb = df['used_nun_fallback'].sum()
        print(f"Queries using NUN fallback : {nun_fb}/{len(df)} "
              f"({100*nun_fb/len(df):.1f}%)\n")

    if len(df) == 0:
        print("ERROR: NICE produced no valid counterfactuals.")
        return

    # ── Results table ─────────────────────────────────────────────────────────
    print("=" * 70)
    print("NICE Results — Student Performance (exhaustive NUN + greedy")
    print(f"restoration, 100 queries)  |  Coverage = {coverage:.1f}%")
    print("=" * 70)

    metrics = [
        ('best_sparsity',    'Spar.      ↓'),
        ('best_proximity',   'Prox.      ↓'),
        ('plausibility_pct', 'Plaus. (%) ↑'),
    ]

    print(f"\n{'Metric':<18} {'Mean':>10} {'Std':>10}   {'Formatted':>22}")
    print("-" * 66)
    for col, label in metrics:
        m = df[col].mean()
        s = df[col].std()
        print(f"{label:<18} {m:>10.4f} {s:>10.4f}   {m:.2f} ± {s:.2f}")

    print("\n  Avg Spar., Avg Prox., Div.: N/A  "
          "(NICE generates one CF per query)")
    print(f"\n  Coverage: {coverage:.1f}%  "
          f"({len(df)}/{n_take} queries returned a valid CF)")

    # ── Save per-query results ────────────────────────────────────────────────
    out_path = 'table5_nice_StudentPerformance.csv'
    df.to_csv(out_path, index=False)
    print(f"\nPer-query results saved to: {out_path}")



if __name__ == "__main__":
    main()