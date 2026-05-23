"""
VanillaCF Evaluation for Table 5 (Main Results Table)
Dataset: Graduate Admission

VanillaCF (Wachter et al. 2017): gradient-based optimization minimizing
prediction loss + lambda * L1(cf - query). Uses 40 random restarts for
diversity, matching VanilaCF2.py.

Protocol (matches SPICE/FACE):
  - Saved best_admission_rf_model.pkl
  - 100 same model-predicted-negative queries as SPICE (seed=42)
  - 40 random restarts per query
  - Isolation Forest plausibility: contamination=0.10
  - Diversity: mean pairwise Hamming distance x 100
  - All 6 metrics reported (Best+Avg Spar/Prox, Diversity, Plausibility)
"""

import numpy as np
import pandas as pd
import joblib
import time
import warnings
warnings.filterwarnings('ignore')

from itertools import combinations
from scipy.spatial.distance import hamming
from scipy.optimize import minimize
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

# ============================================================
# CONFIG
# ============================================================

TRAIN_PATH = '/Users/saikat/Desktop/Summer 2025/CFNet_AAAI_26/Graduate Admission Dataset/admission_train.csv'
TEST_PATH  = '/Users/saikat/Desktop/Summer 2025/CFNet_AAAI_26/Graduate Admission Dataset/admission_test.csv'
MODEL_PATH = '/Users/saikat/Desktop/Summer 2025/CFNet_AAAI_26/Graduate Admission Dataset/AISTATS Graduate Admission/best_admission_rf_model.pkl'

FEATURE_COLUMNS    = ['GRE Score', 'TOEFL Score', 'University Rating',
                      'SOP', 'LOR', 'CGPA', 'Research']
CATEGORICAL_FEATS  = ['University Rating', 'Research']
NUMERICAL_FEATS    = ['GRE Score', 'TOEFL Score', 'SOP', 'LOR', 'CGPA']
IMMUTABLE_FEATS    = []

NUM_QUERIES       = 100
RANDOM_SEED       = 42
ISO_CONTAMINATION = 0.10

# VanillaCF hyperparameters (match VanilaCF2.py)
N_RESTARTS        = 40
LAMBDA_PARAM      = 0.5
MAX_ITER          = 1000
NOISE_SCALE       = 0.1
TARGET_CLASS      = 1


# ============================================================
# VanillaCF class (matches VanilaCF2.py)
# ============================================================

class VanillaCF:
    def __init__(self, model, scaler, target_class=1,
                 lambda_param=0.5, max_iter=1000):
        self.model = model
        self.scaler = scaler
        self.target_class = target_class
        self.lambda_param = lambda_param
        self.max_iter = max_iter

    def objective_function(self, cf_scaled, x_scaled):
        cf_reshaped = cf_scaled.reshape(1, -1)
        pred_proba = self.model.predict_proba(cf_reshaped)[0][self.target_class]
        prediction_loss = -np.log(pred_proba + 1e-10)
        distance_loss = np.sum(np.abs(cf_scaled - x_scaled.flatten()))
        return prediction_loss + self.lambda_param * distance_loss

    def generate_counterfactual(self, x_scaled):
        cf_initial = x_scaled.copy().flatten()
        result = minimize(
            fun=self.objective_function,
            x0=cf_initial,
            args=(x_scaled,),
            method='L-BFGS-B',
            options={'maxiter': self.max_iter, 'disp': False},
        )
        cf_scaled = result.x.reshape(1, -1)
        cf_unscaled = self.scaler.inverse_transform(cf_scaled)
        return cf_unscaled[0], result.success, result.fun


# ============================================================
# Metric helpers
# ============================================================

def heom_distance(query, candidate, cat_idx, num_idx, ranges):
    d = 0.0
    for i in cat_idx:
        d += 1.0 if query[i] != candidate[i] else 0.0
    for i in num_idx:
        rng = ranges[i] if ranges[i] != 0 else 1
        d += abs(query[i] - candidate[i]) / rng
    return d


def sparsity_ratio(q, c):
    # Match VanilaCF2.py's "significant change" threshold of 1e-6
    return float(np.sum(np.abs(q - c) > 1e-6)) / len(q)


def filter_for_diversity(arr, q_raw, nun_pseudo, cat_idx, immutable_idx):
    """Same filter formula as SPICE, with nun_pseudo as the centroid."""
    keep = []
    for i in range(len(arr)):
        if i in immutable_idx:
            continue
        if i in cat_idx and q_raw[i] == nun_pseudo[i]:
            continue
        keep.append(arr[i])
    return np.array(keep)


def compute_diversity(candidates, q_raw, cat_idx, immutable_idx):
    """Compute diversity using mean of candidates as pseudo-NUN for filter
       (matches VanilaCF2.py)."""
    if len(candidates) < 2:
        return 0.0
    nun_pseudo = np.mean(candidates, axis=0)
    filtered = [filter_for_diversity(cf, q_raw, nun_pseudo, cat_idx, immutable_idx)
                for cf in candidates]
    filtered = [f for f in filtered if len(f) > 0]
    if len(filtered) < 2:
        return 0.0
    pairs = list(combinations(filtered, 2))
    dists = []
    for p in pairs:
        if len(p[0]) == len(p[1]) and len(p[0]) > 0:
            dists.append(hamming(p[0], p[1]))
    if len(dists) == 0:
        return 0.0
    return float(np.mean(dists)) * 100.0


def compute_plausibility(candidates_scaled, isf):
    if len(candidates_scaled) == 0:
        return 0.0
    arr = np.array(candidates_scaled)
    preds = isf.predict(arr)
    return float(np.mean(preds == 1)) * 100.0


# ============================================================
# Main
# ============================================================

def main():
    # Load data
    d1 = pd.read_csv(TRAIN_PATH)
    d2 = pd.read_csv(TEST_PATH)
    data = pd.concat([d1, d2], ignore_index=True)

    X = data[FEATURE_COLUMNS].values.astype(float)

    # Match SPICE's scaler setup
    split_idx = int(0.85 * len(X))
    X_train = X[:split_idx]
    scaler = StandardScaler()
    scaler.fit(X_train)

    model = joblib.load(MODEL_PATH)

    isf = IsolationForest(contamination=ISO_CONTAMINATION,
                          random_state=RANDOM_SEED)
    isf.fit(scaler.transform(X_train))

    # Identify model-predicted negatives (same as SPICE)
    all_X_scaled = scaler.transform(X)
    all_preds = model.predict(all_X_scaled)
    neg_indices = np.where(all_preds == 0)[0]

    rng = np.random.default_rng(RANDOM_SEED)
    n_take = min(NUM_QUERIES, len(neg_indices))
    chosen = rng.choice(neg_indices, size=n_take, replace=False)

    queries_raw    = X[chosen]
    queries_scaled = all_X_scaled[chosen]

    print(f"Dataset size: {len(X)}")
    print(f"Drawing {n_take} queries (seed={RANDOM_SEED})")
    print(f"Restarts per query: {N_RESTARTS}\n")

    # Setup
    cat_idx = [FEATURE_COLUMNS.index(f) for f in CATEGORICAL_FEATS]
    num_idx = [FEATURE_COLUMNS.index(f) for f in NUMERICAL_FEATS]
    immutable_idx = [FEATURE_COLUMNS.index(f) for f in IMMUTABLE_FEATS]

    ranges = np.zeros(len(FEATURE_COLUMNS))
    for i, f in enumerate(FEATURE_COLUMNS):
        if f in NUMERICAL_FEATS:
            ranges[i] = data[f].max() - data[f].min()

    vanilla = VanillaCF(model=model, scaler=scaler,
                        target_class=TARGET_CLASS,
                        lambda_param=LAMBDA_PARAM, max_iter=MAX_ITER)

    print(f"Running VanillaCF on {n_take} queries...")

    rows = []
    n_fail = 0
    start = time.time()

    for q_i in range(n_take):
        if q_i % 10 == 0 and q_i > 0:
            print(f"  ...{q_i}/{n_take} (elapsed {time.time()-start:.0f}s)")

        q_raw    = queries_raw[q_i]
        q_scaled = queries_scaled[q_i]

        successful_raw = []      # CFs in raw (unscaled) space
        successful_scaled = []   # CFs in scaled space

        for restart in range(N_RESTARTS):
            np.random.seed(restart)
            noise = np.random.normal(0, NOISE_SCALE, q_scaled.shape)
            noisy_scaled = (q_scaled + noise).reshape(1, -1)

            cf_unscaled, _, _ = vanilla.generate_counterfactual(noisy_scaled)
            cf_scaled = scaler.transform(cf_unscaled.reshape(1, -1))[0]

            # Verify the CF flips the prediction
            if model.predict(cf_scaled.reshape(1, -1))[0] == TARGET_CLASS:
                successful_raw.append(cf_unscaled)
                successful_scaled.append(cf_scaled)

        if len(successful_raw) == 0:
            n_fail += 1
            continue

        # Per-CF metrics (raw space for HEOM/sparsity)
        proxs = [heom_distance(q_raw, cf, cat_idx, num_idx, ranges)
                 for cf in successful_raw]
        spars = [sparsity_ratio(q_raw, cf) for cf in successful_raw]

        div  = compute_diversity(successful_raw, q_raw, cat_idx, immutable_idx)
        # Plausibility uses scaled CFs (IsolationForest was fit on scaled data)
        plau = compute_plausibility(successful_scaled, isf)

        rows.append({
            'query_idx':       q_i,
            'n_candidates':    len(successful_raw),
            'best_sparsity':   float(min(spars)),
            'best_proximity':  float(min(proxs)),
            'avg_sparsity':    float(np.mean(spars)),
            'avg_proximity':   float(np.mean(proxs)),
            'diversity_pct':   div,
            'plausibility_pct': plau,
        })

    df = pd.DataFrame(rows)
    elapsed = time.time() - start
    print(f"\nQueries with valid CFs: {len(df)}/{n_take}")
    print(f"Failed queries:         {n_fail}/{n_take}")
    print(f"Total runtime:          {elapsed:.1f}s")
    if len(df) > 0:
        print(f"Avg CFs per query:      {df['n_candidates'].mean():.1f}\n")

    if len(df) == 0:
        print("ERROR: no queries produced counterfactuals.")
        return

    print("=" * 70)
    print("VanillaCF Table 5 Results - Graduate Admission (100 queries, 40 restarts)")
    print("Reported as mean +/- std")
    print("=" * 70)
    cols = ['best_sparsity', 'best_proximity',
            'avg_sparsity', 'avg_proximity',
            'diversity_pct', 'plausibility_pct']
    labels = ['Spar.', 'Prox.', 'Avg Spar.', 'Avg Prox.', 'Div. (%)', 'Plaus. (%)']
    print(f"{'Metric':<15} {'Mean':>12} {'Std':>12} {'Formatted':>20}")
    print("-" * 62)
    for c, l in zip(cols, labels):
        m = df[c].mean()
        s = df[c].std()
        formatted = f"{m:.2f} +/- {s:.2f}"
        print(f"{l:<15} {m:>12.4f} {s:>12.4f} {formatted:>20}")

    df.to_csv('table5_vanillacf_GraduateAdmission.csv', index=False)
    print(f"\nPer-query results saved to: table5_vanillacf_GraduateAdmission.csv")


if __name__ == "__main__":
    main()
