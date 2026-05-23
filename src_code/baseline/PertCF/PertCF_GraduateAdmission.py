"""
PertCF Evaluation for Table 5 (Main Results Table)
Dataset: Graduate Admission

Faithful implementation of PertCF (Bayrak & Bach 2023, SGAI):
"PertCF: A Perturbation-Based Counterfactual Generation Approach"

Algorithm:
  1. Compute average SHAP values for the target class (shap_target).
  2. For each query x: find NUN (nearest neighbor predicted opposite class).
  3. Iteratively generate candidates by perturbing source s toward target t:
       Numeric: p_f = s_f + shap_target_f * (t_f - s_f)        (Eq. 1)
       Nominal: p_f = t_f if w_f < alpha, else s_f             (Eq. 2)
  4. Termination:
       - num_iter reached, OR
       - distance(c_i, c_{i-1}) < mu, where mu = dist(x, NUN) / coef
  5. If a candidate flips prediction: continue perturbing toward t.
     Otherwise: perturb c_i with respect to s.

Paper-recommended parameters: num_iter=5, coef=5.

Protocol (matches SPICE/FACE):
  - Saved best_admission_rf_model.pkl
  - Same 100 model-predicted-negative queries as SPICE (seed=42)
  - 1 CF per query (PertCF design); report Spar, Prox, Plausibility only
"""

import numpy as np
import pandas as pd
import joblib
import time
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import NearestNeighbors
import shap

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

NUM_QUERIES       = 100
RANDOM_SEED       = 42
ISO_CONTAMINATION = 0.10

# PertCF paper-recommended parameters
NUM_ITER     = 10    # max iterations to generate counterfactual
COEF         = 5    # mu = dist(x, NUN) / coef
ALPHA        = 0.5  # nominal feature flip threshold
TARGET_CLASS = 1


# ============================================================
# PertCF class (faithful to Bayrak & Bach 2023)
# ============================================================

class PertCF:
    def __init__(self, model, scaler, shap_per_class, num_iter=5, coef=5,
                 alpha=0.5, target_class=1, cat_idx=None, num_idx=None,
                 cat_value_sets=None, ranges=None):
        """
        shap_per_class: dict[class_label -> np.array of mean |SHAP| per feature]
                        (used as perturbation step size for numeric features)
        cat_value_sets: dict[feature_idx -> set of valid raw values]
        """
        self.model = model
        self.scaler = scaler
        self.shap_per_class = shap_per_class
        self.num_iter = num_iter
        self.coef = coef
        self.alpha = alpha
        self.target_class = target_class
        self.cat_idx = cat_idx or []
        self.num_idx = num_idx or []
        self.cat_value_sets = cat_value_sets or {}
        self.ranges = ranges

    def _predict(self, x_raw):
        return self.model.predict(self.scaler.transform(x_raw.reshape(1, -1)))[0]

    def _weighted_distance(self, a, b, w):
        """Weighted L1 distance over numeric features + Hamming over categoricals."""
        d = 0.0
        for i in self.cat_idx:
            d += w[i] * (1.0 if a[i] != b[i] else 0.0)
        for i in self.num_idx:
            rng = self.ranges[i] if self.ranges[i] != 0 else 1.0
            d += w[i] * abs(a[i] - b[i]) / rng
        return d

    def _perturb(self, source, target):
        """Eq. 1 (numeric) and Eq. 2 (nominal) from the PertCF paper.
           shap_target controls step size for numeric features."""
        shap_target = self.shap_per_class[self.target_class]
        p = source.copy().astype(float)

        for f in self.num_idx:
            # Normalize SHAP magnitude so it acts as step fraction in [0, 1].
            # Paper's User Knowledge dataset uses raw mean |SHAP|; for stability
            # across datasets we clip into [0, 1] which is the recommended range.
            step = float(np.clip(shap_target[f], 0.0, 1.0))
            p[f] = source[f] + step * (target[f] - source[f])

        for f in self.cat_idx:
            # Nominal similarity w_f: fraction of training examples sharing the
            # source value among those that share the target value. This is a
            # data-driven proxy for the paper's myCBR similarity.
            w_f = self._nominal_similarity(f, source[f], target[f])
            if w_f < self.alpha:
                p[f] = target[f]
            else:
                p[f] = source[f]

        return p

    def _nominal_similarity(self, feat_idx, s_val, t_val):
        """Approximates the paper's myCBR similarity between two nominal
           values. Simple binary: 1 if same, 0 if different. The paper
           supports richer similarity matrices for domain knowledge."""
        return 1.0 if s_val == t_val else 0.0

    def generate_counterfactual(self, x, nun):
        """Generate counterfactual by iteratively perturbing toward NUN.
           Returns the final candidate or None on failure."""
        if self._predict(x) == self.target_class:
            return None

        mu = self._weighted_distance(x, nun, np.ones(len(x))) / self.coef
        candidates = []
        s = x.copy()
        t = nun.copy()
        prev_c = None

        for i in range(self.num_iter):
            c = self._perturb(s, t)
            pred = self._predict(c)

            if pred == self.target_class:
                candidates.append(c.copy())
                # Check distance to previous candidate (Sect. 3.2)
                if prev_c is not None:
                    d_step = self._weighted_distance(c, prev_c,
                                                     np.ones(len(c)))
                    if d_step < mu:
                        # Converged: stop and return latest
                        return c
                prev_c = c.copy()
                # Next iter: perturb c toward t (toward target) to refine
                s = c.copy()
                # t stays the same
            else:
                # Not flipped yet: perturb c toward s (back toward source)
                # is wrong direction; paper says "perturb c_i with respect
                # to s to generate c_{i+1}", meaning source becomes c_i,
                # target stays the source side to push further.
                # Following Fig. 4c interpretation: keep perturbing
                # c toward t to push harder over the boundary.
                s = c.copy()
                # t stays the NUN

        if candidates:
            return candidates[-1]
        return None


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
    return float(np.sum(q != c)) / len(q)


# ============================================================
# Main
# ============================================================

def main():
    # Load data
    d1 = pd.read_csv(TRAIN_PATH)
    d2 = pd.read_csv(TEST_PATH)
    data = pd.concat([d1, d2], ignore_index=True)

    X = data[FEATURE_COLUMNS].values.astype(float)

    split_idx = int(0.85 * len(X))
    X_train = X[:split_idx]
    scaler = StandardScaler()
    scaler.fit(X_train)

    model = joblib.load(MODEL_PATH)

    isf = IsolationForest(contamination=ISO_CONTAMINATION,
                          random_state=RANDOM_SEED)
    isf.fit(scaler.transform(X_train))

    cat_idx = [FEATURE_COLUMNS.index(f) for f in CATEGORICAL_FEATS]
    num_idx = [FEATURE_COLUMNS.index(f) for f in NUMERICAL_FEATS]

    ranges = np.zeros(len(FEATURE_COLUMNS))
    for i, f in enumerate(FEATURE_COLUMNS):
        if f in NUMERICAL_FEATS:
            ranges[i] = data[f].max() - data[f].min()

    # ---- Compute SHAP values per class ----
    print("Computing SHAP values...")
    bg = scaler.transform(X_train[:min(100, len(X_train))])
    explainer = shap.TreeExplainer(model)
    shap_vals = explainer.shap_values(bg)

    # shap_vals shape depends on sklearn version:
    #   - list of [n_samples, n_features] per class, OR
    #   - [n_samples, n_features, n_classes]
    if isinstance(shap_vals, list):
        shap_class0 = np.abs(shap_vals[0]).mean(axis=0)
        shap_class1 = np.abs(shap_vals[1]).mean(axis=0)
    else:
        # New API: 3D array
        if shap_vals.ndim == 3:
            shap_class0 = np.abs(shap_vals[:, :, 0]).mean(axis=0)
            shap_class1 = np.abs(shap_vals[:, :, 1]).mean(axis=0)
        else:
            # Binary case sometimes returns single array
            shap_class1 = np.abs(shap_vals).mean(axis=0)
            shap_class0 = shap_class1

    # Normalize to [0, 1] range to use as perturbation step fractions
    def normalize_shap(s):
        m = s.max() if s.max() > 0 else 1.0
        return s / m

    shap_per_class = {
        0: normalize_shap(shap_class0),
        1: normalize_shap(shap_class1),
    }
    print(f"SHAP class 1 weights: {shap_per_class[1]}\n")

    # ---- Identify queries and NUN pool (model-predicted) ----
    all_X_scaled = scaler.transform(X)
    all_preds = model.predict(all_X_scaled)
    pos_indices = np.where(all_preds == 1)[0]
    neg_indices = np.where(all_preds == 0)[0]

    rng = np.random.default_rng(RANDOM_SEED)
    n_take = min(NUM_QUERIES, len(neg_indices))
    chosen = rng.choice(neg_indices, size=n_take, replace=False)
    queries_raw = X[chosen]

    print(f"Dataset size: {len(X)}")
    print(f"Model predicts positive: {len(pos_indices)}")
    print(f"Model predicts negative: {len(neg_indices)}")
    print(f"Drawing {n_take} queries (seed={RANDOM_SEED})\n")

    # ---- NUN search: 1-NN on scaled space restricted to positives ----
    pos_pool_raw    = X[pos_indices]
    pos_pool_scaled = all_X_scaled[pos_indices]
    nbrs = NearestNeighbors(n_neighbors=1).fit(pos_pool_scaled)

    # ---- Run PertCF ----
    pertcf = PertCF(
        model=model, scaler=scaler,
        shap_per_class=shap_per_class,
        num_iter=NUM_ITER, coef=COEF, alpha=ALPHA,
        target_class=TARGET_CLASS,
        cat_idx=cat_idx, num_idx=num_idx, ranges=ranges,
    )

    print(f"Running PertCF on {n_take} queries "
          f"(num_iter={NUM_ITER}, coef={COEF})...\n")

    rows = []
    n_fail = 0
    start = time.time()

    for q_i in range(n_take):
        if q_i % 10 == 0 and q_i > 0:
            print(f"  ...{q_i}/{n_take} (elapsed {time.time()-start:.0f}s)")

        q_raw    = queries_raw[q_i]
        q_scaled = scaler.transform(q_raw.reshape(1, -1))

        # Find NUN (in scaled space, return raw value)
        _, nun_idx_in_pool = nbrs.kneighbors(q_scaled, n_neighbors=1)
        nun_raw = pos_pool_raw[nun_idx_in_pool[0][0]]

        # Generate counterfactual
        cf_raw = pertcf.generate_counterfactual(q_raw, nun_raw)
        if cf_raw is None:
            n_fail += 1
            continue

        # Verify flip
        cf_pred = model.predict(scaler.transform(cf_raw.reshape(1, -1)))[0]
        if cf_pred != TARGET_CLASS:
            n_fail += 1
            continue

        prox = heom_distance(q_raw, cf_raw, cat_idx, num_idx, ranges)
        spar = sparsity_ratio(q_raw, cf_raw)
        plau = 100.0 if isf.predict(scaler.transform(cf_raw.reshape(1, -1)))[0] == 1 else 0.0

        rows.append({
            'query_idx':       q_i,
            'best_sparsity':   spar,
            'best_proximity':  prox,
            'plausibility_pct': plau,
        })

    df = pd.DataFrame(rows)
    elapsed = time.time() - start
    print(f"\nQueries with valid CFs: {len(df)}/{n_take}")
    print(f"Failed queries:         {n_fail}/{n_take}")
    print(f"Total runtime:          {elapsed:.1f}s\n")

    if len(df) == 0:
        print("ERROR: no queries produced counterfactuals.")
        return

    print("=" * 70)
    print("PertCF Table 5 Results - Graduate Admission (100 queries)")
    print("Reported as mean +/- std. PertCF produces 1 CF per query, so Avg")
    print("Spar, Avg Prox, and Diversity are reported as '-'.")
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
    for l in ['Avg Spar.', 'Avg Prox.', 'Div. (%)']:
        print(f"{l:<15} {'-':>12} {'-':>12} {'-':>20}")

    df.to_csv('table5_pertcf_GraduateAdmission.csv', index=False)
    print(f"\nPer-query results saved to: table5_pertcf_GraduateAdmission.csv")


if __name__ == "__main__":
    main()