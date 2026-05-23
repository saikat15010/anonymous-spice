"""
PertCF Evaluation for Table 5 (Main Results Table)
Dataset: HELOC

Faithful implementation of PertCF (Bayrak & Bach 2023).
Uses shap.KernelExplainer for the MLPClassifier model (MLPs require
model-agnostic SHAP since they're not tree-based or linear).

NOTE: KernelExplainer is slow. We use a small background sample (50)
and only 50 SHAP samples per data point. This is a standard adjustment
for KernelExplainer on neural networks.
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

NUM_QUERIES       = 100
RANDOM_SEED       = 42
ISO_CONTAMINATION = 0.10

NUM_ITER     = 10
COEF         = 5
ALPHA        = 0.5
TARGET_CLASS = 1

# KernelExplainer-specific: smaller background and fewer samples
SHAP_BG_SIZE   = 50
SHAP_NSAMPLES  = 50


class PertCF:
    def __init__(self, model, scaler, shap_per_class, num_iter=5, coef=5,
                 alpha=0.5, target_class=1, cat_idx=None, num_idx=None,
                 ranges=None):
        self.model = model
        self.scaler = scaler
        self.shap_per_class = shap_per_class
        self.num_iter = num_iter
        self.coef = coef
        self.alpha = alpha
        self.target_class = target_class
        self.cat_idx = cat_idx or []
        self.num_idx = num_idx or []
        self.ranges = ranges

    def _predict(self, x_raw):
        return self.model.predict(self.scaler.transform(x_raw.reshape(1, -1)))[0]

    def _weighted_distance(self, a, b, w):
        d = 0.0
        for i in self.cat_idx:
            d += w[i] * (1.0 if a[i] != b[i] else 0.0)
        for i in self.num_idx:
            rng = self.ranges[i] if self.ranges[i] != 0 else 1.0
            d += w[i] * abs(a[i] - b[i]) / rng
        return d

    def _perturb(self, source, target):
        shap_target = self.shap_per_class[self.target_class]
        p = source.copy().astype(float)
        for f in self.num_idx:
            step = float(np.clip(shap_target[f], 0.0, 1.0))
            p[f] = source[f] + step * (target[f] - source[f])
        for f in self.cat_idx:
            w_f = 1.0 if source[f] == target[f] else 0.0
            if w_f < self.alpha:
                p[f] = target[f]
            else:
                p[f] = source[f]
        return p

    def generate_counterfactual(self, x, nun):
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
                if prev_c is not None:
                    d_step = self._weighted_distance(c, prev_c, np.ones(len(c)))
                    if d_step < mu:
                        return c
                prev_c = c.copy()
                s = c.copy()
            else:
                s = c.copy()
        if candidates:
            return candidates[-1]
        return None


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


def main():
    data = pd.read_csv(DATA_PATH)
    data['target'] = data['target'].map({'Bad': 0, 'Good': 1})

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

    # ---- SHAP (KernelExplainer for MLP - slow but model-agnostic) ----
    print("Computing SHAP values using KernelExplainer (this is slow)...")
    bg_scaled = scaler.transform(X_train[:SHAP_BG_SIZE])

    def f_predict_proba(X_in_scaled):
        return model.predict_proba(X_in_scaled)

    explainer = shap.KernelExplainer(f_predict_proba, bg_scaled)
    shap_vals = explainer.shap_values(bg_scaled, nsamples=SHAP_NSAMPLES)

    if isinstance(shap_vals, list):
        shap_class0 = np.abs(shap_vals[0]).mean(axis=0)
        shap_class1 = np.abs(shap_vals[1]).mean(axis=0)
    else:
        if shap_vals.ndim == 3:
            shap_class0 = np.abs(shap_vals[:, :, 0]).mean(axis=0)
            shap_class1 = np.abs(shap_vals[:, :, 1]).mean(axis=0)
        else:
            shap_class1 = np.abs(shap_vals).mean(axis=0)
            shap_class0 = shap_class1

    def normalize(s):
        m = s.max() if s.max() > 0 else 1.0
        return s / m

    shap_per_class = {0: normalize(shap_class0), 1: normalize(shap_class1)}
    print(f"SHAP class 1 weights computed.\n")

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

    pos_pool_raw    = X[pos_indices]
    pos_pool_scaled = all_X_scaled[pos_indices]
    nbrs = NearestNeighbors(n_neighbors=1).fit(pos_pool_scaled)

    pertcf = PertCF(model=model, scaler=scaler, shap_per_class=shap_per_class,
                    num_iter=NUM_ITER, coef=COEF, alpha=ALPHA,
                    target_class=TARGET_CLASS,
                    cat_idx=cat_idx, num_idx=num_idx, ranges=ranges)

    print(f"Running PertCF on {n_take} queries...\n")
    rows, n_fail, start = [], 0, time.time()

    for q_i in range(n_take):
        if q_i % 10 == 0 and q_i > 0:
            print(f"  ...{q_i}/{n_take} (elapsed {time.time()-start:.0f}s)")
        q_raw = queries_raw[q_i]
        q_scaled = scaler.transform(q_raw.reshape(1, -1))
        _, nun_idx_in_pool = nbrs.kneighbors(q_scaled, n_neighbors=1)
        nun_raw = pos_pool_raw[nun_idx_in_pool[0][0]]

        cf_raw = pertcf.generate_counterfactual(q_raw, nun_raw)
        if cf_raw is None:
            n_fail += 1
            continue
        if model.predict(scaler.transform(cf_raw.reshape(1, -1)))[0] != TARGET_CLASS:
            n_fail += 1
            continue

        prox = heom_distance(q_raw, cf_raw, cat_idx, num_idx, ranges)
        spar = sparsity_ratio(q_raw, cf_raw)
        plau = 100.0 if isf.predict(scaler.transform(cf_raw.reshape(1, -1)))[0] == 1 else 0.0

        rows.append({
            'query_idx': q_i,
            'best_sparsity': spar,
            'best_proximity': prox,
            'plausibility_pct': plau,
        })

    df = pd.DataFrame(rows)
    elapsed = time.time() - start
    print(f"\nQueries with valid CFs: {len(df)}/{n_take}")
    print(f"Failed queries:         {n_fail}/{n_take}")
    print(f"Total runtime:          {elapsed:.1f}s\n")
    if len(df) == 0:
        return

    print("=" * 70)
    print("PertCF Table 5 Results - HELOC (100 queries)")
    print("=" * 70)
    cols   = ['best_sparsity', 'best_proximity', 'plausibility_pct']
    labels = ['Spar.', 'Prox.', 'Plaus. (%)']
    print(f"{'Metric':<15} {'Mean':>12} {'Std':>12} {'Formatted':>20}")
    print("-" * 62)
    for c, l in zip(cols, labels):
        m, s = df[c].mean(), df[c].std()
        print(f"{l:<15} {m:>12.4f} {s:>12.4f} {f'{m:.2f} +/- {s:.2f}':>20}")
    for l in ['Avg Spar.', 'Avg Prox.', 'Div. (%)']:
        print(f"{l:<15} {'-':>12} {'-':>12} {'-':>20}")

    df.to_csv('table5_pertcf_HELOC.csv', index=False)
    print(f"\nPer-query results saved to: table5_pertcf_HELOC.csv")


if __name__ == "__main__":
    main()