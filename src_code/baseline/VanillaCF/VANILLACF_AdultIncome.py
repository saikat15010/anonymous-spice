"""
VanillaCF Evaluation for Table 5 - Adult Income
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
from sklearn.model_selection import train_test_split

DATA_PATH  = '/Users/saikat/Desktop/Summer 2025/CFNet_AAAI_26/Adult Income Dataset/AISTATS Adult Income /processed_adult.csv'
MODEL_PATH = '/Users/saikat/Desktop/Summer 2025/CFNet_AAAI_26/Adult Income Dataset/AISTATS Adult Income /best_adult_income_gb_model.pkl'

FEATURE_COLUMNS = ['age', 'workclass', 'fnlwgt', 'education', 'educational-num',
                   'marital-status', 'occupation', 'relationship', 'race',
                   'gender', 'capital-gain', 'capital-loss', 'hours-per-week',
                   'native-country']
CATEGORICAL_FEATS = ['workclass', 'education', 'marital-status', 'occupation',
                     'relationship', 'race', 'gender', 'native-country']
NUMERICAL_FEATS   = ['age', 'fnlwgt', 'educational-num', 'capital-gain',
                     'capital-loss', 'hours-per-week']
IMMUTABLE_FEATS   = ['age', 'gender', 'workclass', 'race', 'fnlwgt','education', 'native-country']

NUM_QUERIES = 100
RANDOM_SEED = 42
ISO_CONTAMINATION = 0.10
N_RESTARTS = 40
LAMBDA_PARAM = 0.5
MAX_ITER = 1000
NOISE_SCALE = 0.1
TARGET_CLASS = 1


class VanillaCF:
    def __init__(self, model, scaler, target_class=1, lambda_param=0.5, max_iter=1000):
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
        result = minimize(self.objective_function, cf_initial, args=(x_scaled,),
                          method='L-BFGS-B',
                          options={'maxiter': self.max_iter, 'disp': False})
        cf_scaled = result.x.reshape(1, -1)
        return self.scaler.inverse_transform(cf_scaled)[0], result.success, result.fun


def heom_distance(query, candidate, cat_idx, num_idx, ranges):
    d = 0.0
    for i in cat_idx:
        d += 1.0 if query[i] != candidate[i] else 0.0
    for i in num_idx:
        rng = ranges[i] if ranges[i] != 0 else 1
        d += abs(query[i] - candidate[i]) / rng
    return d


def sparsity_ratio(q, c):
    return float(np.sum(np.abs(q - c) > 1e-6)) / len(q)


def filter_for_diversity(arr, q_raw, nun_pseudo, cat_idx, immutable_idx):
    keep = []
    for i in range(len(arr)):
        if i in immutable_idx:
            continue
        if i in cat_idx and q_raw[i] == nun_pseudo[i]:
            continue
        keep.append(arr[i])
    return np.array(keep)


def compute_diversity(candidates, q_raw, cat_idx, immutable_idx):
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
    return float(np.mean(isf.predict(np.array(candidates_scaled)) == 1)) * 100.0


def main():
    data = pd.read_csv(DATA_PATH)
    X = data[FEATURE_COLUMNS].values.astype(float)
    y = data['target'].values
    X_train, _, _, _ = train_test_split(X, y, test_size=0.15, random_state=0)
    X_train, _, _, _ = train_test_split(X_train, y[:len(X_train)],
                                        test_size=0.10, random_state=0)
    scaler = StandardScaler()
    scaler.fit(X_train)
    model = joblib.load(MODEL_PATH)
    isf = IsolationForest(contamination=ISO_CONTAMINATION, random_state=RANDOM_SEED)
    isf.fit(scaler.transform(X_train))

    all_X_scaled = scaler.transform(X)
    all_preds = model.predict(all_X_scaled)
    neg_indices = np.where(all_preds == 0)[0]
    rng = np.random.default_rng(RANDOM_SEED)
    n_take = min(NUM_QUERIES, len(neg_indices))
    chosen = rng.choice(neg_indices, size=n_take, replace=False)
    queries_raw = X[chosen]
    queries_scaled = all_X_scaled[chosen]

    print(f"Dataset size: {len(X)}")
    print(f"Drawing {n_take} queries (seed={RANDOM_SEED})")
    print(f"Restarts per query: {N_RESTARTS}\n")

    cat_idx = [FEATURE_COLUMNS.index(f) for f in CATEGORICAL_FEATS]
    num_idx = [FEATURE_COLUMNS.index(f) for f in NUMERICAL_FEATS]
    immutable_idx = [FEATURE_COLUMNS.index(f) for f in IMMUTABLE_FEATS]
    ranges = np.zeros(len(FEATURE_COLUMNS))
    for i, f in enumerate(FEATURE_COLUMNS):
        if f in NUMERICAL_FEATS:
            ranges[i] = data[f].max() - data[f].min()

    vanilla = VanillaCF(model=model, scaler=scaler, target_class=TARGET_CLASS,
                        lambda_param=LAMBDA_PARAM, max_iter=MAX_ITER)

    print(f"Running VanillaCF on {n_take} queries (may take 30-60 min)...")
    rows, n_fail, start = [], 0, time.time()

    for q_i in range(n_take):
        if q_i % 5 == 0 and q_i > 0:
            print(f"  ...{q_i}/{n_take} (elapsed {time.time()-start:.0f}s)")
        q_raw = queries_raw[q_i]
        q_scaled = queries_scaled[q_i]
        successful_raw, successful_scaled = [], []

        for restart in range(N_RESTARTS):
            np.random.seed(restart)
            noise = np.random.normal(0, NOISE_SCALE, q_scaled.shape)
            noisy_scaled = (q_scaled + noise).reshape(1, -1)
            cf_unscaled, _, _ = vanilla.generate_counterfactual(noisy_scaled)
            cf_scaled = scaler.transform(cf_unscaled.reshape(1, -1))[0]
            if model.predict(cf_scaled.reshape(1, -1))[0] == TARGET_CLASS:
                successful_raw.append(cf_unscaled)
                successful_scaled.append(cf_scaled)

        if len(successful_raw) == 0:
            n_fail += 1
            continue

        proxs = [heom_distance(q_raw, cf, cat_idx, num_idx, ranges) for cf in successful_raw]
        spars = [sparsity_ratio(q_raw, cf) for cf in successful_raw]
        div = compute_diversity(successful_raw, q_raw, cat_idx, immutable_idx)
        plau = compute_plausibility(successful_scaled, isf)

        rows.append({
            'query_idx': q_i, 'n_candidates': len(successful_raw),
            'best_sparsity': float(min(spars)), 'best_proximity': float(min(proxs)),
            'avg_sparsity': float(np.mean(spars)), 'avg_proximity': float(np.mean(proxs)),
            'diversity_pct': div, 'plausibility_pct': plau,
        })

    df = pd.DataFrame(rows)
    elapsed = time.time() - start
    print(f"\nQueries with valid CFs: {len(df)}/{n_take}")
    print(f"Failed queries:         {n_fail}/{n_take}")
    print(f"Total runtime:          {elapsed:.1f}s\n")
    if len(df) == 0:
        return

    print("=" * 70)
    print("VanillaCF Table 5 Results - Adult Income (100 queries, 40 restarts)")
    print("=" * 70)
    cols = ['best_sparsity', 'best_proximity', 'avg_sparsity', 'avg_proximity',
            'diversity_pct', 'plausibility_pct']
    labels = ['Spar.', 'Prox.', 'Avg Spar.', 'Avg Prox.', 'Div. (%)', 'Plaus. (%)']
    print(f"{'Metric':<15} {'Mean':>12} {'Std':>12} {'Formatted':>20}")
    print("-" * 62)
    for c, l in zip(cols, labels):
        m, s = df[c].mean(), df[c].std()
        print(f"{l:<15} {m:>12.4f} {s:>12.4f} {f'{m:.2f} +/- {s:.2f}':>20}")

    df.to_csv('table5_vanillacf_AdultIncome.csv', index=False)
    print(f"\nPer-query results saved to: table5_vanillacf_AdultIncome.csv")


if __name__ == "__main__":
    main()
