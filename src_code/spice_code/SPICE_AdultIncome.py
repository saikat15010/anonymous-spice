"""
SPICE Evaluation for Table 5 (Main Results Table)
Dataset: Adult Income

KEY FIXES:
  1. NUN pool filtered by MODEL PREDICTION (not ground-truth label).
     The previous version used data[target==1] but 85% of those points
     were classified as negative by the model, making them unusable as
     counterfactual destinations.
  2. Query pool filtered by model prediction == 0 (not target == 0).
  3. Reporting in mean +/- std format.

Protocol:
  - 100 random queries (seed=42), LIME feature importance
  - All mutable features, lambda=0.5, delta=0.3, P=10
  - Isolation Forest: trained on X_train, contamination=0.10
  - Diversity: mean pairwise Hamming distance x 100 over candidate set
"""

import numpy as np
import pandas as pd
import joblib
from itertools import combinations
from scipy.spatial.distance import hamming
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import normalize, StandardScaler
from sklearn.model_selection import train_test_split

DATA_PATH = '/Users/saikat/Desktop/Summer 2025/CFNet_AAAI_26/Adult Income Dataset/AISTATS Adult Income /processed_adult.csv'
MODEL_PATH = '/Users/saikat/Desktop/Summer 2025/CFNet_AAAI_26/Adult Income Dataset/AISTATS Adult Income /best_adult_income_gb_model.pkl'

SEMANTIC_WEIGHTS = np.array([0.746599, 0.057077, 0.031928, 0.022926, 0.017415,
                             0.013765, 0.010968, 0.008879, 0.007059, 0.005514,
                             0.003976, 0.002518, 0.001171, 0.0000001])

FEATURE_COLUMNS = ['age', 'workclass', 'fnlwgt', 'education', 'educational-num',
                   'marital-status', 'occupation', 'relationship', 'race',
                   'gender', 'capital-gain', 'capital-loss', 'hours-per-week',
                   'native-country']
CATEGORICAL_FEATS = ['workclass', 'education', 'marital-status', 'occupation',
                     'relationship', 'race', 'gender', 'native-country']
NUMERICAL_FEATS   = ['age', 'fnlwgt', 'educational-num', 'capital-gain',
                     'capital-loss', 'hours-per-week']
IMMUTABLE_FEATS   = ['age', 'gender', 'workclass', 'race', 'fnlwgt','education', 'native-country']

LAMBDA         = 0.5
PERTURB_FACTOR = 0.3
PERTURB_STEPS  = 10

NUM_PROJECTIONS = 15
NUM_TABLES      = 10
ALPHA           = 0.25
I_PROBES        = 1
Q_PROBES        = 6

NUM_QUERIES = 100
RANDOM_SEED = 42
ISO_CONTAMINATION = 0.10


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


def filter_for_diversity(arr, q_raw, nun, cat_idx, immutable_idx):
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
    filtered = [filter_for_diversity(cf, q_raw, nun, cat_idx, immutable_idx)
                for cf in candidates]
    filtered = [f for f in filtered if len(f) > 0]
    if len(filtered) < 2:
        return 0.0
    pairs = list(combinations(filtered, 2))
    dists = [hamming(p[0], p[1]) for p in pairs]
    return float(np.mean(dists)) * 100.0


def compute_plausibility(candidates, isf, scaler):
    if not candidates:
        return 0.0
    arr = np.array(candidates)
    arr_scaled = scaler.transform(arr)
    preds = isf.predict(arr_scaled)
    return float(np.mean(preds == 1)) * 100.0


def load_everything():
    data = pd.read_csv(DATA_PATH)
    X = data[FEATURE_COLUMNS].values
    y = data['target'].values

    X_train, _, _, _ = train_test_split(X, y, test_size=0.15, random_state=0)
    X_train, _, _, _ = train_test_split(X_train, y[:len(X_train)],
                                        test_size=0.10, random_state=0)
    scaler = StandardScaler()
    scaler.fit(X_train)

    model = joblib.load(MODEL_PATH)

    isf = IsolationForest(contamination=ISO_CONTAMINATION,
                          random_state=RANDOM_SEED)
    isf.fit(scaler.transform(X_train))

    cat_idx = [FEATURE_COLUMNS.index(f) for f in CATEGORICAL_FEATS]
    num_idx = [FEATURE_COLUMNS.index(f) for f in NUMERICAL_FEATS]
    immutable_idx = [FEATURE_COLUMNS.index(f) for f in IMMUTABLE_FEATS]

    ranges = np.zeros(len(FEATURE_COLUMNS))
    for i, f in enumerate(FEATURE_COLUMNS):
        if f in NUMERICAL_FEATS:
            ranges[i] = data[f].max() - data[f].min()

    return data, model, scaler, isf, cat_idx, num_idx, immutable_idx, ranges


def build_lsh_index(weighted_neighbors, dim):
    np.random.seed(RANDOM_SEED)
    projections = [np.random.randn(NUM_PROJECTIONS, dim) for _ in range(NUM_TABLES)]
    tables = [{} for _ in range(NUM_TABLES)]
    for t_idx in range(NUM_TABLES):
        proj = projections[t_idx]
        for p_idx, point in enumerate(weighted_neighbors):
            proj_vals = np.dot(proj, point)
            sorted_idx = np.argsort(-np.abs(proj_vals))
            for i in range(I_PROBES):
                h = int(sorted_idx[i])
                tables[t_idx].setdefault(h, []).append(p_idx)
    return projections, tables


def query_lsh(query_norm, projections, tables, weighted_neighbors):
    candidates = set()
    for t_idx, table in enumerate(tables):
        proj_vectors = projections[t_idx]
        proj_query = np.dot(proj_vectors, query_norm)
        sorted_idx_q = np.argsort(-np.abs(proj_query))
        for h in sorted_idx_q[:1 + Q_PROBES]:
            bucket = table.get(int(h), [])
            if not bucket:
                continue
            scored = []
            for idx in bucket:
                cand_vec = weighted_neighbors[idx]
                proj_cand = np.dot(proj_vectors, cand_vec)
                scored.append((idx, np.max(np.abs(proj_cand))))
            scored.sort(key=lambda x: x[1], reverse=True)
            num_keep = max(1, int(ALPHA * len(scored)))
            candidates.update([s[0] for s in scored[:num_keep]])
    return list(candidates)


def find_weighted_nun(q_raw, neighbors_raw, weighted_neighbors, q_norm,
                     projections, tables, cat_idx, num_idx, ranges):
    cand_idx = query_lsh(q_norm, projections, tables, weighted_neighbors)
    if not cand_idx:
        return None
    best_idx, best_dist = None, float('inf')
    for i in cand_idx:
        d = heom_distance(q_raw, neighbors_raw[i], cat_idx, num_idx, ranges)
        if d < best_dist:
            best_dist, best_idx = d, i
    return neighbors_raw[best_idx] if best_idx is not None else None


def generate_candidates(q_raw, nun, model, scaler, num_idx, immutable_idx):
    mutable_idx = [i for i in range(len(FEATURE_COLUMNS)) if i not in immutable_idx]
    feature_order = sorted(mutable_idx, key=lambda i: -SEMANTIC_WEIGHTS[i])

    cands = []
    cf = q_raw.copy().astype(float)
    orig_pred = model.predict(scaler.transform(q_raw.reshape(1, -1)))[0]
    sub_list = []

    for feat_idx in feature_order:
        cf[feat_idx] = nun[feat_idx]
        sub_list.append(feat_idx)
        if model.predict(scaler.transform(cf.reshape(1, -1)))[0] != orig_pred:
            cands.append(cf.copy())
        for num_feat_idx in sub_list:
            if num_feat_idx not in num_idx:
                continue
            curr = cf.copy()
            sub_val = nun[num_feat_idx]
            orig_val = q_raw[num_feat_idx]
            for _ in range(PERTURB_STEPS):
                new_val = sub_val + PERTURB_FACTOR * (orig_val - sub_val)
                curr[num_feat_idx] = new_val
                if model.predict(scaler.transform(curr.reshape(1, -1)))[0] != orig_pred:
                    cands.append(curr.copy())
                sub_val = new_val
    return cands


def dedup(cands):
    seen = set()
    out = []
    for cf in cands:
        key = tuple(cf.tolist())
        if key not in seen:
            seen.add(key)
            out.append(cf)
    return out


def main():
    (data, model, scaler, isf, cat_idx, num_idx,
     immutable_idx, ranges) = load_everything()

    # =========================================================
    # FIX: NUN pool and query pool use MODEL PREDICTIONS, not labels
    # =========================================================
    all_X = data[FEATURE_COLUMNS].values.astype(float)
    all_X_scaled = scaler.transform(all_X)
    all_preds = model.predict(all_X_scaled)

    pos_mask = (all_preds == 1)
    neg_mask = (all_preds == 0)

    print(f"Dataset size: {len(data)}")
    print(f"Model predicts as positive: {pos_mask.sum()} ({100*pos_mask.mean():.1f}%)")
    print(f"Model predicts as negative: {neg_mask.sum()} ({100*neg_mask.mean():.1f}%)")
    print(f"(Ground-truth: {(data['target']==1).sum()} positive, "
          f"{(data['target']==0).sum()} negative)\n")

    neighbors_raw = all_X[pos_mask]
    norm_neighbors = normalize(neighbors_raw)
    weighted_neighbors = norm_neighbors * SEMANTIC_WEIGHTS
    projections, tables = build_lsh_index(weighted_neighbors, len(FEATURE_COLUMNS))

    query_pool = all_X[neg_mask]
    n_take = min(NUM_QUERIES, len(query_pool))
    rng = np.random.default_rng(RANDOM_SEED)
    query_idx = rng.choice(len(query_pool), size=n_take, replace=False)
    queries_raw = query_pool[query_idx]
    queries_norm_for_hash = normalize(queries_raw)

    print(f"Running SPICE on {n_take} queries (LIME, all features)...")

    rows = []
    n_no_cand = 0
    for q_idx in range(n_take):
        if q_idx % 20 == 0 and q_idx > 0:
            print(f"  ...{q_idx}/{n_take}")

        q_raw  = queries_raw[q_idx]
        q_hash = queries_norm_for_hash[q_idx]

        nun = find_weighted_nun(q_raw, neighbors_raw, weighted_neighbors,
                                q_hash, projections, tables,
                                cat_idx, num_idx, ranges)
        if nun is None:
            n_no_cand += 1
            continue

        cands = generate_candidates(q_raw, nun, model, scaler,
                                    num_idx, immutable_idx)
        if not cands:
            n_no_cand += 1
            continue
        cands = dedup(cands)

        metrics = [(heom_distance(q_raw, cf, cat_idx, num_idx, ranges),
                    sparsity_ratio(q_raw, cf)) for cf in cands]
        proxs = [p for p, _ in metrics]
        spars = [s for _, s in metrics]
        best_spar = float(min(spars))
        best_prox = float(min(proxs))
        avg_spar  = float(np.mean(spars))
        avg_prox  = float(np.mean(proxs))

        div  = compute_diversity(cands, q_raw, nun, cat_idx, immutable_idx)
        plau = compute_plausibility(cands, isf, scaler)

        rows.append({
            'query_idx':       q_idx,
            'n_candidates':    len(cands),
            'best_sparsity':   best_spar,
            'best_proximity':  best_prox,
            'avg_sparsity':    avg_spar,
            'avg_proximity':   avg_prox,
            'diversity_pct':   div,
            'plausibility_pct': plau,
        })

    df = pd.DataFrame(rows)
    print(f"\nQueries with valid candidates: {len(df)}/{n_take}")
    print(f"Queries with no candidates:    {n_no_cand}/{n_take}")
    if len(df) > 0:
        print(f"Avg candidates per query:      {df['n_candidates'].mean():.1f}\n")

    if len(df) == 0:
        print("ERROR: no queries produced candidates.")
        return

    print("=" * 70)
    print("SPICE Table 5 Results - Adult Income (LIME, 100 queries)")
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

    df.to_csv('table5_spice_AdultIncome.csv', index=False)
    print(f"\nPer-query results saved to: table5_spice_AdultIncome.csv")


if __name__ == "__main__":
    main()