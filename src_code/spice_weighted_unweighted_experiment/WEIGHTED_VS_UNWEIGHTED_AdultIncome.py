"""
SPICE: Weighted vs Unweighted NUN Ablation
Dataset: Adult Income

Reports only mean values for:
  - Best Sparsity, Best Proximity, Avg Sparsity, Avg Proximity
"""

import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import normalize, StandardScaler
from sklearn.model_selection import train_test_split

DATA_PATH = '/Users/saikat/Desktop/Summer 2025/CFNet_AAAI_26/Adult Income Dataset/AISTATS Adult Income /processed_adult.csv'
MODEL_PATH = '/Users/saikat/Desktop/Summer 2025/CFNet_AAAI_26/Adult Income Dataset/AISTATS Adult Income /best_adult_income_gb_model.pkl'

# SHAP weights for Adult Income
SEMANTIC_WEIGHTS = np.array([0.546833, 0.062240, 0.071102, 0.022808, 0.470401,
                             0.228216, 0.271241, 0.978600, 0.032014, 0.126574,
                             0.454561, 0.124117, 0.297832, 0.030739])

FEATURE_COLUMNS = ['age', 'workclass', 'fnlwgt', 'education', 'educational-num',
                   'marital-status', 'occupation', 'relationship', 'race',
                   'gender', 'capital-gain', 'capital-loss', 'hours-per-week',
                   'native-country']
CATEGORICAL_FEATS = ['workclass', 'education', 'marital-status', 'occupation',
                     'relationship', 'race', 'gender', 'native-country']
NUMERICAL_FEATS   = ['age', 'fnlwgt', 'educational-num', 'capital-gain',
                     'capital-loss', 'hours-per-week']
IMMUTABLE_FEATS   = ['age', 'gender']

PERTURB_FACTOR = 0.3
PERTURB_STEPS  = 10
LAMBDA         = 0.5

NUM_PROJECTIONS = 15
NUM_TABLES      = 10
ALPHA           = 0.25
I_PROBES        = 1
Q_PROBES        = 6

TOP_K_FRAC = .30
NUM_QUERIES  = 100
RANDOM_SEED  = 42


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


def load_everything():
    data = pd.read_csv(DATA_PATH)
    X = data[FEATURE_COLUMNS].values
    y = data['target'].values

    # Match Evaluation.py: train_test_split with random_state=0
    X_train, _, _, _ = train_test_split(X, y, test_size=0.15, random_state=0)
    X_train, _, _, _ = train_test_split(X_train, y[:len(X_train)],
                                        test_size=0.10, random_state=0)
    scaler = StandardScaler()
    scaler.fit(X_train)

    model = joblib.load(MODEL_PATH)

    cat_idx = [FEATURE_COLUMNS.index(f) for f in CATEGORICAL_FEATS]
    num_idx = [FEATURE_COLUMNS.index(f) for f in NUMERICAL_FEATS]

    ranges = np.zeros(len(FEATURE_COLUMNS))
    for i, f in enumerate(FEATURE_COLUMNS):
        if f in NUMERICAL_FEATS:
            ranges[i] = data[f].max() - data[f].min()

    return data, model, scaler, cat_idx, num_idx, ranges


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


def find_unweighted_nun(q_raw, neighbors_raw, cat_idx, num_idx, ranges):
    best_idx, best_dist = None, float('inf')
    for i, p in enumerate(neighbors_raw):
        d = heom_distance(q_raw, p, cat_idx, num_idx, ranges)
        if d < best_dist:
            best_dist, best_idx = d, i
    return neighbors_raw[best_idx]


def generate_candidates(q_raw, nun, model, scaler,
                        cat_idx, num_idx, ranges, feature_order):
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


def select_best(metrics, lam):
    scored = [(lam * p + (1 - lam) * s, idx) for idx, (p, s) in enumerate(metrics)]
    scored.sort()
    return scored[0][1]


def main():
    data, model, scaler, cat_idx, num_idx, ranges = load_everything()

    neighbors_raw = data[data['target'] == 1][FEATURE_COLUMNS].values.astype(float)
    norm_neighbors = normalize(neighbors_raw)
    weighted_neighbors = norm_neighbors * SEMANTIC_WEIGHTS
    projections, tables = build_lsh_index(weighted_neighbors, len(FEATURE_COLUMNS))

    all_neg = data[data['target'] == 0]
    n_take = min(NUM_QUERIES, len(all_neg))
    queries_df = all_neg.sample(n=n_take, random_state=RANDOM_SEED)
    queries_raw = queries_df[FEATURE_COLUMNS].values.astype(float)
    queries_norm_for_hash = normalize(queries_raw)

    mutable_idx = [i for i, f in enumerate(FEATURE_COLUMNS)
                   if f not in IMMUTABLE_FEATS]
    top_k = max(1, int(np.ceil(TOP_K_FRAC * len(mutable_idx))))
    weighted_order = sorted(mutable_idx, key=lambda i: -SEMANTIC_WEIGHTS[i])[:top_k]
    rng = np.random.default_rng(RANDOM_SEED)

    print(f"Running ablation on {n_take} queries...")

    bsw, bsw_p, asw, asw_p = [], [], [], []
    bsu, bsu_p, asu, asu_p = [], [], [], []

    for q_idx in range(n_take):
        if q_idx % 20 == 0 and q_idx > 0:
            print(f"  ...{q_idx}/{n_take}")

        q_raw  = queries_raw[q_idx]
        q_hash = queries_norm_for_hash[q_idx]

        nun_w = find_weighted_nun(q_raw, neighbors_raw, weighted_neighbors,
                                  q_hash, projections, tables,
                                  cat_idx, num_idx, ranges)
        if nun_w is None:
            continue
        cands_w = generate_candidates(q_raw, nun_w, model, scaler,
                                      cat_idx, num_idx, ranges, weighted_order)
        if not cands_w:
            cands_w = [nun_w.copy()]
        metrics_w = [(heom_distance(q_raw, cf, cat_idx, num_idx, ranges),
                      sparsity_ratio(q_raw, cf)) for cf in cands_w]

        nun_u = find_unweighted_nun(q_raw, neighbors_raw, cat_idx, num_idx, ranges)
        unweighted_order = list(rng.choice(mutable_idx, size=top_k, replace=False))
        cands_u = generate_candidates(q_raw, nun_u, model, scaler,
                                      cat_idx, num_idx, ranges, unweighted_order)
        if not cands_u:
            cands_u = [nun_u.copy()]
        metrics_u = [(heom_distance(q_raw, cf, cat_idx, num_idx, ranges),
                      sparsity_ratio(q_raw, cf)) for cf in cands_u]

        bw = select_best(metrics_w, LAMBDA)
        bu = select_best(metrics_u, LAMBDA)

        bsw.append(metrics_w[bw][1])
        bsw_p.append(metrics_w[bw][0])
        asw.append(np.mean([s for _, s in metrics_w]))
        asw_p.append(np.mean([p for p, _ in metrics_w]))

        bsu.append(metrics_u[bu][1])
        bsu_p.append(metrics_u[bu][0])
        asu.append(np.mean([s for _, s in metrics_u]))
        asu_p.append(np.mean([p for p, _ in metrics_u]))

    print(f"\nQueries evaluated: {len(bsw)}\n")
    print("=" * 70)
    print("WEIGHTED vs UNWEIGHTED - Adult Income (SHAP, 100 queries)")
    print("=" * 70)
    print(f"{'Metric':<25} {'Weighted':>15} {'Unweighted':>15}")
    print("-" * 57)
    print(f"{'Best Sparsity':<25} {np.mean(bsw):>15.4f} {np.mean(bsu):>15.4f}")
    print(f"{'Best Proximity':<25} {np.mean(bsw_p):>15.4f} {np.mean(bsu_p):>15.4f}")
    print(f"{'Avg Sparsity':<25} {np.mean(asw):>15.4f} {np.mean(asu):>15.4f}")
    print(f"{'Avg Proximity':<25} {np.mean(asw_p):>15.4f} {np.mean(asu_p):>15.4f}")


if __name__ == "__main__":
    main()
