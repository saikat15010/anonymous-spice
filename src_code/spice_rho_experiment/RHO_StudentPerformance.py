"""
SPICE: Quantitative Approximation Error Study
Dataset: Student Performance

Reports:
  - Exact NUN recovery rate
  - Mean c, Median c

Note: Student Performance has features on highly different scales (age: 15-21,
G1/G2: 9-19, while binary features are 0-1). After L2 normalization, age and
G1/G2 absorb ~89% of the energy regardless of weights, which makes LSH hashing
effectively low-dimensional. We apply min-max scaling to put all features on
[0,1] before L2 normalization, and use log-scaled ANOVA weights to preserve
importance rank without letting top features dominate.

This preprocessing is applied only to the LSH search space; the final HEOM
distance over the original features is unchanged, so SPICE's counterfactual
ordering remains identical.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize, MinMaxScaler

DATA_PATH = '/Users/saikat/Desktop/Summer 2025/CFNet_AAAI_26/Student Performance Datastet/processed_student.csv'

RAW_WEIGHTS = np.array([
    4.4797, 49.0592, 27.6233, 28.7447, 0.3854, 66.1713,
    16.8917, 2.7596, 7.3206, 4.9569, 2.3510, 13.4245,
    650.8234, 720.1519
])

# Log scaling preserves rank order while compressing dynamic range
SEMANTIC_WEIGHTS = np.log1p(RAW_WEIGHTS)

FEATURE_COLUMNS = ['age', 'Medu', 'Fedu', 'studytime', 'famsup', 'higher',
                   'internet', 'romantic', 'freetime', 'goout', 'health',
                   'absences', 'G1', 'G2']
CATEGORICAL_FEATS = ['famsup', 'higher', 'internet', 'romantic']
NUMERICAL_FEATS   = ['age', 'Medu', 'Fedu', 'studytime', 'freetime', 'goout',
                     'health', 'absences', 'G1', 'G2']

NUM_PROJECTIONS = 15
NUM_TABLES      = 10
ALPHA           = 0.20
I_PROBES        = 1
Q_PROBES        = 2

NUM_TEST_QUERIES = 100
RANDOM_SEED      = 42


def heom_distance(query, candidate, cat_idx, num_idx, ranges):
    d = 0.0
    for i in cat_idx:
        d += 1.0 if query[i] != candidate[i] else 0.0
    for i in num_idx:
        rng = ranges[i] if ranges[i] != 0 else 1
        d += abs(query[i] - candidate[i]) / rng
    return d


def load_and_preprocess():
    combined = pd.read_csv(DATA_PATH)
    for f in CATEGORICAL_FEATS:
        combined[f] = combined[f].map({'yes': 1, 'no': 0})

    cat_idx = [FEATURE_COLUMNS.index(f) for f in CATEGORICAL_FEATS]
    num_idx = [FEATURE_COLUMNS.index(f) for f in NUMERICAL_FEATS]

    ranges = np.zeros(len(FEATURE_COLUMNS))
    for i, f in enumerate(FEATURE_COLUMNS):
        if f in NUMERICAL_FEATS:
            ranges[i] = combined[f].max() - combined[f].min()

    neighbors_raw = combined[combined['target'] == 1][FEATURE_COLUMNS].values.astype(float)

    all_negatives = combined[combined['target'] == 0]
    n_take = min(NUM_TEST_QUERIES, len(all_negatives))
    queries_raw = all_negatives.sample(n=n_take, random_state=RANDOM_SEED)[FEATURE_COLUMNS].values.astype(float)

    # Min-max scale using the full combined data so the [0,1] mapping is consistent
    mm = MinMaxScaler()
    mm.fit(combined[FEATURE_COLUMNS].values.astype(float))
    neighbors_mm = mm.transform(neighbors_raw)
    queries_mm   = mm.transform(queries_raw)

    norm_neighbors = normalize(neighbors_mm)
    norm_queries   = normalize(queries_mm)
    weighted_neighbors = norm_neighbors * SEMANTIC_WEIGHTS

    return {
        'neighbors_raw':      neighbors_raw,
        'queries_raw':        queries_raw,
        'weighted_neighbors': weighted_neighbors,
        'norm_queries':       norm_queries,
        'cat_idx':            cat_idx,
        'num_idx':            num_idx,
        'ranges':             ranges,
        'dim':                len(FEATURE_COLUMNS),
    }


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
        proj_query   = np.dot(proj_vectors, query_norm)
        sorted_idx_q = np.argsort(-np.abs(proj_query))
        buckets_to_check = sorted_idx_q[:1 + Q_PROBES]

        for h in buckets_to_check:
            bucket = table.get(int(h), [])
            if not bucket:
                continue
            scored = []
            for idx in bucket:
                cand_vec = weighted_neighbors[idx]
                proj_cand = np.dot(proj_vectors, cand_vec)
                score = np.max(np.abs(proj_cand))
                scored.append((idx, score))
            scored.sort(key=lambda x: x[1], reverse=True)
            num_keep = max(1, int(ALPHA * len(scored)))
            candidates.update([s[0] for s in scored[:num_keep]])

    return list(candidates)


def main():
    data = load_and_preprocess()
    projections, tables = build_lsh_index(data['weighted_neighbors'], data['dim'])

    c_values = []
    for q_idx in range(len(data['queries_raw'])):
        q_raw  = data['queries_raw'][q_idx]
        q_norm = data['norm_queries'][q_idx]

        cand_idx = query_lsh(q_norm, projections, tables, data['weighted_neighbors'])
        if not cand_idx:
            continue

        approx_dist = min(
            heom_distance(q_raw, data['neighbors_raw'][i],
                          data['cat_idx'], data['num_idx'], data['ranges'])
            for i in cand_idx
        )
        true_dist = min(
            heom_distance(q_raw, p,
                          data['cat_idx'], data['num_idx'], data['ranges'])
            for p in data['neighbors_raw']
        )

        c = 1.0 if true_dist < 1e-9 else approx_dist / true_dist
        c_values.append(c)

    if not c_values:
        print("No valid queries produced candidates.")
        return

    c_arr = np.array(c_values)
    exact_recovery = np.sum(c_arr == 1.0) / len(c_arr) * 100

    print("=" * 60)
    print("APPROXIMATION ERROR REPORT - Student Performance")
    print("=" * 60)
    print(f"Queries evaluated:        {len(c_values)}")
    print(f"Exact NUN recovery rate:  {exact_recovery:.1f}%")
    print(f"Mean c:                   {c_arr.mean():.4f}")
    print(f"Median c:                 {np.median(c_arr):.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()