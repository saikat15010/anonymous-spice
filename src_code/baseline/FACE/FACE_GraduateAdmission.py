"""
FACE Evaluation for Table 5 (Main Results Table)
Dataset: Graduate Admission

Wraps the user's working FACE.py implementation with the 100-query evaluation
protocol from SPICE. Key choices:
  - Fresh RandomForest trained inside the script (matches the working FACE.py)
  - Same 100 model-predicted-negative queries as SPICE (seed=42)
  - StandardScaler applied to features (matches FACE.py)
  - FACE produces 1 CF per query; only Spar, Prox, Plausibility reported
    (Avg Spar, Avg Prox, Diversity will appear as "-" in the final table)
"""

import numpy as np
import pandas as pd
import time
import warnings
warnings.filterwarnings('ignore')

from sklearn.neighbors import NearestNeighbors, KernelDensity
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

# ============================================================
# CONFIG
# ============================================================

TRAIN_PATH = '/Users/saikat/Desktop/Summer 2025/CFNet_AAAI_26/Graduate Admission Dataset/AISTATS Graduate Admission/admission_train.csv'
TEST_PATH  = '/Users/saikat/Desktop/Summer 2025/CFNet_AAAI_26/Graduate Admission Dataset/AISTATS Graduate Admission/admission_test.csv'

FEATURE_COLUMNS    = ['GRE Score', 'TOEFL Score', 'University Rating',
                      'SOP', 'LOR', 'CGPA', 'Research']
# For HEOM metric computation (matches SPICE evaluation)
CATEGORICAL_FEATS  = ['University Rating', 'Research']
NUMERICAL_FEATS    = ['GRE Score', 'TOEFL Score', 'SOP', 'LOR', 'CGPA']

NUM_QUERIES       = 100
RANDOM_SEED       = 42
ISO_CONTAMINATION = 0.10

# FACE hyperparameters (from working FACE.py)
GRAPH_TYPE      = "knn"
K_NEIGHBORS     = 15
WEIGHT_FUNCTION = "inverse"
PRED_THRESHOLD  = 0.5


# ============================================================
# FACE class (copied verbatim from working FACE.py, minus prints)
# ============================================================

class FACE:
    def __init__(self, data, model, graph_type="knn", k_neighbors=10,
                 epsilon=0.5, weight_function="inverse",
                 distance_metric="euclidean", verbose=False):
        self.data = data
        self.model = model
        self.graph_type = graph_type
        self.k_neighbors = k_neighbors
        self.epsilon = epsilon
        self.weight_function = weight_function
        self.distance_metric = distance_metric
        self.verbose = verbose

        self.kde = KernelDensity(kernel='gaussian',
                                 bandwidth=self._estimate_bandwidth())
        self.kde.fit(data)
        self.densities = np.exp(self.kde.score_samples(data))

    def _estimate_bandwidth(self):
        n, d = self.data.shape
        return (n * (d + 2) / 4.) ** (-1. / (d + 4))

    def _weight_func(self, density):
        if self.weight_function == "negative_log":
            return -np.log(max(density, 1e-10))
        elif self.weight_function == "inverse":
            return 1.0 / max(density, 1e-10)
        else:
            return 1.0

    def _build_knn_graph(self):
        n_samples = len(self.data)
        nbrs = NearestNeighbors(
            n_neighbors=min(self.k_neighbors + 1, n_samples),
            metric=self.distance_metric, n_jobs=-1)
        nbrs.fit(self.data)
        distances, indices = nbrs.kneighbors(self.data)

        row_ind, col_ind, weights = [], [], []
        for i in range(n_samples):
            for j in range(1, min(self.k_neighbors + 1, indices.shape[1])):
                if j < indices.shape[1]:
                    neighbor_idx = indices[i, j]
                    dist = distances[i, j]
                    if dist > 0:
                        avg_density = (self.densities[i] +
                                       self.densities[neighbor_idx]) / 2
                        dw = self._weight_func(avg_density)
                        edge_weight = dist * dw
                        row_ind.extend([i, neighbor_idx])
                        col_ind.extend([neighbor_idx, i])
                        weights.extend([edge_weight, edge_weight])
        return csr_matrix((weights, (row_ind, col_ind)),
                          shape=(n_samples, n_samples))

    def _build_graph(self):
        if self.graph_type == "knn":
            return self._build_knn_graph()
        else:
            raise ValueError(f"Unsupported graph type: {self.graph_type}")

    def _get_candidate_targets(self, target_class, threshold=0.5):
        try:
            preds = self.model.predict_proba(self.data)
            if preds.shape[1] <= target_class:
                return []
            confs = preds[:, target_class]
            return [i for i in range(len(self.data)) if confs[i] >= threshold]
        except Exception:
            return []

    def generate_counterfactual(self, query_instance, target_class,
                                prediction_threshold=0.5):
        extended_data = np.vstack([self.data, query_instance.reshape(1, -1)])
        query_idx = len(self.data)

        orig_data = self.data
        orig_densities = self.densities

        try:
            self.data = extended_data
            query_density = np.exp(self.kde.score_samples(
                query_instance.reshape(1, -1)))[0]
            self.densities = np.concatenate([orig_densities, [query_density]])

            graph = self._build_graph()
            candidates = self._get_candidate_targets(target_class,
                                                     prediction_threshold)
            if not candidates:
                return None

            try:
                distances, _ = dijkstra(graph, directed=False,
                                        indices=query_idx,
                                        return_predecessors=True)
            except Exception:
                return None

            best_candidate = None
            best_distance = np.inf
            for c_idx in candidates:
                if c_idx < len(distances) and distances[c_idx] < best_distance \
                        and not np.isinf(distances[c_idx]):
                    best_candidate = c_idx
                    best_distance = distances[c_idx]

            if best_candidate is None or best_candidate >= len(orig_data):
                return None
            return orig_data[best_candidate]
        except Exception:
            return None
        finally:
            self.data = orig_data
            self.densities = orig_densities


# ============================================================
# Metric helpers (compatible with SPICE script's formulas)
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
    # ---- Load data (matches FACE.py) ----
    d1 = pd.read_csv(TRAIN_PATH)
    d2 = pd.read_csv(TEST_PATH)
    data = pd.concat([d1, d2], ignore_index=True)

    X = data[FEATURE_COLUMNS].values.astype(float)
    y = data['target'].values

    # Same split logic as FACE.py
    split_idx = int(0.85 * len(X))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    val_idx = int(0.10 * split_idx)
    X_val, X_train = X_train[:val_idx], X_train[val_idx:]
    y_val, y_train = y_train[:val_idx], y_train[val_idx:]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    # ---- Train fresh RF (matches FACE.py) ----
    rf = RandomForestClassifier(n_estimators=100, random_state=0)
    rf.fit(X_train_scaled, y_train)

    # ---- Isolation Forest for plausibility ----
    isf = IsolationForest(contamination=ISO_CONTAMINATION,
                          random_state=RANDOM_SEED)
    isf.fit(X_train_scaled)

    # ---- Same 100 queries as SPICE: model-predicted negatives, seed=42 ----
    all_X_scaled = scaler.transform(X)
    all_preds = rf.predict(all_X_scaled)
    neg_indices = np.where(all_preds == 0)[0]

    rng = np.random.default_rng(RANDOM_SEED)
    n_take = min(NUM_QUERIES, len(neg_indices))
    chosen = rng.choice(neg_indices, size=n_take, replace=False)

    # Keep both raw and scaled query versions
    queries_raw    = X[chosen]
    queries_scaled = all_X_scaled[chosen]

    print(f"Dataset size: {len(X)}")
    print(f"Train size:   {len(X_train)}")
    print(f"Model predicts negative: {neg_indices.sum() if isinstance(neg_indices, np.ndarray) else len(neg_indices)}")
    print(f"Drawing {n_take} queries (seed={RANDOM_SEED})\n")

    # ---- Setup FACE (built once on training data) ----
    face = FACE(
        data=X_train_scaled,
        model=rf,
        graph_type=GRAPH_TYPE,
        k_neighbors=K_NEIGHBORS,
        weight_function=WEIGHT_FUNCTION,
        verbose=False,
    )

    cat_idx = [FEATURE_COLUMNS.index(f) for f in CATEGORICAL_FEATS]
    num_idx = [FEATURE_COLUMNS.index(f) for f in NUMERICAL_FEATS]
    ranges = np.zeros(len(FEATURE_COLUMNS))
    for i, f in enumerate(FEATURE_COLUMNS):
        if f in NUMERICAL_FEATS:
            ranges[i] = data[f].max() - data[f].min()

    print(f"Running FACE on {n_take} queries (k={K_NEIGHBORS}, "
          f"weight={WEIGHT_FUNCTION})...")
    print("NOTE: FACE rebuilds the KNN graph for each query - this is slow.\n")

    rows = []
    n_fail = 0
    start_overall = time.time()

    for q_i in range(n_take):
        if q_i % 10 == 0 and q_i > 0:
            elapsed = time.time() - start_overall
            print(f"  ...{q_i}/{n_take} (elapsed {elapsed:.0f}s)")

        q_raw    = queries_raw[q_i]
        q_scaled = queries_scaled[q_i]

        # Original pred should be 0; target is 1
        orig_pred = rf.predict(q_scaled.reshape(1, -1))[0]
        target_class = 1 if orig_pred == 0 else 0

        cf_scaled = face.generate_counterfactual(
            q_scaled, target_class, prediction_threshold=PRED_THRESHOLD)

        if cf_scaled is None:
            n_fail += 1
            continue

        # Verify CF flips
        cf_pred = rf.predict(cf_scaled.reshape(1, -1))[0]
        if cf_pred != target_class:
            n_fail += 1
            continue

        # Inverse-transform CF back to raw space for HEOM/sparsity
        cf_raw = scaler.inverse_transform(cf_scaled.reshape(1, -1))[0]

        prox = heom_distance(q_raw, cf_raw, cat_idx, num_idx, ranges)
        spar = sparsity_ratio(q_raw, cf_raw)

        # Plausibility: 100% if IsolationForest predicts inlier on the CF (scaled)
        plau = 100.0 if isf.predict(cf_scaled.reshape(1, -1))[0] == 1 else 0.0

        rows.append({
            'query_idx':       q_i,
            'best_sparsity':   spar,
            'best_proximity':  prox,
            'plausibility_pct': plau,
        })

    df = pd.DataFrame(rows)
    elapsed_total = time.time() - start_overall
    print(f"\nQueries with valid CFs: {len(df)}/{n_take}")
    print(f"Failed queries:         {n_fail}/{n_take}")
    print(f"Total runtime:          {elapsed_total:.1f}s\n")

    if len(df) == 0:
        print("ERROR: no queries produced counterfactuals.")
        return

    print("=" * 70)
    print("FACE Table 5 Results - Graduate Admission (100 queries)")
    print("Reported as mean +/- std. FACE produces 1 CF per query, so Avg")
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

    # Show the dashes for missing metrics
    for l in ['Avg Spar.', 'Avg Prox.', 'Div. (%)']:
        print(f"{l:<15} {'-':>12} {'-':>12} {'-':>20}")

    df.to_csv('table5_face_GraduateAdmission.csv', index=False)
    print(f"\nPer-query results saved to: table5_face_GraduateAdmission.csv")


if __name__ == "__main__":
    main()
