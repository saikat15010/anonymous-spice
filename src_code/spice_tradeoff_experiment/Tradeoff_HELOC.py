import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
import joblib

# ============================================================
# CONFIG
# ============================================================

DATA_PATH = "/Users/saikat/Desktop/Summer 2025/CFNet/New Experiment/HELOC Dataset/heloc_dataset_v1.csv"
MODEL_PATH = "/Users/saikat/Desktop/Summer 2025/CFNet/New Experiment/HELOC Dataset/best_heloc_mlp_model.pkl"

LAMBDAS = [0.0, 0.5, 1.0]
PERTURB_STEPS = 8
DELTA = 0.3

# ============================================================
# LOAD DATA
# ============================================================

data = pd.read_csv(DATA_PATH)
data["target"] = data["target"].map({"Bad":0, "Good":1})

feature_order = [c for c in data.columns if c != "target"]

X = data[feature_order].values
y = data["target"].values

split = int(0.85 * len(X))

X_train = X[:split]
X_test = X[split:]
y_train = y[:split]
y_test = y[split:]

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

rf_model = joblib.load(MODEL_PATH)

# ============================================================
# FEATURE TYPES (HELOC)
# ============================================================

categorical_features = [
    "NumTrades60Ever2DerogPubRec",
    "NumTrades90Ever2DerogPubRec"
]

numerical_features = [f for f in feature_order if f not in categorical_features]

important_features = feature_order.copy()   # Already sorted by importance in your earlier setup

# ============================================================
# UTILITIES
# ============================================================

def inst_to_array(x):
    return np.array([x[f] for f in feature_order]).reshape(1,-1)

feature_ranges = {
    f: data[f].max() - data[f].min() if data[f].max() != data[f].min() else 1
    for f in numerical_features
}

def heom(x1, x2):

    d = 0.0

    for f in categorical_features:
        d += 1 if x1[f] != x2[f] else 0

    for f in numerical_features:
        d += abs(x1[f] - x2[f]) / feature_ranges[f]

    return d


def sparsity(q, cf):
    diff = 0
    for f in feature_order:
        if abs(q[f] - cf[f]) > 1e-6:
            diff += 1
    return diff / len(feature_order)

# ============================================================
# NUN SEARCH
# ============================================================

def find_nun(query):

    q_scaled = scaler.transform(inst_to_array(query))
    q_label = rf_model.predict(q_scaled)[0]

    pool = data[data["target"] != q_label]

    best = None
    best_d = 1e9

    for _, row in pool.iterrows():
        cand = {f: row[f] for f in feature_order}
        d = heom(query, cand)

        if d < best_d:
            best_d = d
            best = cand

    return best

# ============================================================
# SPICE CANDIDATE GENERATION
# ============================================================

def generate_spice(query, nun):

    y0 = rf_model.predict(
        scaler.transform(inst_to_array(query))
    )[0]

    results = []
    seen = set()

    for i in range(len(important_features)):

        base = query.copy()

        for j in range(i+1):
            f = important_features[j]
            base[f] = nun[f]

        pred = rf_model.predict(
            scaler.transform(inst_to_array(base))
        )[0]

        if pred != y0:
            key = tuple(round(base[f],6) for f in feature_order)
            if key not in seen:
                seen.add(key)
                results.append(base.copy())

        # perturb numeric
        pert = base.copy()

        for f in important_features[:i+1]:

            if f not in numerical_features:
                continue

            curr = pert[f]

            for _ in range(PERTURB_STEPS):

                curr = curr + DELTA * (query[f] - curr)
                pert[f] = curr

                pred2 = rf_model.predict(
                    scaler.transform(inst_to_array(pert))
                )[0]

                if pred2 != y0:
                    key = tuple(round(pert[k],6) for k in feature_order)
                    if key not in seen:
                        seen.add(key)
                        results.append(pert.copy())

    return results

# ============================================================
# LAMBDA SELECTION
# ============================================================

def best_cf(cfs, query, lam):

    best = None
    best_val = 1e9
    best_sr = None
    best_px = None

    for cf in cfs:

        px = heom(query, cf)
        sr = sparsity(query, cf)

        obj = lam * px + (1-lam) * sr

        if obj < best_val:
            best_val = obj
            best = cf
            best_sr = sr
            best_px = px

    return best, best_sr, best_px

# ============================================================
# FULL DATASET TRADEOFF SCAN
# ============================================================

print("\nScanning HELOC for SPICE trade-offs")
print("="*80)

tradeoffs = []

test_pool = data.iloc[split:][data.iloc[split:]["target"] == 0]

for idx in range(len(test_pool)):

    query = test_pool.iloc[idx][feature_order].to_dict()

    nun = find_nun(query)
    if nun is None:
        continue

    cfs = generate_spice(query, nun)

    if len(cfs) < 2:
        continue

    stats = {}

    for lam in LAMBDAS:
        _, sr, px = best_cf(cfs, query, lam)
        stats[lam] = (sr, px)

    srs = [v[0] for v in stats.values()]
    pxs = [v[1] for v in stats.values()]

    sr_diff = max(srs) - min(srs)
    px_diff = max(pxs) - min(pxs)

    if sr_diff >= (1/len(feature_order)) and px_diff > 0.05:

        tradeoffs.append({
            "query_id": idx,
            "num_cfs": len(cfs),
            "stats": stats,
            "sr_diff": sr_diff,
            "px_diff": px_diff,
            "score": sr_diff + px_diff
        })

# ============================================================
# SORT + REPORT
# ============================================================

tradeoffs.sort(key=lambda x: x["score"], reverse=True)

print("\nTOP 5 HELOC TRADEOFF QUERIES")
print("="*80)

top5 = tradeoffs[:5]

for r, item in enumerate(top5,1):

    print(f"\nRank {r} — Query #{item['query_id']}")
    print(f"Candidate Pool = {item['num_cfs']}")

    for lam in LAMBDAS:
        sr, px = item["stats"][lam]
        print(f"λ={lam:.1f}   Spar={sr:.3f}   Prox={px:.3f}")

    print(f"Δ Sparsity = {item['sr_diff']:.3f}")
    print(f"Δ Proximity = {item['px_diff']:.3f}")
    print(f"Tradeoff Score = {item['score']:.3f}")

# ============================================================
# BEST PAPER EXAMPLE
# ============================================================

if len(tradeoffs) > 0:

    best = tradeoffs[0]

    print("\nBEST HELOC TRADEOFF INSTANCE (FOR PAPER)")
    print("="*80)

    print(f"Query ID: {best['query_id']}")
    print(f"Candidate Pool Size: {best['num_cfs']}")

    for lam in LAMBDAS:
        sr, px = best["stats"][lam]
        print(f"λ={lam:.1f}   Spar={sr:.3f}   Prox={px:.3f}")

    print(f"\nΔ Sparsity = {best['sr_diff']:.3f}")
    print(f"Δ Proximity = {best['px_diff']:.3f}")
    print(f"Overall Tradeoff Score = {best['score']:.3f}")

else:
    print("\nNo HELOC trade-offs detected.")
