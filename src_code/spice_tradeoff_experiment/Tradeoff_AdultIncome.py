import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
import joblib

# ============================================================
# CONFIG
# ============================================================

DATA_PATH  = '/Users/saikat/Desktop/Summer 2025/CFNet_AAAI_26/Adult Income Dataset/AISTATS Adult Income /processed_adult.csv'
MODEL_PATH = '/Users/saikat/Desktop/Summer 2025/CFNet_AAAI_26/Adult Income Dataset/AISTATS Adult Income /best_adult_income_gb_model.pkl'

LAMBDAS = [0.0, 0.5, 1.0]
PERTURB_STEPS = 25
DELTA = 0.45
#PERTURB_STEPS = 10
#DELTA = 0.30

# ============================================================
# LOAD DATA
# ============================================================

data = pd.read_csv(DATA_PATH)

feature_order = [
    'age','workclass','fnlwgt','education','educational-num',
    'marital-status','occupation','relationship','race','gender',
    'capital-gain','capital-loss','hours-per-week','native-country'
]

X = data[feature_order].values
y = data['target'].values

# 15% test split (same as your training code)
split_idx = int(0.15 * len(X))

X_test = X[:split_idx]
y_test = y[:split_idx]

X_train = X[split_idx:]
y_train = y[split_idx:]

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ============================================================
# LOAD MODEL
# ============================================================

rf_model = joblib.load(MODEL_PATH)

# ============================================================
# FEATURE TYPES
# ============================================================

immutable_features = ['age','gender']

categorical_features = [
    'workclass','education','marital-status','occupation',
    'relationship','race','native-country'
]

numerical_features = [
    'fnlwgt','educational-num','capital-gain',
    'capital-loss','hours-per-week'
]

important_features = [
    f for f in feature_order if f not in immutable_features
]

# ============================================================
# UTILITIES
# ============================================================

def instance_to_array(instance):
    return np.array([instance[f] for f in feature_order]).reshape(1,-1)


feature_ranges = {
    f: data[f].max() - data[f].min() for f in numerical_features
}


def heom_distance(x1, x2):

    d = 0.0

    for f in categorical_features:
        d += 1 if x1[f] != x2[f] else 0

    for f in numerical_features:
        r = feature_ranges[f] if feature_ranges[f] != 0 else 1
        d += abs(x1[f] - x2[f]) / r

    return d


def sparsity_ratio(q, cf, eps=1e-6):

    changed = 0

    for f in feature_order:
        if abs(q[f] - cf[f]) > eps:
            changed += 1

    return changed / len(feature_order)


# ============================================================
# NUN SEARCH
# ============================================================

def find_nun(query):

    q_scaled = scaler.transform(instance_to_array(query))
    q_label = rf_model.predict(q_scaled)[0]

    pool = data[data['target'] != q_label]

    best = None
    best_dist = float("inf")

    for _, row in pool.iterrows():

        cand = {f: row[f] for f in feature_order}
        dist = heom_distance(query, cand)

        if dist < best_dist:
            best_dist = dist
            best = cand

    return best


# ============================================================
# SPICE CANDIDATE GENERATION
# ============================================================

def generate_counterfactuals(query, nun):

    y0 = rf_model.predict(
        scaler.transform(instance_to_array(query))
    )[0]

    candidates = []
    seen = set()

    for i in range(len(important_features)):

        base = query.copy()

        # Sequential substitution
        for j in range(i+1):
            f = important_features[j]
            base[f] = nun[f]

        pred = rf_model.predict(
            scaler.transform(instance_to_array(base))
        )[0]

        if pred != y0:
            key = tuple(round(base[f],6) for f in feature_order)
            if key not in seen:
                seen.add(key)
                candidates.append(base.copy())

        # Iterative perturbation
        perturbed = base.copy()

        for f in important_features[:i+1]:

            if f not in numerical_features:
                continue

            curr = perturbed[f]

            for _ in range(PERTURB_STEPS):

                curr = curr + DELTA * (query[f] - curr)
                perturbed[f] = curr

                pred2 = rf_model.predict(
                    scaler.transform(instance_to_array(perturbed))
                )[0]

                if pred2 != y0:

                    key = tuple(round(perturbed[k],6) for k in feature_order)

                    if key not in seen:
                        seen.add(key)
                        candidates.append(perturbed.copy())

    return candidates


# ============================================================
# LAMBDA OPTIMIZATION
# ============================================================

def select_best(candidates, query, lam):

    best_cf = None
    best_sr = None
    best_px = None
    best_val = float("inf")

    for cf in candidates:

        px = heom_distance(query, cf)
        sr = sparsity_ratio(query, cf)

        obj = lam * px + (1-lam) * sr

        if obj < best_val:
            best_val = obj
            best_cf = cf
            best_sr = sr
            best_px = px

    return best_cf, best_sr, best_px


# ============================================================
# FULL DATASET TRADE-OFF SCAN
# ============================================================

print("\nScanning Adult Income dataset for SPICE trade-offs")
print("="*80)

test_queries = data.iloc[:split_idx]      # same test region
test_queries = test_queries[test_queries['target'] == 0]

tradeoff_results = []

for idx in range(len(test_queries)):

    query = test_queries.iloc[idx][feature_order].to_dict()

    nun = find_nun(query)

    if nun is None:
        continue

    candidates = generate_counterfactuals(query, nun)

    if len(candidates) < 2:
        continue

    lambda_stats = {}

    for lam in LAMBDAS:
        _, sr, px = select_best(candidates, query, lam)
        lambda_stats[lam] = (sr, px)

    srs = [v[0] for v in lambda_stats.values()]
    pxs = [v[1] for v in lambda_stats.values()]

    sr_diff = max(srs) - min(srs)
    px_diff = max(pxs) - min(pxs)

    if sr_diff >= (1/len(feature_order)) and px_diff > 0.05:

        score = sr_diff + px_diff

        tradeoff_results.append({
            "query_id": idx,
            "num_candidates": len(candidates),
            "lambda_results": lambda_stats,
            "sr_diff": sr_diff,
            "px_diff": px_diff,
            "score": score
        })


# ============================================================
# SORT + REPORT
# ============================================================

tradeoff_results.sort(key=lambda x: x["score"], reverse=True)

print("\nTOP 5 ADULT INCOME TRADE-OFF QUERIES")
print("="*80)

top5 = tradeoff_results[:5]

for rank, item in enumerate(top5, 1):

    print(f"\nRank {rank} — Query #{item['query_id']}")
    print(f"Candidate Pool Size: {item['num_candidates']}")

    for lam in LAMBDAS:
        sr, px = item["lambda_results"][lam]
        print(f"λ={lam:.1f}   Spar={sr:.3f}   Prox={px:.3f}")

    print(f"Δ Sparsity = {item['sr_diff']:.3f}")
    print(f"Δ Proximity = {item['px_diff']:.3f}")
    print(f"Tradeoff Score = {item['score']:.3f}")


# ============================================================
# BEST PAPER EXAMPLE
# ============================================================

if len(tradeoff_results) > 0:

    best = tradeoff_results[0]

    print("\nBEST ADULT INCOME TRADE-OFF INSTANCE (USE IN PAPER)")
    print("="*80)

    print(f"Query ID: {best['query_id']}")
    print(f"Candidate Pool Size: {best['num_candidates']}")

    for lam in LAMBDAS:
        sr, px = best["lambda_results"][lam]
        print(f"λ={lam:.1f}   Spar={sr:.3f}   Prox={px:.3f}")

    print(f"\nΔ Sparsity = {best['sr_diff']:.3f}")
    print(f"Δ Proximity = {best['px_diff']:.3f}")
    print(f"Overall Tradeoff Score = {best['score']:.3f}")

else:
    print("\nNo Adult Income trade-offs detected.")
