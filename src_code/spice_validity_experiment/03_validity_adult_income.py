import os
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier

BASE = "/Users/saikat/Desktop/Summer 2025/CFNet_AAAI_26"
RESULT_DIR = "results"
os.makedirs(RESULT_DIR, exist_ok=True)

DEFAULT_N_QUERIES = 100
TOP_K = 6
LAMBDA_VALUE = 0.5
PERTURBATION_ROUNDS = 10
DELTA = 0.1
RANDOM_STATE = 42


def instance_to_array(instance, feature_order):
    return np.array([instance[f] for f in feature_order]).reshape(1, -1)


def predict_label(model, scaler, instance, feature_order):
    x = instance_to_array(instance, feature_order)
    return model.predict(scaler.transform(x))[0]


def calculate_sparsity(query_instance, cf_instance, feature_order, immutable_features):
    mutable = [f for f in feature_order if f not in immutable_features]
    if not mutable:
        return 0.0
    changed = sum(query_instance[f] != cf_instance[f] for f in mutable)
    return changed / len(mutable)


def calculate_heom(instance1, instance2, feature_order, categorical_features, numerical_features, feature_ranges):
    distance = 0.0
    for f in feature_order:
        if f in categorical_features:
            distance += 1.0 if instance1[f] != instance2[f] else 0.0
        elif f in numerical_features:
            rng = feature_ranges[f] if feature_ranges[f] != 0 else 1.0
            distance += abs(instance1[f] - instance2[f]) / rng
    return distance


def objective_value(query_instance, cf_instance, feature_order, categorical_features, numerical_features, feature_ranges, lambda_value, immutable_features):
    prox = calculate_heom(query_instance, cf_instance, feature_order, categorical_features, numerical_features, feature_ranges)
    spar = calculate_sparsity(query_instance, cf_instance, feature_order, immutable_features)
    return lambda_value * prox + (1.0 - lambda_value) * spar


def perturb_toward_query(original_value, substituted_value, delta=0.1):
    return substituted_value + delta * (original_value - substituted_value)


def make_feature_ranking(feature_order, semantic_weights, immutable_features, top_k):
    ranked = sorted(
        [(f, w) for f, w in zip(feature_order, semantic_weights) if f not in immutable_features],
        key=lambda x: x[1],
        reverse=True,
    )
    return [f for f, _ in ranked][:top_k]


def find_exact_nun_fast(query_instance, data_df, model, scaler, feature_order, categorical_features, numerical_features, feature_ranges, immutable_features):
    query_label = predict_label(model, scaler, query_instance, feature_order)
    candidate_df = data_df[feature_order].copy()

    for f in immutable_features:
        candidate_df = candidate_df[candidate_df[f] == query_instance[f]]

    if candidate_df.empty:
        return None, float("inf")

    candidate_values = candidate_df[feature_order].values
    candidate_pred = model.predict(scaler.transform(candidate_values))
    unlike_df = candidate_df[candidate_pred != query_label]

    if unlike_df.empty:
        return None, float("inf")

    distances = []
    for _, row in unlike_df.iterrows():
        cand = row[feature_order].to_dict()
        dist = calculate_heom(query_instance, cand, feature_order, categorical_features, numerical_features, feature_ranges)
        distances.append(dist)

    best_pos = int(np.argmin(distances))
    best_row = unlike_df.iloc[best_pos]
    return best_row[feature_order].to_dict(), distances[best_pos]


def run_validity_experiment(dataset_name, data_df, model, scaler, feature_order, target_col, negative_label, categorical_features, numerical_features, semantic_weights, immutable_features, top_k, n_queries, lambda_value, perturbation_rounds, delta):
    feature_ranges = {f: data_df[f].max() - data_df[f].min() for f in numerical_features}
    ranked_features = make_feature_ranking(feature_order, semantic_weights, immutable_features, top_k)

    query_pool = data_df[data_df[target_col] == negative_label].copy()
    n_queries = min(n_queries, len(query_pool))
    query_indices = query_pool.sample(n_queries, random_state=RANDOM_STATE).index

    total_trials = 0
    successful_trials = 0
    total_queries_with_cf = 0
    nun_improvement_count = 0
    nun_missing_count = 0
    per_query_results = []

    for idx in query_indices:
        query_instance = data_df.loc[idx, feature_order].to_dict()
        query_pred = predict_label(model, scaler, query_instance, feature_order)

        nun_instance, nun_distance = find_exact_nun_fast(
            query_instance, data_df, model, scaler, feature_order,
            categorical_features, numerical_features, feature_ranges, immutable_features
        )

        if nun_instance is None:
            nun_missing_count += 1
            continue

        nun_obj = objective_value(query_instance, nun_instance, feature_order, categorical_features, numerical_features, feature_ranges, lambda_value, immutable_features)
        candidate_list = [nun_instance.copy()]

        for i in range(len(ranked_features)):
            current_features = ranked_features[:i + 1]
            trial_instance = query_instance.copy()
            for f in current_features:
                trial_instance[f] = nun_instance[f]

            total_trials += 1
            trial_pred = predict_label(model, scaler, trial_instance, feature_order)

            if trial_pred != query_pred:
                successful_trials += 1
                candidate_list.append(trial_instance.copy())

                for f in current_features:
                    if f not in numerical_features:
                        continue
                    perturbed_instance = trial_instance.copy()
                    substituted_value = trial_instance[f]
                    original_value = query_instance[f]

                    for _ in range(perturbation_rounds):
                        perturbed_value = perturb_toward_query(original_value, substituted_value, delta)
                        perturbed_instance[f] = perturbed_value
                        total_trials += 1
                        perturbed_pred = predict_label(model, scaler, perturbed_instance, feature_order)

                        if perturbed_pred != query_pred:
                            successful_trials += 1
                            candidate_list.append(perturbed_instance.copy())
                        else:
                            break
                        substituted_value = perturbed_value

        unique_candidates = []
        seen = set()
        for cand in candidate_list:
            key = tuple((f, cand[f]) for f in feature_order)
            if key not in seen:
                seen.add(key)
                unique_candidates.append(cand)

        if unique_candidates:
            total_queries_with_cf += 1

        best_obj = min(objective_value(query_instance, cand, feature_order, categorical_features, numerical_features, feature_ranges, lambda_value, immutable_features) for cand in unique_candidates)
        improved = best_obj + 1e-12 < nun_obj
        if improved:
            nun_improvement_count += 1

        per_query_results.append({
            "query_index": idx,
            "num_candidates": len(unique_candidates),
            "nun_objective": nun_obj,
            "best_objective": best_obj,
            "improved_over_nun": improved,
        })

    trial_validity_rate = 100.0 * successful_trials / total_trials if total_trials else 0.0
    coverage_rate = 100.0 * total_queries_with_cf / n_queries if n_queries else 0.0
    nun_improvement_rate = 100.0 * nun_improvement_count / max(1, len(per_query_results))

    print("\n" + "=" * 70)
    print(f"{dataset_name} VALIDITY AND NUN IMPROVEMENT REPORT")
    print("=" * 70)
    print(f"Queries requested:              {n_queries}")
    print(f"Queries evaluated:              {len(per_query_results)}")
    print(f"Queries without feasible NUN:   {nun_missing_count}")
    print(f"Top-k mutable features used:    {top_k}")
    print(f"Total trials:                   {total_trials}")
    print(f"Successful trials:              {successful_trials}")
    print(f"Trial validity rate:            {trial_validity_rate:.2f}%")
    print(f"Coverage rate:                  {coverage_rate:.2f}%")
    print(f"NUN improvement rate:           {nun_improvement_rate:.2f}%")
    print("=" * 70)

    summary = pd.DataFrame([{
        "dataset": dataset_name,
        "queries_requested": n_queries,
        "queries_evaluated": len(per_query_results),
        "queries_without_feasible_nun": nun_missing_count,
        "top_k": top_k,
        "total_trials": total_trials,
        "successful_trials": successful_trials,
        "trial_validity_rate": trial_validity_rate,
        "coverage_rate": coverage_rate,
        "nun_improvement_rate": nun_improvement_rate,
    }])

    safe_name = dataset_name.replace(" ", "_").lower()
    summary.to_csv(f"{RESULT_DIR}/{safe_name}_summary.csv", index=False)
    pd.DataFrame(per_query_results).to_csv(f"{RESULT_DIR}/{safe_name}_per_query_results.csv", index=False)
    print(f"Saved: {RESULT_DIR}/{safe_name}_summary.csv")
    print(f"Saved: {RESULT_DIR}/{safe_name}_per_query_results.csv")


if __name__ == "__main__":
    data = pd.read_csv(BASE + "/Adult Income Dataset/AISTATS Adult Income /processed_adult.csv")

    feature_order = [
        "age", "workclass", "fnlwgt", "education", "educational-num",
        "marital-status", "occupation", "relationship", "race", "gender",
        "capital-gain", "capital-loss", "hours-per-week", "native-country"
    ]

    X = data[feature_order].values
    y = data["target"].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=0)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    model = GradientBoostingClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        random_state=0,
    )
    model.fit(X_train_scaled, y_train)

    semantic_weights = np.array([
        2021.1135, 3.5984, 1.8878, 240.5079,
        4399.2564, 1349.7858, 86.8274, 2381.2899,
        175.5655, 1708.3080, 1816.0313, 770.3146,
        1866.2539, 21.3586
    ])

    run_validity_experiment(
        dataset_name="Adult Income",
        data_df=data,
        model=model,
        scaler=scaler,
        feature_order=feature_order,
        target_col="target",
        negative_label=0,
        categorical_features=[
            "workclass", "education", "marital-status", "occupation",
            "relationship", "race", "gender", "native-country"
        ],
        numerical_features=["age", "fnlwgt", "educational-num", "capital-gain", "capital-loss", "hours-per-week"],
        semantic_weights=semantic_weights,
        immutable_features=["age", "gender"],
        top_k=TOP_K,
        n_queries=DEFAULT_N_QUERIES,
        lambda_value=LAMBDA_VALUE,
        perturbation_rounds=PERTURBATION_ROUNDS,
        delta=DELTA,
    )
