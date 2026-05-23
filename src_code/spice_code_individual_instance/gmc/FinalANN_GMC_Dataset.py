import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize
import time

# HEOM distance for numpy arrays with feature mapping
def heom_distance_vectorized(query_point, candidate_point, categorical_indices, numerical_indices, feature_ranges_array):
    distance = 0
    # Categorical features
    for idx in categorical_indices:
        distance += 1 if query_point[idx] != candidate_point[idx] else 0
    # Numerical features
    for idx in numerical_indices:
        range_val = feature_ranges_array[idx] if feature_ranges_array[idx] != 0 else 1
        distance += abs(query_point[idx] - candidate_point[idx]) / range_val
    return distance

# Load dataset
combined_data  = pd.read_csv('/Users/saikat/Desktop/Summer 2025/CFNet_AAAI_26/GMC Dataset/AISTATS GMC Dataset/preprocessed_credit_data.csv')

# Label encoding the binary categorical features
categorical_features = ['NumberOfDependents', 'NumberOfTime30-59DaysPastDueNotWorse', 'NumberRealEstateLoansOrLines'] 
numerical_features = ['RevolvingUtilizationOfUnsecuredLines', 'age', 'DebtRatio', 'MonthlyIncome', 'NumberOfOpenCreditLinesAndLoans', 'NumberOfTimes90DaysLate', 'NumberOfTime60-89DaysPastDueNotWorse']

# Filter the dataset where target == 0 for the query point
query_candidates = combined_data[combined_data['SeriousDlqin2yrs'] == 0]

# Select the 6th element from the filtered dataset (index 5)
query_point_data = query_candidates.iloc[10]

# Step 1: Preprocessing the data with semantic weighting
feature_columns = ['RevolvingUtilizationOfUnsecuredLines', 'age', 'NumberOfTime30-59DaysPastDueNotWorse', 'DebtRatio', 'MonthlyIncome', 'NumberOfOpenCreditLinesAndLoans', 'NumberOfTimes90DaysLate', 'NumberRealEstateLoansOrLines', 'NumberOfTime60-89DaysPastDueNotWorse', 'NumberOfDependents']

# Get indices for categorical and numerical features
categorical_indices = [feature_columns.index(f) for f in categorical_features]
numerical_indices = [feature_columns.index(f) for f in numerical_features]

# Compute feature ranges for HEOM distance
feature_ranges = {}
feature_ranges_array = np.zeros(len(feature_columns))
for i, feature in enumerate(feature_columns):
    if feature in numerical_features:
        feature_range = combined_data[feature].max() - combined_data[feature].min()
        feature_ranges[feature] = feature_range
        feature_ranges_array[i] = feature_range

print(f"Feature ranges for HEOM: {feature_ranges}")

# SEMANTIC ENHANCEMENT: Define importance weights for all features

#semantic_weights = np.array([0.839242, 0.259924, 0.393044, 0.200153, 0.131902, 0.188618, 0.250719, 0.104775, 0.145119, 0.059607]) # SHAP Method
#semantic_weights = np.array([0.168549, 0.080057, 0.064094, 0.032998, 0.018477, 0.012226, 0.014331, 0.006502, 0.006534, 0.002307]) # LIME Method
#semantic_weights = np.array([0.037074, 0.007276, 0.028447, 0.005027, 0.003193, 0.004644, 0.031869, 0.003548, 0.021735, 0.004704]) # MI Method
semantic_weights = np.array([0.2895, 1722.9974, 10328.1365, 10.9695, 44.0967, 103.5971, 13770.3610, 12.5579, 9505.8989, 256.6856]) # ANOVA Method


print(f"Semantic weights applied: {dict(zip(feature_columns, semantic_weights))}")

# Step 1: Extract all feature sets
dataset_features = combined_data[feature_columns].values
query_point_features = query_point_data[feature_columns].values
query_point_features = query_point_features.reshape(1, -1)
neighbor_candidates = combined_data[combined_data['SeriousDlqin2yrs'] == 1]
neighbor_features = neighbor_candidates[feature_columns].values

# Step 2: Normalize all features first (L2 normalization)
normalized_dataset_features = normalize(dataset_features)
normalized_query_point_features = normalize(query_point_features)[0]
normalized_neighbor_features = normalize(neighbor_features)

# Step 3: Apply semantic weighting to normalized features
def apply_semantic_weighting(features, weights):
    """Apply semantic weights to emphasize important features"""
    return features * weights

# Apply semantic weighting to normalized features
weighted_dataset_features = apply_semantic_weighting(normalized_dataset_features, semantic_weights)
weighted_query_point_features = apply_semantic_weighting(normalized_query_point_features, semantic_weights)
weighted_neighbor_features = apply_semantic_weighting(normalized_neighbor_features, semantic_weights)

print(f"Original query point: {query_point_features}")
print(f"Weighted query point: {weighted_query_point_features}")
print(f"Normalized query point: {normalized_query_point_features}")
print(f"Dataset shape after processing: {weighted_dataset_features.shape}")
print(f"Neighbor features shape after processing: {weighted_neighbor_features.shape}")

# Step 2: Generate random projection vectors (Cross-Polytope LSH)
# Parameters
num_projections_per_table = 15
num_feature_dimensions = weighted_neighbor_features.shape[1]
num_hash_tables = 10
alpha = .25
iProbes = 1
qProbes = 6
k = 200

# Generate random projection vectors for each hash table
np.random.seed(42)
random_projections = [np.random.randn(num_projections_per_table, num_feature_dimensions) for _ in range(num_hash_tables)]

# Step 3: Hash each point based on the max projection (Hashing into Buckets) with iProbes
def compute_hash_value_max_projection(data_point, projection_vectors):
    projection_result = np.dot(projection_vectors, data_point)
    max_projection_index = np.argmax(np.abs(projection_result))
    return max_projection_index

# Create the hash tables using points where target == 1
hash_tables = [{} for _ in range(num_hash_tables)]

for table_index in range(num_hash_tables):
    for point_index, data_point in enumerate(weighted_neighbor_features):
        projection_result = np.dot(random_projections[table_index], data_point)
        sorted_indices = np.argsort(-np.abs(projection_result))

        for i in range(iProbes):
            hash_value = sorted_indices[i]
            if hash_value not in hash_tables[table_index]:
                hash_tables[table_index][hash_value] = []
            hash_tables[table_index][hash_value].append(point_index)

# Step 4: Locality-Sensitive Filtering (LSF) with projection values
def apply_locality_sensitive_filtering(query_point, candidate_indices, projection_vectors, alpha):
    filtered_candidates = []
    num_to_keep = max(1, int(alpha * len(candidate_indices)))

    for candidate_index in candidate_indices:
        candidate_point = weighted_neighbor_features[candidate_index]
        projection_similarity = np.dot(projection_vectors, candidate_point)
        filtered_candidates.append((candidate_index, np.max(np.abs(projection_similarity))))
    
    filtered_candidates.sort(key=lambda x: x[1], reverse=True)
    filtered_candidates = [c[0] for c in filtered_candidates[:num_to_keep]]
    
    return filtered_candidates

# Step 5: Nearby Hash Generation based on Max Projections for qProbes
def generate_nearby_hash_values(query_point, projection_vectors, num_probes):
    projection_result = np.dot(projection_vectors, query_point)
    sorted_indices = np.argsort(-np.abs(projection_result))
    nearby_hashes = [sorted_indices[i] for i in range(1, min(num_probes + 1, len(sorted_indices)))]
    return nearby_hashes

# Step 6: Querying with multi-probing and filtering 
def multi_probe_query_with_filtering(query_point, random_projections, hash_tables, num_probes=6, alpha=0.5):
    candidate_indices = set()

    for table_index in range(len(hash_tables)):
        query_hash = compute_hash_value_max_projection(query_point, random_projections[table_index])
        primary_bucket = hash_tables[table_index].get(query_hash, [])
        print(f"Table {table_index}: Primary bucket hash value: {query_hash}, Candidates in primary bucket: {len(primary_bucket)}")

        if primary_bucket:
            primary_filtered = apply_locality_sensitive_filtering(query_point, primary_bucket, random_projections[table_index], alpha)
            candidate_indices.update(primary_filtered)

        nearby_hashes = generate_nearby_hash_values(query_point, random_projections[table_index], num_probes)

        for nearby_hash in nearby_hashes:
            nearby_bucket = hash_tables[table_index].get(nearby_hash, [])
            print(f"Table {table_index}: Nearby bucket hash value: {nearby_hash}, Candidates in nearby bucket: {len(nearby_bucket)}")
            if nearby_bucket:
                nearby_filtered = apply_locality_sensitive_filtering(query_point, nearby_bucket, random_projections[table_index], alpha)
                candidate_indices.update(nearby_filtered)

    return list(candidate_indices)

# Step 7: Compute exact distances using HEOM for final candidates
def compute_exact_distances_heom(query_point, filtered_candidates):
    distances = []
    for candidate_index in filtered_candidates:
        candidate_point = neighbor_features[candidate_index]  # Use original features for HEOM
        distance = heom_distance_vectorized(query_point_features[0], candidate_point, 
                                           categorical_indices, numerical_indices, feature_ranges_array)
        distances.append((candidate_index, distance))
    return sorted(distances, key=lambda x: x[1])

# Start the timer just before the ANN query
start_time = time.time()

filtered_candidate_indices = multi_probe_query_with_filtering(normalized_query_point_features, random_projections, hash_tables, num_probes=qProbes, alpha=alpha)

# Compute the nearest neighbors using HEOM distance
approx_nearest_neighbors = compute_exact_distances_heom(normalized_query_point_features, filtered_candidate_indices)

# Stop the timer after ANN computation
end_time = time.time()
time_taken_ms = (end_time - start_time) * 1000
print(f"Time taken to find the approximate nearest neighbors: {time_taken_ms:.2f} ms")

# Step 8: Compute ground truth using brute-force search with HEOM distance
def compute_ground_truth_nearest_neighbors_heom(query_point_orig, all_points_orig, k):
    distances = []
    for i, point in enumerate(all_points_orig):
        distance = heom_distance_vectorized(query_point_orig, point, 
                                           categorical_indices, numerical_indices, feature_ranges_array)
        distances.append((i, distance))
    return sorted(distances, key=lambda x: x[1])[:k]

# Compute the ground truth nearest neighbors using brute-force search with HEOM
ground_truth_neighbors = compute_ground_truth_nearest_neighbors_heom(query_point_features[0], neighbor_features, k)

# Step 9: Calculate recall
def calculate_recall(ground_truth, approx_neighbors):
    if not approx_neighbors:
        return 0.0
    
    ground_truth_set = set([idx for idx, _ in ground_truth])
    approx_neighbors_set = set([idx for idx, _ in approx_neighbors[:k]])
    num_correct = len(ground_truth_set.intersection(approx_neighbors_set))
    recall = num_correct / k
    return recall

# Calculate and print the recall
recall_value = calculate_recall(ground_truth_neighbors, approx_nearest_neighbors)
print(f"Recall: {recall_value:.2f}")

# Get the best nearest neighbor (ANNN)
if approx_nearest_neighbors:
    best_neighbor_index = approx_nearest_neighbors[0][0]
    best_neighbor_distance = approx_nearest_neighbors[0][1]
    best_neighbor_original_features = neighbor_features[best_neighbor_index]

     # Convert arrays into dict format with feature names
    query_instance = dict(zip(feature_columns, query_point_features[0]))
    nun_instance = dict(zip(feature_columns, best_neighbor_original_features))
    
    # Print in the required format
    print("\nquery_instance =", query_instance)
    print("\nnun_instance   =", nun_instance)
    
#print(f"Time taken to find the approximate nearest neighbors: {time_taken_ms:.2f} ms")
