import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
import time
from itertools import combinations
from scipy.spatial.distance import hamming
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import joblib
from scipy import stats


# Load dataset
data1 = pd.read_csv('/Users/saikat/Desktop/Summer 2025/CFNet_AAAI_26/Student Performance Datastet/AISTATS Student Performance/student_train.csv')
data2 = pd.read_csv('/Users/saikat/Desktop/Summer 2025/CFNet_AAAI_26/Student Performance Datastet/AISTATS Student Performance/student_test.csv')

# Combine datasets
data = pd.concat([data1, data2], ignore_index=True)

# Display the combined dataset
print(data.head())
print(data.shape)

# Check the distribution of the target feature
target_distribution = data['target'].value_counts()

# Display the distribution
print(target_distribution)

# Label encoding the binary categorical features
binary_features = ['famsup', 'higher', 'internet', 'romantic']

for feature in binary_features:
    data[feature] = data[feature].map({'yes': 1, 'no': 0})

print(data.head())

# Define features and target
X = data.drop(columns=['target']).values
y = data['target'].values

# Compute 15% split index
split_idx = int(0.15 * len(X))

# First 15% → test set
X_test = X[:split_idx]
y_test = y[:split_idx]

# Remaining 85% → train set
X_train = X[split_idx:]
y_train = y[split_idx:]

# Standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train Logistic Regression model
rf_model = LogisticRegression(max_iter=1000, random_state=0)
rf_model.fit(X_train, y_train)

# Save the trained model
model_path = '/Users/saikat/Desktop/Summer 2025/CFNet_AAAI_26/Student Performance Datastet/AISTATS Student Performance/best_admission_lr_model.pkl'
joblib.dump(rf_model, model_path)

# Load the best model
rf_model = joblib.load(model_path)

# Evaluate model performance on the test set
y_pred_test = rf_model.predict(X_test)
accuracy_test = accuracy_score(y_test, y_pred_test)
conf_matrix_test = confusion_matrix(y_test, y_pred_test)
class_report_test = classification_report(y_test, y_pred_test)

print(f'Accuracy on Test Set: {accuracy_test:.2f}')
print('Confusion Matrix:')
print(conf_matrix_test)
print('Classification Report:')
print(class_report_test)


# Query and NUN instance from the previous results
query_instance = {'age': 15.0, 'Medu': 2.0, 'Fedu': 2.0, 'studytime': 1.0, 'famsup': 1.0, 'higher': 1.0, 'internet': 1.0, 'romantic': 0.0, 'freetime': 4.0, 'goout': 1.0, 'health': 1.0, 'absences': 0.0, 'G1': 9.0, 'G2': 10.0}

nun_instance   = {'age': 17.0, 'Medu': 2.0, 'Fedu': 2.0, 'studytime': 2.0, 'famsup': 1.0, 'higher': 1.0, 'internet': 1.0, 'romantic': 0.0, 'freetime': 5.0, 'goout': 2.0, 'health': 1.0, 'absences': 0.0, 'G1': 12.0, 'G2': 13.0}



# Immutable feature
immutable_features = ['age']

# Helper function to convert instance dictionary to array
def instance_to_array(instance, feature_order):
    return np.array([instance[feature] for feature in feature_order]).reshape(1, -1)

# Function to perturb numerical feature
def perturb_numerical_feature(original_value, substituted_value, perturbation_factor=0.1):
    return substituted_value + perturbation_factor * (original_value - substituted_value)

# Features ordered by importance
important_features = [f for f in ['age', 'Medu', 'Fedu', 'studytime', 'famsup', 'higher', 'internet', 'romantic', 'freetime', 'goout', 'health', 'absences', 'G1', 'G2'] if f not in immutable_features]


# Define categorical and numerical features
categorical_features = [f for f in ['famsup', 'higher', 'internet', 'romantic']if f not in immutable_features]
numerical_features = [f for f in ['age','Medu','Fedu','studytime','freetime','goout','health','absences','G1','G2']if f not in immutable_features]

# List to keep track of counterfactual candidates and their substituted features
counterfactual_candidates = []
substituted_features_list = []
prediction_values = []  # Store prediction values for each counterfactual

# Feature order for arrays
feature_order = ['age', 'Medu', 'Fedu', 'studytime', 'famsup', 'higher', 'internet', 'romantic', 'freetime', 'goout', 'health', 'absences', 'G1', 'G2']

# Initialize the substituted instance with the query instance values
substituted_instance = query_instance.copy()

# Track runtime
start_time = time.time()

# Iterate through the important features and substitute (skipping immutable features)
for i in range(len(important_features)):
    substituted_features = []
    print(f"\nSubstituting {i + 1} features:")
    
    for j in range(i + 1):
        feature = important_features[j]
        if feature not in immutable_features:  # Skip immutable features
            substituted_instance[feature] = nun_instance[feature]
            substituted_features.append(feature)
            print(f"Substituted feature '{feature}' with value {nun_instance[feature]}")

    print("Substituted instance:", substituted_instance)

    substituted_instance_array = instance_to_array(substituted_instance, feature_order)
    prediction = rf_model.predict(substituted_instance_array)[0]
    prediction_proba = rf_model.predict_proba(substituted_instance_array)[0][1]

    if prediction == 1:
        counterfactual_candidates.append(substituted_instance.copy())
        substituted_features_list.append(substituted_features.copy())
        prediction_values.append(prediction_proba)
        print(f"Substitution results in label flip. Candidate added.")

    perturbed_instance = substituted_instance.copy()
    for feature in substituted_features:
        if feature in numerical_features and feature not in immutable_features:
            original_value = query_instance[feature]
            substituted_value = nun_instance[feature]
            for perturbation_round in range(10):
                perturbed_value = perturb_numerical_feature(original_value, substituted_value)
                perturbed_instance[feature] = perturbed_value
                print(f"Perturbed '{feature}' from {substituted_value} to {perturbed_value}")

                perturbed_instance_array = instance_to_array(perturbed_instance, feature_order)
                print(f"Perturbed instance for checking: {perturbed_instance}")

                prediction = rf_model.predict(perturbed_instance_array)[0]
                prediction_proba = rf_model.predict_proba(perturbed_instance_array)[0][1]

                if prediction == 1:
                    counterfactual_candidates.append(perturbed_instance.copy())
                    substituted_features_list.append(substituted_features.copy())
                    prediction_values.append(prediction_proba)
                    print(f"Perturbation results in label flip. Candidate added.")

                substituted_value = perturbed_value

# Remove duplicate counterfactual candidates
unique_counterfactual_candidates = []
unique_substituted_features_list = []
unique_prediction_values = []
seen_candidates = set()

for candidate, features, pred_value in zip(counterfactual_candidates, substituted_features_list, prediction_values):
    candidate_frozenset = frozenset(candidate.items())
    if candidate_frozenset not in seen_candidates:
        seen_candidates.add(candidate_frozenset)
        unique_counterfactual_candidates.append(candidate)
        unique_substituted_features_list.append(features)
        unique_prediction_values.append(pred_value)

# Define the print_instance function
def print_instance(instance, excluded_features=None):
    if excluded_features is None:
        excluded_features = []
    for feature, value in instance.items():
        if feature not in excluded_features:
            print(f"{feature}: {value}")

# HEOM distance function
def heom_distance(instance1, instance2, categorical_features, numerical_features, feature_ranges):
    distance = 0
    for feature in categorical_features + numerical_features:
        if feature in categorical_features:
            distance += 1 if instance1[feature] != instance2[feature] else 0
        else:
            range_val = feature_ranges[feature] if feature_ranges[feature] != 0 else 1
            distance += abs(instance1[feature] - instance2[feature]) / range_val
    return distance

# Calculate feature ranges for numerical features
feature_ranges = {feature: data[feature].max() - data[feature].min() for feature in numerical_features}

# Calculate proximity
def calculate_proximity(query_instance, cf_instance, feature_order):
    query_array = instance_to_array(query_instance, feature_order)
    cf_array = instance_to_array(cf_instance, feature_order)
    return np.mean(np.abs(query_array - cf_array))

# Calculate sparsity (as percentage of unchanged features)
def calculate_sparsity(query_instance, cf_instance):
    query_array = np.array([query_instance[feature] for feature in query_instance.keys() if feature != 'target'])
    cf_array = np.array([cf_instance[feature] for feature in cf_instance.keys() if feature != 'target'])
    changed_features = np.sum(query_array != cf_array)
    total_features = len(query_array)
    sparsity = changed_features / total_features  # decimal between 0 and 1
    return sparsity

# Find the best candidate based on proximity and sparsity
def find_best_candidate(counterfactual_candidates, query_instance, feature_order, lambda_value=0.80):
    min_objective_value = float('inf')
    best_candidate = None
    best_prediction_value = None

    for candidate, prediction_value in zip(counterfactual_candidates, unique_prediction_values):
        proximity = calculate_proximity(query_instance, candidate, feature_order)
        sparsity = calculate_sparsity(query_instance, candidate)

        objective_value = lambda_value * proximity + (1 - lambda_value) * sparsity

        if objective_value < min_objective_value:
            min_objective_value = objective_value
            best_candidate = candidate
            best_prediction_value = prediction_value

    return best_candidate, best_prediction_value, min_objective_value

# Set the trade-off parameter lambda
lambda_value = 0.80

# Find the best counterfactual candidate based on proximity and sparsity
best_candidate, best_prediction_value, best_objective_value = find_best_candidate(
    unique_counterfactual_candidates, query_instance, feature_order, lambda_value
)

# Calculate coverage (excluding immutable features)
covered_features = set()
for features in substituted_features_list:
    for feature in features:
        if feature not in immutable_features:
            covered_features.update([feature])

total_features = len([f for f in feature_order if f not in immutable_features])
coverage = len(covered_features) / total_features * 100

# Display the unique counterfactual candidates
print("\nUnique Counterfactual Candidates:")
excluded_features = ['target']
#for i, (candidate, features, pred_value) in enumerate(zip(unique_counterfactual_candidates, unique_substituted_features_list, unique_prediction_values), 1):
#    print(f"\nCandidate {i}:")
#    print_instance(candidate, excluded_features)
#    print(f"Substituted {len(features)} features: {features}")
#    print(f"Prediction probability: {pred_value:.2f}")

# Display the best counterfactual candidate
if best_candidate is not None:
    print("\nBest Counterfactual Candidate based on Proximity and Sparsity:")
    print_instance(best_candidate, excluded_features)
    print(f"Prediction probability: {best_prediction_value:.2f}")
    print(f"Objective function value: {best_objective_value:.2f}")
else:
    print("\nNo valid counterfactual candidates were found.")

# Redefine the calculate_proximity function to use HEOM
def calculate_proximity(query_instance, cf_instance, categorical_features, numerical_features, feature_ranges):
    return heom_distance(query_instance, cf_instance, categorical_features, numerical_features, feature_ranges)

# Display the results
print(f"\nMetrics for Best Counterfactual Candidate:")
if best_candidate is not None:
    best_proximity = calculate_proximity(query_instance, best_candidate, categorical_features, numerical_features, feature_ranges)
    best_sparsity = calculate_sparsity(query_instance, best_candidate)
    
    print("\nBest Proximity: {:.3f}".format(best_proximity))
    print("Best Sparsity: {:.2f}".format(best_sparsity))
else:
    print("\nNo valid counterfactual candidates were found.")
    
#print(f"Coverage: {coverage:.2f}%")

# Calculate feature ranges for numerical features for HEOM
feature_ranges = {feature: data[feature].max() - data[feature].min() for feature in numerical_features}

# Lists to store sparsity and proximity for each candidate
sparsity_list = []
proximity_list = []

# Loop through all unique counterfactual candidates
for candidate in unique_counterfactual_candidates:
    sparsity_value = calculate_sparsity(query_instance, candidate)
    proximity_value = calculate_proximity(query_instance, candidate, categorical_features, numerical_features, feature_ranges)
    
    sparsity_list.append(sparsity_value)
    proximity_list.append(proximity_value)

# Calculate average sparsity and proximity
average_sparsity = np.mean(sparsity_list)
average_proximity = np.mean(proximity_list)

# Display the average metrics
print(f"\nAverage Sparsity: {average_sparsity:.2f}")
print(f"Average Proximity (HEOM): {average_proximity:.2f}")

# Calculate Diversity and Harmonic Mean
def instance_to_filtered_array(instance, feature_order, immutable_features, query_instance, nun_instance, categorical_features):
    return np.array([
        instance[feature] for feature in feature_order
        if feature not in immutable_features and
           not (feature in categorical_features and query_instance[feature] == nun_instance[feature])
    ])


def calculate_diversity(counterfactuals, feature_order, immutable_features, query_instance, nun_instance, categorical_features):
    counterfactual_arrays = [
        instance_to_filtered_array(cf, feature_order, immutable_features, query_instance, nun_instance, categorical_features)
        for cf in counterfactuals
    ]

    pairs = list(combinations(counterfactual_arrays, 2))
    distances = [hamming(pair[0], pair[1]) for pair in pairs]

    average_distance = np.mean(distances)
    diversity_percent = average_distance * 100  
    
    return diversity_percent

diversity = calculate_diversity(unique_counterfactual_candidates, feature_order, immutable_features, query_instance, nun_instance, categorical_features)

# Convert to unchanged percentage
average_sparsity2 = (1 - average_sparsity) * 100

def calculate_harmonic_mean(diversity, sparsity):
    if diversity + sparsity == 0:
        return 0
    return 2 * (diversity * sparsity) / (diversity + sparsity)

harmonic_mean = calculate_harmonic_mean(diversity, average_sparsity2)

print(f"Diversity: {diversity:.2f}%")
print(f"Harmonic Mean: {harmonic_mean:.2f}%")


from sklearn.ensemble import IsolationForest

# ----------------------------
# Train Isolation Forest model
# ----------------------------
isf = IsolationForest(contamination=0.05, random_state=42)
isf.fit(X_train)  # Use standardized training features

# ----------------------------
# Function to check plausibility
# ----------------------------
def check_plausibility(instance, scaler, model, feature_order):
    instance_array = instance_to_array(instance, feature_order)
    scaled_array = scaler.transform(instance_array)
    return model.predict(scaled_array)[0] == 1  # 1 means plausible, -1 means anomaly

# ----------------------------
# Evaluate best candidate
# ----------------------------
cf_array = instance_to_array(best_candidate, feature_order)
cf_array_scaled = scaler.transform(cf_array)
plausibility_score = isf.predict(cf_array_scaled)[0]
is_plausible = (plausibility_score == 1)

# ----------------------------
# Evaluate plausibility coverage across all counterfactuals
# ----------------------------
plausibility_flags = [
    check_plausibility(cf, scaler, isf, feature_order)
    for cf in unique_counterfactual_candidates
]
plausibility_coverage = 100 * np.mean(plausibility_flags)

print(f"Plausibility Coverage across all counterfactuals: {plausibility_coverage:.2f}%")

# ----------------------------
# Runtime reporting
# ----------------------------
end_time = time.time()
runtime_seconds = end_time - start_time

if best_candidate is not None:
    print(f"Total Runtime: {runtime_seconds:.2f} seconds")
    print("===================")

