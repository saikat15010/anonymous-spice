import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
import joblib
from itertools import combinations
from scipy.spatial.distance import hamming
from scipy import stats

import time
# Track runtime
start_time = time.time()

# Load dataset
data = pd.read_csv('/Users/saikat/Desktop/Summer 2025/CFNet/New Experiment/HELOC Dataset/heloc_dataset_v1.csv')

# Preprocess the target variable
data['target'] = data['target'].map({'Bad': 0, 'Good': 1})

# Define features (X) and target (y)
X = data.drop(columns=['target']).values
y = data['target'].values

# Split data into training, validation, and test sets
# 75% training, 10% validation, 15% testing
X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=0.15, random_state=0)
X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=0.10, random_state=0)

# Standardize features - This is important for Neural Networks
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

# Initialize the Multilayer Perceptron (MLP) Classifier
rf_model = MLPClassifier(
    hidden_layer_sizes=(100, 50), # Two hidden layers with 100 and 50 neurons
    activation='relu',            # ReLU activation function
    solver='adam',                # Adam optimizer
    max_iter=500,                 # Maximum number of iterations
    random_state=0,               # For reproducibility
    early_stopping=True,          # Enable early stopping
    validation_fraction=0.1,      # Proportion of training data to set aside for validation
    n_iter_no_change=10,          # Number of iterations with no improvement to wait before stopping
    tol=1e-4                      # Tolerance for the optimization
)

print("Training Multilayer Perceptron (MLP) model...")

# Train the model
# Note: MLPClassifier uses its own internal validation set for early stopping,
# so we don't need to pass X_val, y_val to the fit method.
rf_model.fit(X_train, y_train)

# Save the trained model
model_path = 'best_heloc_mlp_model.pkl'
joblib.dump(rf_model, model_path)
print(f"Model saved to: {model_path}")

# Load the model (simulating loading for a separate prediction task)
loaded_model = joblib.load(model_path)

# Make predictions on the test set
y_pred_test = loaded_model.predict(X_test)
y_pred_proba_test = loaded_model.predict_proba(X_test)[:, 1]  # Probabilities for the positive class

# Evaluate model performance
accuracy_test = accuracy_score(y_test, y_pred_test)
conf_matrix_test = confusion_matrix(y_test, y_pred_test)
class_report_test = classification_report(y_test, y_pred_test)

print(f'\nAccuracy on Test Set: {accuracy_test:.4f}')
print('\nConfusion Matrix on Test Set:')
print(conf_matrix_test)
print('\nClassification Report on Test Set:')
print(class_report_test)

# Get and print feature names as a Python list
feature_names = data.drop(columns=['target']).columns.tolist()
print("\nFeature List Used in the Model:")
print(feature_names)


# Query and NUN instance from the previous results
query_instance = {'ExternalRiskEstimate': 67, 'MSinceOldestTradeOpen': 66, 'MSinceMostRecentTradeOpen': 5, 'AverageMInFile': 24, 'NumSatisfactoryTrades': 9, 'NumTrades60Ever2DerogPubRec': 0, 'NumTrades90Ever2DerogPubRec': 0, 'PercentTradesNeverDelq': 100, 'MSinceMostRecentDelq': -7, 'MaxDelq2PublicRecLast12M': 7, 'MaxDelqEver': 8, 'NumTotalTrades': 9, 'NumTradesOpeninLast12M': 4, 'PercentInstallTrades': 44, 'MSinceMostRecentInqexcl7days': 0, 'NumInqLast6M': 4, 'NumInqLast6Mexcl7days': 4, 'NetFractionRevolvingBurden': 53, 'NetFractionInstallBurden': 66, 'NumRevolvingTradesWBalance': 4, 'NumInstallTradesWBalance': 2, 'NumBank2NatlTradesWHighUtilization': 1, 'PercentTradesWBalance': 86}

nun_instance   = {'ExternalRiskEstimate': 62, 'MSinceOldestTradeOpen': 46, 'MSinceMostRecentTradeOpen': 1, 'AverageMInFile': 24, 'NumSatisfactoryTrades': 8, 'NumTrades60Ever2DerogPubRec': 0, 'NumTrades90Ever2DerogPubRec': 0, 'PercentTradesNeverDelq': 100, 'MSinceMostRecentDelq': -7, 'MaxDelq2PublicRecLast12M': 7, 'MaxDelqEver': 8, 'NumTotalTrades': 10, 'NumTradesOpeninLast12M': 4, 'PercentInstallTrades': 40, 'MSinceMostRecentInqexcl7days': 0, 'NumInqLast6M': 4, 'NumInqLast6Mexcl7days': 4, 'NetFractionRevolvingBurden': 46, 'NetFractionInstallBurden': 9, 'NumRevolvingTradesWBalance': 5, 'NumInstallTradesWBalance': 3, 'NumBank2NatlTradesWHighUtilization': 4, 'PercentTradesWBalance': 89}


# Immutable features
immutable_features = []

# Helper function to convert instance dictionary to array
def instance_to_array(instance, feature_order):
    return np.array([instance[feature] for feature in feature_order]).reshape(1, -1)

# Function to perturb numerical feature
def perturb_numerical_feature(original_value, substituted_value, perturbation_factor=0.1):
    return substituted_value + perturbation_factor * (original_value - substituted_value)

# Features ordered by importance
important_features = [f for f in ['ExternalRiskEstimate', 'NetFractionRevolvingBurden', 'AverageMInFile', 'NumInqLast6Mexcl7days', 'PercentInstallTrades', 'NumSatisfactoryTrades', 'NumTrades60Ever2DerogPubRec', 'MaxDelq2PublicRecLast12M', 'NumRevolvingTradesWBalance', 'NumTrades90Ever2DerogPubRec', 'PercentTradesNeverDelq', 'MaxDelqEver', 'NumTradesOpeninLast12M', 'MSinceMostRecentInqexcl7days', 'NumBank2NatlTradesWHighUtilization', 'MSinceMostRecentTradeOpen', 'MSinceOldestTradeOpen', 'NumInstallTradesWBalance', 'NumInqLast6M', 'MSinceMostRecentDelq', 'NetFractionInstallBurden', 'NumTotalTrades', 'PercentTradesWBalance'] if f not in immutable_features]


# Define categorical and numerical features
categorical_features = [f for f in ['NumTrades60Ever2DerogPubRec','NumTrades90Ever2DerogPubRec'] if f not in immutable_features]
numerical_features = [f for f in ['NetFractionRevolvingBurden', 'ExternalRiskEstimate', 'AverageMInFile', 'NumSatisfactoryTrades', 'MaxDelq2PublicRecLast12M', 'PercentTradesNeverDelq', 'PercentInstallTrades', 'NumRevolvingTradesWBalance', 'MaxDelqEver', 'NumInqLast6Mexcl7days', 'MSinceMostRecentInqexcl7days', 'NumTradesOpeninLast12M', 'MSinceOldestTradeOpen', 'NumInqLast6M', 'NumInstallTradesWBalance', 'MSinceMostRecentDelq', 'MSinceMostRecentTradeOpen', 'NumBank2NatlTradesWHighUtilization', 'PercentTradesWBalance', 'NumTotalTrades', 'NetFractionInstallBurden'] if f not in immutable_features]

# List to keep track of counterfactual candidates and their substituted features
counterfactual_candidates = []
substituted_features_list = []
prediction_values = []  # Store prediction values for each counterfactual

# Feature order for arrays
feature_order = ['ExternalRiskEstimate', 'MSinceOldestTradeOpen', 'MSinceMostRecentTradeOpen', 'AverageMInFile', 'NumSatisfactoryTrades', 'NumTrades60Ever2DerogPubRec', 'NumTrades90Ever2DerogPubRec', 'PercentTradesNeverDelq', 'MSinceMostRecentDelq', 'MaxDelq2PublicRecLast12M', 'MaxDelqEver', 'NumTotalTrades', 'NumTradesOpeninLast12M', 'PercentInstallTrades', 'MSinceMostRecentInqexcl7days', 'NumInqLast6M', 'NumInqLast6Mexcl7days', 'NetFractionRevolvingBurden', 'NetFractionInstallBurden', 'NumRevolvingTradesWBalance', 'NumInstallTradesWBalance', 'NumBank2NatlTradesWHighUtilization', 'PercentTradesWBalance']

# Initialize the substituted instance with the query instance values
substituted_instance = query_instance.copy()



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
    unchanged_features = np.sum(query_array == cf_array)
    total_features = len(query_array)
    sparsity = (unchanged_features / total_features) * 100
    return sparsity

# Find the best candidate based on proximity and sparsity
def find_best_candidate(counterfactual_candidates, query_instance, feature_order, lambda_value=0.5):
    min_objective_value = float('inf')
    best_candidate = None
    best_prediction_value = None

    for candidate, prediction_value in zip(counterfactual_candidates, unique_prediction_values):
        proximity = calculate_proximity(query_instance, candidate, feature_order)
        sparsity = calculate_sparsity(query_instance, candidate)

        # Invert sparsity to fit into a minimization framework
        if sparsity > 0:
            inverted_sparsity = 1 / sparsity
        else:
            inverted_sparsity = float('inf')

        objective_value = lambda_value * proximity + (1 - lambda_value) * inverted_sparsity

        if objective_value < min_objective_value:
            min_objective_value = objective_value
            best_candidate = candidate
            best_prediction_value = prediction_value

    return best_candidate, best_prediction_value, min_objective_value

# Set the trade-off parameter lambda
lambda_value = 0.5

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
for i, (candidate, features, pred_value) in enumerate(zip(unique_counterfactual_candidates, unique_substituted_features_list, unique_prediction_values), 1):
    print(f"\nCandidate {i}:")
    print_instance(candidate, excluded_features)
    print(f"Substituted {len(features)} features: {features}")
    print(f"Prediction probability: {pred_value:.2f}")

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
    
    print("\nBest Proximity using HEOM distance metric: {:.3f}".format(best_proximity))
    print("Best Sparsity: {:.2f}%".format(best_sparsity))
else:
    print("\nNo valid counterfactual candidates were found.")

# ----------------------------
# Runtime reporting
# ----------------------------
end_time = time.time()
runtime_seconds = end_time - start_time
print(f"Total Runtime: {runtime_seconds:.2f} seconds")

print(f"Coverage: {coverage:.2f}%")

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
print(f"\nAverage Sparsity: {average_sparsity:.2f}%")
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

def calculate_harmonic_mean(diversity, sparsity):
    if diversity + sparsity == 0:
        return 0
    return 2 * (diversity * sparsity) / (diversity + sparsity)

harmonic_mean = calculate_harmonic_mean(diversity, average_sparsity)

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




