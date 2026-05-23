import numpy as np
import pandas as pd
import dice_ml
import joblib
import warnings
from itertools import combinations
from scipy.spatial.distance import hamming
from sklearn.ensemble import IsolationForest, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Suppress sklearn warnings about feature names
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

# --- Configuration & Paths ---
DATA_PATH = '/Users/saikat/Desktop/Summer 2025/CFNet_AAAI_26/Adult Income Dataset/AISTATS Adult Income /processed_adult.csv'
MODEL_PATH = '/Users/saikat/Desktop/Summer 2025/CFNet_AAAI_26/Adult Income Dataset/AISTATS Adult Income /best_adult_income_gb_model.pkl'

FEATURE_COLUMNS = ['age', 'workclass', 'fnlwgt', 'education', 'educational-num', 
                   'marital-status', 'occupation', 'relationship', 'race', 'gender', 
                   'capital-gain', 'capital-loss', 'hours-per-week', 'native-country']
CATEGORICAL_FEATS = ['workclass', 'education', 'marital-status', 'occupation', 
                     'relationship', 'race', 'gender', 'native-country']
NUMERICAL_FEATS   = ['age', 'fnlwgt', 'educational-num', 'capital-gain', 
                     'capital-loss', 'hours-per-week']
IMMUTABLE_FEATS   = ['age', 'gender', 'workclass', 'race', 'fnlwgt','education', 'native-country']
TARGET_COL        = 'target'

NUM_QUERIES = 100
RANDOM_SEED = 42

# --- Helper Functions ---
def calculate_heom(q, c, cat_idx, num_idx, ranges):
    dist = 0.0
    for i in cat_idx:
        dist += 1.0 if q[i] != c[i] else 0.0
    for i in num_idx:
        rng = ranges[i] if ranges[i] != 0 else 1
        dist += abs(q[i] - c[i]) / rng
    return dist

def calculate_diversity(cf_list):
    if len(cf_list) < 2: return 0.0
    pairs = list(combinations(cf_list, 2))
    dists = [hamming(p[0], p[1]) for p in pairs]
    return np.mean(dists) * 100.0

def main():
    # 1. Load Data & Model
    df = pd.read_csv(DATA_PATH)
    
    X_raw = df[FEATURE_COLUMNS].values
    y_raw = df[TARGET_COL].values
    
    # Split matches SPICE: First 15% for test, remaining 85% for training
    split_idx = int(0.15 * len(X_raw))
    X_test_raw, X_train_raw = X_raw[:split_idx], X_raw[split_idx:]
    y_test_raw, y_train_raw = y_raw[:split_idx], y_raw[split_idx:]
    
    scaler = StandardScaler().fit(X_train_raw)
    loaded_model = joblib.load(MODEL_PATH)

    # Plausibility Model (matches SPICE protocol: trained on X_train, contamination=0.10)
    isf = IsolationForest(contamination=0.10, random_state=RANDOM_SEED)
    isf.fit(scaler.transform(X_train_raw))
    
    # Range precomputation for HEOM
    ranges = np.array([df[f].max() - df[f].min() if f in NUMERICAL_FEATS else 0 for f in FEATURE_COLUMNS])
    cat_idx = [FEATURE_COLUMNS.index(f) for f in CATEGORICAL_FEATS]
    num_idx = [FEATURE_COLUMNS.index(f) for f in NUMERICAL_FEATS]

    # 2. Setup DiCE with Wrapper to fix "Feature Names" Warning
    class ModelWrapper:
        def __init__(self, model, scaler):
            self.model = model
            self.scaler = scaler
        def predict_proba(self, data):
            scaled_data = self.scaler.transform(data.values)
            return self.model.predict_proba(scaled_data)
        def predict(self, data):
            scaled_data = self.scaler.transform(data.values)
            return self.model.predict(scaled_data)

    wrapped_model = ModelWrapper(loaded_model, scaler)
    
    d_data = dice_ml.Data(dataframe=df, continuous_features=NUMERICAL_FEATS, outcome_name=TARGET_COL)
    d_model = dice_ml.Model(model=wrapped_model, backend="sklearn")
    exp = dice_ml.Dice(d_data, d_model, method="genetic") 

    # 3. Select 100 random queries where model predicts 0 from the test set
    test_X_scaled = scaler.transform(X_test_raw)
    preds = loaded_model.predict(test_X_scaled)
    query_pool_indices = np.where(preds == 0)[0]
    
    rng = np.random.default_rng(RANDOM_SEED)
    selected_indices = rng.choice(query_pool_indices, size=min(NUM_QUERIES, len(query_pool_indices)), replace=False)

    results = []
    print(f"Generating DiCE counterfactuals for {len(selected_indices)} queries...")

    for i, idx in enumerate(selected_indices):
        # Create a single-row DataFrame for the query
        query_instance = pd.DataFrame([X_test_raw[idx]], columns=FEATURE_COLUMNS)
        q_array = X_test_raw[idx]
        
        try:
            dice_exp = exp.generate_counterfactuals(
                query_instance, 
                total_CFs=10, 
                desired_class="opposite",
                features_to_vary=[f for f in FEATURE_COLUMNS if f not in IMMUTABLE_FEATS]
            )
            
            cf_df = dice_exp.cf_examples_list[0].final_cfs_df
            if cf_df is None or cf_df.empty: continue
            if TARGET_COL in cf_df.columns: cf_df = cf_df.drop(columns=[TARGET_COL])
                
            cf_values = cf_df[FEATURE_COLUMNS].values
            
            # Metric Calculation
            proxs = [calculate_heom(q_array, cf, cat_idx, num_idx, ranges) for cf in cf_values]
            spars = [np.sum(q_array != cf)/len(q_array) for cf in cf_values]
            plaus_preds = isf.predict(scaler.transform(cf_values))
            
            results.append({
                'best_sparsity': min(spars),
                'best_proximity': min(proxs),
                'avg_sparsity': np.mean(spars),
                'avg_proximity': np.mean(proxs),
                'diversity': calculate_diversity(cf_values),
                'plausibility': np.mean(plaus_preds == 1) * 100.0
            })
        except Exception:
            continue

    # 4. Final Output
    res_df = pd.DataFrame(results)
    print("\n" + "="*65)
    print("DICE (GENETIC) BASELINE - ADULT INCOME (100 QUERIES)")
    print("="*65)
    metrics = ['best_sparsity', 'best_proximity', 'avg_sparsity', 'avg_proximity', 'diversity', 'plausibility']
    labels = ['Best Spar.', 'Best Prox.', 'Avg Spar.', 'Avg Prox.', 'Div. (%)', 'Plaus. (%)']
    
    for m, l in zip(metrics, labels):
        print(f"{l:<15} {res_df[m].mean():>12.4f} +/- {res_df[m].std():.4f}")
    
    res_df.to_csv('table5_dice_AdultIncome.csv', index=False)

if __name__ == "__main__":
    main()