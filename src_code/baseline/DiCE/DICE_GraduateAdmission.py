import numpy as np
import pandas as pd
import dice_ml
import joblib
import warnings
from itertools import combinations
from scipy.spatial.distance import hamming
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.preprocessing import StandardScaler

# Suppress sklearn warnings about feature names
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

# --- Configuration & Paths ---
TRAIN_PATH = '/Users/saikat/Desktop/Summer 2025/CFNet_AAAI_26/Graduate Admission Dataset/admission_train.csv'
TEST_PATH  = '/Users/saikat/Desktop/Summer 2025/CFNet_AAAI_26/Graduate Admission Dataset/admission_test.csv'
MODEL_PATH = '/Users/saikat/Desktop/Summer 2025/CFNet_AAAI_26/Graduate Admission Dataset/AISTATS Graduate Admission/best_admission_rf_model.pkl'

FEATURE_COLUMNS = ['GRE Score', 'TOEFL Score', 'University Rating', 'SOP', 'LOR', 'CGPA', 'Research']
CATEGORICAL_FEATS = ['University Rating', 'Research']
NUMERICAL_FEATS   = ['GRE Score', 'TOEFL Score', 'SOP', 'LOR', 'CGPA']
IMMUTABLE_FEATS   = [] # Matches SPICE settings for this dataset
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
    df_train = pd.read_csv(TRAIN_PATH)
    df_test = pd.read_csv(TEST_PATH)
    df = pd.concat([df_train, df_test], ignore_index=True)
    
    X_train_raw = df_train[FEATURE_COLUMNS].values
    scaler = StandardScaler().fit(X_train_raw)
    loaded_model = joblib.load(MODEL_PATH)

    # Plausibility Model (matches SPICE protocol)
    isf = IsolationForest(contamination=0.10, random_state=RANDOM_SEED)
    isf.fit(scaler.transform(X_train_raw))
    
    # Range precomputation for HEOM
    ranges = np.array([df[f].max() - df[f].min() if f in NUMERICAL_FEATS else 0 for f in FEATURE_COLUMNS])
    cat_idx = [FEATURE_COLUMNS.index(f) for f in CATEGORICAL_FEATS]
    num_idx = [FEATURE_COLUMNS.index(f) for f in NUMERICAL_FEATS]

    # 2. Setup DiCE with Wrapper to strip feature names
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

    # 3. Select 100 random queries where model predicts 0
    test_X_scaled = scaler.transform(df_test[FEATURE_COLUMNS].values)
    preds = loaded_model.predict(test_X_scaled)
    query_pool_indices = np.where(preds == 0)[0]
    
    rng = np.random.default_rng(RANDOM_SEED)
    selected_indices = rng.choice(query_pool_indices, size=min(NUM_QUERIES, len(query_pool_indices)), replace=False)

    results = []
    print(f"Generating DiCE counterfactuals for {len(selected_indices)} queries...")

    for i, idx in enumerate(selected_indices):
        query_instance = df_test.iloc[[idx]].drop(columns=[TARGET_COL])
        q_array = query_instance.values[0]
        
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
            
            # Calculate Metrics
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
    print("DICE (GENETIC) BASELINE - GRADUATE ADMISSION (100 QUERIES)")
    print("="*65)
    metrics = ['best_sparsity', 'best_proximity', 'avg_sparsity', 'avg_proximity', 'diversity', 'plausibility']
    labels = ['Best Spar.', 'Best Prox.', 'Avg Spar.', 'Avg Prox.', 'Div. (%)', 'Plaus. (%)']
    
    for m, l in zip(metrics, labels):
        print(f"{l:<15} {res_df[m].mean():>12.4f} +/- {res_df[m].std():.4f}")
    
    res_df.to_csv('table5_dice_GraduateAdmission.csv', index=False)

if __name__ == "__main__":
    main()