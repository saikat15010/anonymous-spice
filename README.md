# SPICE: Sparse and ProxImate Counterfactual Explanations via Feature-Importance-Weighted Perturbation

> Anonymous submission

---

## Overview

SPICE is a model-agnostic framework for generating sparse, proximate, and plausible counterfactual explanations for tabular classification models. SPICE uses a feature-importance-driven approximate nearest unlike neighbor (NUN) search to identify a meaningful starting point for counterfactual generation, and then applies importance-guided substitution and perturbation to produce multiple high-quality counterfactual candidates.

## Key Properties

- **Sparsity**: changes a small number of features.
- **Proximity**: keeps the counterfactual close to the original query instance.
- **Plausibility**: encourages generated candidates to lie within the data distribution.
- **Diversity**: produces multiple distinct counterfactual candidates.
- **Coverage**: achieves complete coverage in the reported experiments through the nearest unlike neighbor fallback.
- **Model-agnosticism**: works with black-box classifiers using only input-output access.

---

## Repository Structure

```text
anonymous-spice/
├── preprocess_dataset/
│   ├── adult_income/
│   ├── gmc/
│   ├── graduate_admission/
│   ├── heloc/
│   └── student_performance/
│
├── src_code/
│   ├── baseline/                     # Baseline implementations
│   ├── spice_code/                   # Main SPICE implementation
│   ├── spice_code_individual_instance/
│   │                                  # Individual-query SPICE scripts with feature-importance methods
│   ├── spice_rho_experiment/         # ANN approximation factor experiment
│   ├── spice_tradeoff_experiment/    # Lambda trade-off sensitivity experiment
│   ├── spice_validity_experiment/    # Validity and NUN improvement experiment
│   └── spice_weighted_unweighted_experiment/
│                                      # Weighted vs. unweighted NUN ablation
│
└── README.md
```

---

## Datasets

SPICE is evaluated on five tabular binary classification datasets.

| Dataset | Size | Features | Source |
|---|---:|---:|---|
| Graduate Admission | 500 | 7 | Kaggle |
| Adult Income | 45,022 | 14 | UCI Machine Learning Repository |
| Student Performance | 649 | 14 | UCI Machine Learning Repository |
| HELOC | 10,459 | 23 | FICO Explainable Machine Learning Challenge |
| GMC | 150,000 | 10 | Give Me Some Credit / CARLA |

---

## Requirements

```text
Python >= 3.12
numpy >= 1.26.4
pandas >= 1.5.3
scikit-learn >= 1.3.2
scipy >= 1.11.0
shap >= 0.44.0
joblib >= 1.3.0
```

Install dependencies with:

```bash
pip install -r requirements.txt
```

If `requirements.txt` is not available, install the listed packages manually according to your local environment.

---

## Running SPICE

The main SPICE implementation is available under:

```text
src_code/spice_code/
```

To run the main SPICE implementation, use the corresponding script in this directory.

Example:

```bash
cd src_code/spice_code
python <main_script>.py
```

Replace `<main_script>` with the script for the target dataset or experiment.

The generated results are saved as a `.csv` file in the corresponding directory.

The directory below contains individual-query SPICE scripts with feature-importance methods:

```text
src_code/spice_code_individual_instance/
```

These scripts are used for per-dataset or individual-instance analysis where feature-importance scores are computed and used to guide substitution and perturbation.

---

## Running Baselines

Baseline scripts are provided under:

```text
src_code/baseline/
```

The repository includes implementations for the six comparison methods used in the experiments:

- DiCE
- NICE
- PertCF
- UFCE
- FACE
- VanillaCF

Each baseline follows the same evaluation protocol as SPICE:

- 100 randomly sampled test queries
- Random seed set to 42
- Same HEOM-based proximity metric
- Same sparsity ratio
- Same diversity metric
- Same plausibility evaluation using IsolationForest with contamination set to 0.10

Example:

```bash
cd src_code/baseline/NICE
python NICE_GraduateAdmission.py
```

---

## Reproducing Experiments

| Experiment | Script Location |
|---|---|
| Main SPICE implementation | `src_code/spice_code/` |
| Individual-instance SPICE analysis with feature importance | `src_code/spice_code_individual_instance/` |
| Baseline comparison | `src_code/baseline/` |
| Weighted vs. unweighted NUN ablation | `src_code/spice_weighted_unweighted_experiment/` |
| Feature-importance sensitivity analysis | `src_code/spice_code_individual_instance/` |
| Lambda trade-off analysis | `src_code/spice_tradeoff_experiment/` |
| Validity and NUN improvement analysis | `src_code/spice_validity_experiment/` |
| ANN approximation factor analysis | `src_code/spice_rho_experiment/` |

---

## Feature-Importance Methods

SPICE supports four feature-importance methods for NUN search and perturbation ordering:

- **SHAP**: SHapley Additive exPlanations
- **LIME**: Local Interpretable Model-agnostic Explanations
- **MI**: Mutual Information
- **ANOVA**: Analysis of Variance

The individual-instance scripts include the feature-importance computation used to rank mutable features before substitution and perturbation.

---

## Evaluation Metrics

| Metric | Description |
|---|---|
| Best Sparsity | Minimum fraction of changed features among generated candidates |
| Best Proximity | Minimum HEOM distance to the query among generated candidates |
| Average Sparsity | Mean fraction of changed features across generated candidates |
| Average Proximity | Mean HEOM distance across generated candidates |
| Diversity | Mean pairwise Hamming distance across generated candidates |
| Plausibility | Fraction of candidates classified as inliers by IsolationForest |
| Coverage | Fraction of queries that receive at least one valid counterfactual |

---

## Notes for Reviewers

This repository is prepared for anonymous peer review. Author-identifying information has been removed.

The code is provided to support reproducibility of the experimental results reported in the paper. Depending on the local machine and directory structure, dataset paths and environment settings may need minor adjustment before running the scripts.

---

## License

This repository is provided for anonymous peer review. License information will be updated after the review process.
