# Loan Default Prediction — Ensemble Methods & Hyperparameter Analysis

A portfolio project benchmarking ensemble methods (Random Forest, AdaBoost, Gradient Boosting) on real-world credit risk data, answering three core ML engineering questions with real experimental evidence.

---

## Dataset

**Give Me Some Credit** (Kaggle, 2011)  
150,000 borrowers · 10 features · Binary target: `SeriousDlqin2yrs` (90+ day delinquency within 2 years)  
Class imbalance: 93.3% negative / 6.7% positive  
Download: https://www.kaggle.com/c/GiveMeSomeCredit/data (`cs-training.csv`)

---

## Project Structure

```
loan-default-ensembles/
├── data/
│   ├── cs-training.csv              ← raw data (download from Kaggle)
│   ├── cleaned_loan_data.csv        ← output of 01_eda_cleaning.ipynb
│   ├── noise_robustness.png         ← output of 03_noise_robustness.ipynb
│   └── diminishing_returns.png      ← output of 03_noise_robustness.ipynb
├── notebooks/
│   ├── 01_eda_cleaning.ipynb        ← EDA, data cleaning, feature engineering
│   ├── 02_training.ipynb            ← baseline models + hyperparameter sensitivity
│   └── 03_noise_robustness.ipynb    ← noise experiment + diminishing returns curve
└── README.md
```

---

## Reproducing Results

```bash
# 1. Clone the repo
git clone https://github.com/yourusername/loan-default-ensembles
cd loan-default-ensembles

# 2. Install dependencies
pip install pandas numpy scikit-learn matplotlib mlflow

# 3. Download cs-training.csv from Kaggle and place in data/

# 4. Run notebooks in order
# 01_eda_cleaning.ipynb  →  produces cleaned_loan_data.csv
# 02_training.ipynb      →  baseline models + hyperparameter sweeps
# 03_noise_robustness.ipynb → noise experiment + tree curves

# 5. View MLflow experiment runs
mlflow ui  # open http://localhost:5000
```

---

## Data Cleaning

Four categories of problems identified and resolved:

| Problem | Scale | Fix |
|---|---|---|
| `age = 0` (impossible value) | 1 row | Drop row |
| `RevolvingUtilizationOfUnsecuredLines` extreme outliers | Long tail, max 50,708 | Cap at 99th percentile |
| `DebtRatio` extreme outliers | Long tail, max 329,664 | Cap at 99th percentile |
| 96/98 sentinel codes across 3 delinquency columns | 269 rows (same rows, all 3 columns) | Flag (`delinquency_data_error`) + impute with clean median |
| `MonthlyIncome` missing | 19.8% | Flag (`income_was_missing`) + impute + cap outliers |
| `NumberOfDependents` missing | 2.6% | Median impute |

**Key insight:** 96/98 values across all three late-payment columns always appeared on the same 269 rows simultaneously — confirming they were systemic error codes, not real counts.

---

## Feature Engineering

Three domain-driven features built on top of the cleaned data:

| Feature | Formula | Reasoning |
|---|---|---|
| `TotalTimesLate` | Sum of 3 delinquency columns | Aggregate correlated severity levels into one total signal |
| `IncomePerDependent` | `MonthlyIncome / (NumberOfDependents + 1)` | Capacity relative to obligation — a tree cannot compute ratios on its own |
| `TotalCreditLines` | `NumberOfOpenCreditLinesAndLoans + NumberRealEstateLoansOrLines` | Total credit exposure as a single count |

Final dataset: **149,999 rows · 16 columns** (10 original features + 1 target + 2 flags + 3 engineered features)

---

## Baseline Results

All models trained with default hyperparameters, evaluated on held-out validation set (20% stratified split), scored on **ROC-AUC** (accuracy would be misleading at 93/7 class imbalance).

| Model | Baseline AUC |
|---|---|
| Random Forest | 0.8435 |
| AdaBoost | 0.8497 |
| **Gradient Boosting** | **0.8689** |

---

## Q1: Which Hyperparameter Moves Performance Most?

**Method:** vary one hyperparameter at a time, hold all others at default, measure AUC swing (max − min across tested values).

### Random Forest

| Hyperparameter | Values Tested | AUC Swing |
|---|---|---|
| **max_depth** | 3, 5, 10, None | **0.0244** |
| min_samples_leaf | 1, 5, 20, 50 | 0.0234 |
| n_estimators | 50, 100, 200, 400 | 0.0148 |
| max_features | sqrt, 0.3, 0.6, 1.0 | 0.0031 |

**Winner: `max_depth`** — controls how complex each individual tree is allowed to get. Notably, `max_features` (the conceptually "signature" parameter of RF) had almost no effect on this dataset.

Notable finding: unconstrained trees (`max_depth=None`, AUC 0.8435) performed *worse* than trees capped at depth 10 (AUC 0.8679) — deeper trees memorized noise rather than learning generalizable patterns.

### AdaBoost

| Hyperparameter | Values Tested | AUC Swing |
|---|---|---|
| **learning_rate** | 0.01, 0.1, 0.5, 1.0, 2.0 | **0.3553** |
| n_estimators | 50, 100, 200, 400 | 0.0156 |
| base_estimator depth | 1, 2, 3, 5 | 0.0146 |

**Winner: `learning_rate`** by a massive margin — 20x larger swing than any RF hyperparameter.  
- `learning_rate=2.0` → AUC collapsed to **0.5000** (pure random guessing — total failure)  
- `learning_rate=0.01` → AUC 0.6957 (unfinished — not enough rounds to accumulate corrections)

### Gradient Boosting

| Hyperparameter | Values Tested | AUC Swing |
|---|---|---|
| **learning_rate** | 0.01, 0.1, 0.5, 1.0, 2.0 | **0.2751** |
| max_depth | 1, 2, 3, 5, 10 | 0.0107 |
| n_estimators | 50, 100, 200, 400 | 0.0030 |

**Winner: `learning_rate`** — same pattern as AdaBoost, same structural reason (sequential correction amplifies instability).  
- GBM at `learning_rate=2.0` dropped to 0.5937 — damaged but not fully collapsed, because GBM's default `max_depth=3` trees retain some independent signal even when the sequential weighting goes unstable.

### Summary

| Method | Dominant Hyperparameter | Swing | What It Controls |
|---|---|---|---|
| Random Forest | `max_depth` | 0.024 | Individual tree complexity |
| AdaBoost | `learning_rate` | 0.355 | Aggressiveness of sequential correction |
| GradientBoosting | `learning_rate` | 0.275 | Aggressiveness of sequential correction |

**Key insight:** bagging's performance is driven by controlling tree complexity. Boosting's performance is driven by controlling how aggressively each round corrects the last one. Two completely different failure modes for two structurally different algorithms.

---

## Q2: Does Bagging or Boosting Handle Noise Better?

**Method:** inject controlled label noise at increasing levels (0–25%) into the training set only; evaluate all three models on the clean, unmodified validation set at each level.

Label flipping simulates real-world credit data corruption: misreported outcomes, ambiguous charge-offs, data entry errors.

### Results

| Noise Level | RandomForest | AdaBoost | GradientBoosting |
|---|---|---|---|
| 0% | 0.8678 | 0.8611 | 0.8695 |
| 5% | 0.8679 | 0.8538 | 0.8661 |
| 10% | 0.8670 | 0.8522 | 0.8627 |
| 15% | 0.8633 | 0.8497 | 0.8551 |
| 20% | 0.8629 | 0.8493 | 0.8498 |
| 25% | **0.8605** | 0.8360 | 0.8422 |
| **Total Drop** | **0.0073** | **0.0251** | **0.0273** |

### Key Findings

- **Random Forest degraded 3.5x slower than both boosting methods** under 25% label corruption
- **GBM crossed below RF between 15–20% noise** — the exact point where its clean-data advantage disappears
- **AdaBoost dropped immediately at 5% noise** — the most sensitive from the first corruption level
- RF never fell below 0.860 across the entire noise range; AdaBoost reached 0.836 at 25%

**Why bagging wins on noisy data:** RF builds trees independently in parallel — a corrupted label in one tree's bootstrap sample gets outvoted by the other 99 trees that mostly didn't see it. Boosting builds trees sequentially, each one specifically targeting whatever the previous trees got wrong — flipped labels look "hard" and get chased harder each round, compounding the corruption rather than diluting it.

**Production implication:** for real-world credit data where label quality is never guaranteed, Random Forest is the safer production choice — not because it has the highest clean-data AUC, but because its performance degrades far more gracefully under the inevitable data quality issues present in live systems.

---

## Q3: Performance vs. Number of Trees — Diminishing Returns

**Method:** train each model at 10 tree counts (10, 25, 50, 75, 100, 150, 200, 300, 400, 500) using best hyperparameters from Q1; record validation AUC at each point.

### Results

| n_estimators | RandomForest | AdaBoost | GradientBoosting |
|---|---|---|---|
| 10 | 0.8645 | 0.8486 | 0.8601 |
| 25 | 0.8661 | 0.8546 | 0.8663 |
| 50 | 0.8667 | 0.8553 | 0.8687 |
| 75 | 0.8674 | 0.8572 | 0.8693 |
| 100 | 0.8678 | 0.8576 | **0.8695** |
| 150 | 0.8680 | 0.8597 | **0.8695** |
| 200 | 0.8680 | 0.8611 | 0.8693 |
| 300 | 0.8682 | 0.8626 | 0.8679 |
| 400 | 0.8682 | 0.8640 | 0.8666 |
| 500 | **0.8683** | **0.8655** | 0.8655 |


### Key Findings

**Random Forest:** Classic diminishing returns curve. Gains 0.0033 AUC from 10→100 trees, then only 0.0005 AUC from 100→500 trees. **Practical sweet spot: 100 trees.**

**AdaBoost:** Slowest to start (lowest AUC at 10 trees — boosting needs rounds to accumulate), but never fully plateaus and never declines. Still climbing at 500 trees. **No hard ceiling found in tested range.**

**GradientBoosting:** Peaks at 100–150 trees (0.8695), then **actively declines** to 0.8655 at 500 trees — losing 0.0040 AUC from its own peak. This is the same overfitting mechanism as Q2: after all real patterns are learned, additional trees start fitting residual noise in the training data, which hurts validation performance.

| Method | Sweet Spot | Cost of Over-Treeing |
|---|---|---|
| RandomForest | ~100 trees | Flatlines — safe to over-tree |
| AdaBoost | Still climbing at 500 | Never hurts, but slow start |
| GradientBoosting | ~100–150 trees | **Actively gets worse after 150** |

**Production implication:** running GBM at 500 trees costs 5x more compute than 100 trees while *losing* AUC — not just diminishing returns, but negative returns. Knowing your model's tree curve is how you right-size infrastructure before deployment.

---

## Production Architecture

How this notebook-based system would be redesigned for real-time loan scoring at scale:

```
Loan Application (JSON)
        ↓
   Load Balancer
        ↓
[FastAPI Replica 1] [FastAPI Replica 2] [FastAPI Replica N]
        ↓                                    ↑
Feature Engineering Pipeline          Kubernetes HPA
(same logic as notebooks,             (auto-scales on traffic)
 now in reusable Python functions)
        ↓
Trained RF Model
(loaded from MLflow Model Registry)
        ↓
Risk Score (0.0 → 1.0) in <200ms

Background — Airflow DAG (weekly):
Fresh Data → Clean → Engineer Features → Retrain → Log to MLflow → Promote if AUC improves

Background — Drift Monitor (continuous):
Watch feature distributions + prediction distributions → alert or trigger retraining on drift
```

**Component mapping to existing projects:**
- FastAPI serving → QueryCraft Enterprise pattern
- Docker + Kubernetes HPA → Diffusion Model Lifecycle pattern
- Airflow retraining DAG → Diffusion Model Lifecycle 8-task DAG
- MLflow experiment tracking → used across both projects

**Two production failure modes to monitor:**
- **Data drift** — input distribution shifts (e.g. average debt ratio rises economy-wide in 2026)
- **Concept drift** — feature-to-outcome relationship changes (a debt ratio of 0.8 means something different in a different economic environment)

Neither throws an error. Both silently degrade model quality. Drift monitoring is the only defence.

---

## Tech Stack

```
Python · scikit-learn · pandas · numpy · matplotlib · MLflow
```

---

## Results Summary

| Question | Finding |
|---|---|
| Q1: Key hyperparameter per method | RF: `max_depth` (swing 0.024) · AdaBoost: `learning_rate` (swing 0.355) · GBM: `learning_rate` (swing 0.275) |
| Q2: Bagging vs. boosting on noise | RF lost 0.0073 AUC vs. 0.025–0.027 for boosting (3.5x gap). GBM's advantage disappears at 15–20% noise. |
| Q3: Diminishing returns threshold | RF and GBM both plateau at ~100 trees. GBM actively declines after 150 — negative returns at 500 trees. |
