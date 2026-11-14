# Customer Churn Prediction

Predict whether a telecom customer will churn, using a **synthetic** dataset of 10,000 customers.

> The dataset is fully synthetic and contains no real customer data. See [data/DATA_CARD.md](data/DATA_CARD.md) for the columns, statistics and how the data was generated.

## Project structure

```
Customer-Churn-Prediction/
├── data/
│   ├── customer_churn.csv          # dataset (10,000 rows × 21 columns)
│   └── DATA_CARD.md                # data dictionary and generation assumptions
├── notebooks/
│   └── train.ipynb                 # the full process, step by step, with charts
├── src/
│   ├── generate_customer_churn.py  # makes the synthetic dataset (random_state = 42)
│   └── train.py                    # trains the final model; saves models/model.joblib
├── models/                         # trained model (not committed)
├── requirements.txt
└── README.md
```

- **`notebooks/train.ipynb`**: how every choice was made (EDA, baseline, log tenure, feature selection, tuning, threshold).
- **`src/train.py`**: only the final choices, to retrain the model quickly.

## Getting started

```bash
pip install -r requirements.txt

# (optional) regenerate the dataset
python src/generate_customer_churn.py

# train, evaluate and save the model
python src/train.py
```

Use the saved model:

```python
import joblib
model = joblib.load("models/model.joblib")   # preprocessing and threshold are included
model.predict(new_customers)                  # DataFrame with the 7 feature columns below
```

## Approach

1. **Explore:** churn is 28% overall, 53% for month-to-month contracts, and drops fast in the first year of tenure.
2. **Split:** 80/20 stratified. Every decision uses 5-fold CV on the train set; the test set is used once at the end.
3. **Baseline:** Logistic Regression with all 19 features (CV ROC AUC 0.8495).
4. **Log tenure:** `log1p(tenure)` fits the fast early drop better (0.8495 → 0.8532).
5. **Feature selection:** permutation importance on CV validation folds; features that lower ROC AUC by less than 0.002 when shuffled are dropped. 7 of 19 remain, with the same score (0.8538):
   - `contract`, `tenure`, `monthly_charges`, `internet_service`, `payment_method`, `tech_support`, `online_security`
6. **Tune models:** Logistic Regression, Random Forest and HistGradientBoosting, each tuned with `GridSearchCV` (ROC AUC); the best tuned score wins.
7. **Threshold:** `TunedThresholdClassifierCV` picks the threshold that maximises CV F1.

## Results

| Model | CV ROC AUC (tuned) |
|---|---:|
| Logistic Regression | **0.854** |
| HistGradientBoosting | 0.850 |
| Random Forest | 0.848 |

Best: **Logistic Regression** (`C=1`, no class weight). Decision threshold **0.30** (CV F1 0.671).

Test set (2,000 customers), ROC AUC **0.855**:

| Class | Precision | Recall | F1 |
|---|---:|---:|---:|
| No churn | 0.89 | 0.80 | 0.84 |
| Churn | 0.59 | 0.75 | 0.66 |

Accuracy is **0.78**, against a 0.72 baseline from always predicting "no churn". The model catches 3 in 4 churners. To favour precision (fewer false alarms) or recall (fewer missed churners), change the `scoring` of `TunedThresholdClassifierCV` in `src/train.py`.
