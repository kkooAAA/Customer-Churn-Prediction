# Customer Churn Prediction

Predict whether a telecom customer will churn, using a **synthetic** dataset of 10,000 customers.

> The dataset is fully synthetic and contains no real customer data. See [data/DATA_CARD.md](data/DATA_CARD.md) for the columns, statistics and how the data was generated.

## Project structure

```
Customer-Churn-Prediction/
├── data/
│   ├── customer_churn.csv          # dataset (10,000 rows × 21 columns)
│   └── DATA_CARD.md                # data dictionary and generation assumptions
├── src/
│   ├── generate_customer_churn.py  # makes the synthetic dataset (random_state = 42)
│   ├── feature_selection.py        # checks which features can be dropped (CV on train only)
│   ├── train.py                    # tune, pick threshold, evaluate; saves models/model.joblib
│   └── visualize.py                # charts -> figures/
├── notebooks/
│   └── train.ipynb                 # same steps as train.py, with charts inline
├── models/                         # trained model (not committed)
├── figures/                        # charts from visualize.py
├── requirements.txt
└── README.md
```

## Getting started

Scripts find files relative to the project root, so they can be run from any folder.

```bash
pip install -r requirements.txt

# (optional) regenerate the dataset
python src/generate_customer_churn.py

# (optional) re-run the feature check
python src/feature_selection.py

# train, evaluate and save the model
python src/train.py

# make the charts (needs the trained model)
python src/visualize.py
```

Use the saved model:

```python
import joblib
model = joblib.load("models/model.joblib")   # preprocessing and threshold are included
model.predict(new_customers)           # DataFrame with the 8 feature columns below
```

## Approach

1. **Feature selection** (`src/feature_selection.py`): permutation importance (ROC AUC) on CV validation folds of the training set. Features whose importance was within one std of zero were dropped together, and CV ROC AUC / F1 did not fall (0.8506 → 0.8511). 8 of 19 features remain:
   - Numerical: `senior_citizen`, `tenure`, `monthly_charges`
   - Categorical: `contract`, `tech_support`, `internet_service`, `payment_method`, `online_security`
2. **Preprocessing** (`ColumnTransformer` inside a `Pipeline`, so it is fit on training folds only)
   - Numerical: mean imputation + `StandardScaler`
   - `tenure`: mean imputation + `log1p` + `StandardScaler` (the grid also tries no log; CV keeps the log)
   - Categorical: most-frequent imputation + `OneHotEncoder(drop="first")`
3. **Split:** 80/20 stratified train/test split
4. **Model selection:** Logistic Regression, Random Forest and HistGradientBoosting are each tuned with `GridSearchCV` (5-fold, ROC AUC); the best tuned score wins
5. **Threshold:** `TunedThresholdClassifierCV` picks the decision threshold that maximises CV F1 on the training set
6. **Evaluation:** ROC AUC, accuracy, confusion matrix and classification report on the held-out test set, used once at the end

## Results

| Model | CV ROC AUC (tuned) |
|---|---:|
| Logistic Regression | **0.854** |
| HistGradientBoosting | 0.849 |
| Random Forest | 0.849 |

Best: **Logistic Regression**, `C=100`, no class weight, log tenure. Decision threshold **0.359** (CV F1 0.671).

Test set (2,000 customers), ROC AUC **0.854**:

| Class | Precision | Recall | F1 |
|---|---:|---:|---:|
| No churn | 0.88 | 0.83 | 0.85 |
| Churn | 0.62 | 0.71 | 0.66 |

Accuracy is **0.80**, against a 0.72 baseline from always predicting "no churn". The model catches about 7 in 10 churners. To favour recall (fewer missed churners, more false alarms), change the `scoring` of `TunedThresholdClassifierCV` in `src/train.py`.
