# Customer Churn Prediction

Predict whether a telecom customer will churn, using a **synthetic** dataset of 10,000 customers.

> The dataset is fully synthetic and contains no real customer data. See [DATA_CARD.md](DATA_CARD.md) for the columns, statistics and how the data was generated.

## Project structure

| File | Description |
|---|---|
| `generate_customer_churn.py` | Generates the synthetic dataset (`random_state = 42`, reproducible) |
| `customer_churn.csv` | The generated dataset (10,000 rows × 21 columns) |
| `DATA_CARD.md` | Data dictionary, summary statistics and generation assumptions |
| `train.py` | Preprocessing, model selection and hyperparameter tuning |

## Getting started

```bash
pip install numpy pandas scikit-learn

# (optional) regenerate the dataset
python generate_customer_churn.py

# train and evaluate
python train.py
```

## Approach

1. **Preprocessing** (`ColumnTransformer` inside a `Pipeline`, so it is fit on training folds only)
   - Numerical: mean imputation + `StandardScaler`
   - Categorical: most-frequent imputation + `OneHotEncoder(drop="first")`
   - `customer_id` is dropped; target `churn` is encoded as Yes = 1, No = 0
2. **Split:** 80/20 stratified train/test split
3. **Model selection:** Logistic Regression vs Random Forest, 5-fold CV on F1
4. **Tuning:** `GridSearchCV` on the winning model, using a parameter grid per model
5. **Evaluation:** accuracy, confusion matrix and classification report on the held-out test set

## Results

| Model | CV F1 (default params) |
|---|---:|
| Logistic Regression | 0.636 |
| Random Forest | 0.596 |

Best after tuning: **Logistic Regression** with `C=10`, `class_weight="balanced"`, `fit_intercept=True` (CV F1 0.665).

Test set (2,000 customers):

| Class | Precision | Recall | F1 |
|---|---:|---:|---:|
| No churn | 0.89 | 0.77 | 0.83 |
| Churn | 0.57 | 0.76 | 0.65 |

Accuracy is **0.77**, against a 0.72 baseline from always predicting "no churn". The balanced class weight trades precision for recall, so the model catches about 3 in 4 churners.
