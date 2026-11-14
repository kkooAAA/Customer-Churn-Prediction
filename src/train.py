from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import GridSearchCV, TunedThresholdClassifierCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

# --- Paths ---
ROOT = Path(__file__).resolve().parents[1]  # project root
MODEL_PATH = ROOT / "models" / "model.joblib"

# --- Load data ---
df = pd.read_csv(ROOT / "data" / "customer_churn.csv")

# 8 features kept by feature_selection.py
num_cols = ["senior_citizen", "monthly_charges"]
log_cols = ["tenure"]  # log-transformed below
cat_cols = ["contract", "tech_support", "internet_service", "payment_method", "online_security"]

X = df[num_cols + log_cols + cat_cols]
y = (df["churn"] == "Yes").astype(int)  # Yes = 1, No = 0

# --- Preprocessing ---
# Numbers: fill missing with mean, then scale
num_pipelines = Pipeline([
    ("imp", SimpleImputer(strategy="mean")),
    ("sc", StandardScaler())
])

# Tenure: churn risk drops fast in early months, so use log1p
log_pipelines = Pipeline([
    ("imp", SimpleImputer(strategy="mean")),
    ("log", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
    ("sc", StandardScaler())
])

# Categories: fill missing with most frequent, then one-hot encode
cat_pipelines = Pipeline([
    ("imp", SimpleImputer(strategy="most_frequent")),
    ("ohe", OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False))
])

prep = ColumnTransformer([
    ("num", num_pipelines, num_cols),
    ("tenure", log_pipelines, log_cols),
    ("cat", cat_pipelines, cat_cols)
], verbose_feature_names_out=False)

# --- Train/test split (80/20, keep churn ratio) ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# --- Candidate models ---
models = {
    "LR": LogisticRegression(max_iter=1000),
    "RF": RandomForestClassifier(random_state=42),
    "HGB": HistGradientBoostingClassifier(random_state=42),
}

# Let CV decide: log tenure or raw tenure (trees don't need this)
tenure_options = {"prep__tenure__log": [FunctionTransformer(np.log1p, feature_names_out="one-to-one"), "passthrough"]}

param_grids = {
    "LR": {
        **tenure_options,
        "model__C": [0.01, 0.1, 1, 10, 100],
        "model__class_weight": ["balanced", None],
    },
    "RF": {
        "model__n_estimators": [300],
        "model__max_depth": [5, 10, 20],
        "model__min_samples_leaf": [1, 5, 20],
        "model__class_weight": ["balanced", None],
    },
    "HGB": {
        "model__learning_rate": [0.03, 0.1],
        "model__max_leaf_nodes": [7, 15, 31],
        "model__l2_regularization": [0.0, 1.0],
        "model__class_weight": ["balanced", None],
    },
}

# --- Tune each model, then compare ---
# ROC AUC doesn't depend on the threshold (that's tuned next)
searches = {}
for name, model in models.items():
    search = GridSearchCV(
        Pipeline([("prep", prep), ("model", model)]),
        param_grid=param_grids[name],
        cv=5,
        scoring="roc_auc",
        n_jobs=-1,
    )
    search.fit(X_train, y_train)
    searches[name] = search
    print(f"{name} CV ROC AUC: {search.best_score_:.4f}  {search.best_params_}")

best_name = max(searches, key=lambda n: searches[n].best_score_)
print(f"Best model: {best_name}")

# --- Pick the decision threshold (instead of 0.5) ---
# Maximises CV F1 on train data; change scoring to favour recall or precision
best_model = TunedThresholdClassifierCV(
    searches[best_name].best_estimator_,
    scoring="f1",
    cv=5,
    n_jobs=-1,
)
best_model.fit(X_train, y_train)
print(f"Threshold: {best_model.best_threshold_:.3f}  (CV F1: {best_model.best_score_:.4f})")

# Save model (includes preprocessing and threshold)
MODEL_PATH.parent.mkdir(exist_ok=True)
joblib.dump(best_model, MODEL_PATH)
print(f"Saved {MODEL_PATH}")

# --- Final test (use the test set only here) ---
y_proba = best_model.predict_proba(X_test)[:, 1]
y_pred = best_model.predict(X_test)

print(f"Test ROC AUC: {roc_auc_score(y_test, y_proba):.4f}")
print(f"Test accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

# --- Feature importance (for reporting only, not for choosing features) ---
# How much test ROC AUC drops when each feature is shuffled
result = permutation_importance(
    best_model,
    X_test, y_test,
    scoring="roc_auc",
    n_repeats=10,
    random_state=42,
    n_jobs=-1
)

importance = pd.Series(
    result.importances_mean,
    index=X_test.columns
).sort_values(ascending=False)

print(importance)
