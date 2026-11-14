"""
Check whether dropping low-importance features helps, using only the training set.

1. Permutation importance (ROC AUC) computed on the validation fold of each CV split,
   so the test set is never used to choose features.
2. Compare CV scores of the baseline against dropping each weak feature alone,
   and against dropping all weak features together, on identical CV splits.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedStratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

NUM_COLS = ["senior_citizen", "tenure", "monthly_charges", "total_charges"]
SCORING = ["roc_auc", "f1"]

# Pick the model to test. Use the one train.py chose as best (with its tuned params if you like).
MODEL = LogisticRegression(max_iter=1000)
# MODEL = RandomForestClassifier(random_state=42, n_jobs=-1)


def make_pipeline(columns):
    num = [c for c in columns if c in NUM_COLS]
    cat = [c for c in columns if c not in NUM_COLS]
    prep = ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="mean")), ("sc", StandardScaler())]), num),
        ("cat", Pipeline([
            ("imp", SimpleImputer(strategy="most_frequent")),
            ("ohe", OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False)),
        ]), cat),
    ], verbose_feature_names_out=False)
    return Pipeline([("prep", prep), ("model", clone(MODEL))])


ROOT = Path(__file__).resolve().parents[1]  # project root
df = pd.read_csv(ROOT / "data" / "customer_churn.csv")
X = df.drop(columns=["churn", "customer_id"])
y = (df["churn"] == "Yes").astype(int)

# Same split as train.py; X_test is not touched in this script.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=42)
splits = list(cv.split(X_train, y_train))

# ---- 1. Permutation importance on validation folds -------------------------
fold_importances = []
for train_idx, val_idx in splits:
    pipe = make_pipeline(X_train.columns).fit(X_train.iloc[train_idx], y_train.iloc[train_idx])
    r = permutation_importance(
        pipe, X_train.iloc[val_idx], y_train.iloc[val_idx],
        scoring="roc_auc", n_repeats=5, random_state=42, n_jobs=-1,
    )
    fold_importances.append(r.importances_mean)

fold_importances = pd.DataFrame(fold_importances, columns=X_train.columns)
imp = pd.DataFrame({
    "mean": fold_importances.mean(),
    "std": fold_importances.std(),
}).sort_values("mean", ascending=False)
# A feature is "weak" when its importance can't be told apart from zero.
imp["weak"] = imp["mean"] - imp["std"] <= 0

print("Permutation importance (ROC AUC drop, across CV validation folds)")
print(imp.round(4).to_string())
print()

weak = imp.index[imp["weak"]].tolist()
print(f"Weak features ({len(weak)}): {weak}\n")


# ---- 2. Compare drop candidates on identical CV splits ---------------------
def cv_scores(columns):
    res = cross_validate(make_pipeline(columns), X_train[columns], y_train,
                         cv=splits, scoring=SCORING, n_jobs=-1)
    return {s: res[f"test_{s}"] for s in SCORING}


all_cols = X_train.columns.tolist()
base = cv_scores(all_cols)

candidates = {f"drop {c}": [c] for c in weak}
if len(weak) > 1:
    candidates["drop all weak"] = weak

rows = [{
    "setting": "baseline (all features)",
    "n_features": len(all_cols),
    **{f"{s}_mean": base[s].mean() for s in SCORING},
    **{f"{s}_diff": 0.0 for s in SCORING},
    **{f"{s}_diff_std": 0.0 for s in SCORING},
}]
for label, drop in candidates.items():
    cols = [c for c in all_cols if c not in drop]
    sc = cv_scores(cols)
    row = {"setting": label, "n_features": len(cols)}
    for s in SCORING:
        diff = sc[s] - base[s]  # paired: same folds, so the difference is less noisy
        row[f"{s}_mean"] = sc[s].mean()
        row[f"{s}_diff"] = diff.mean()
        row[f"{s}_diff_std"] = diff.std()
    rows.append(row)

result = pd.DataFrame(rows).set_index("setting")
print("CV comparison vs baseline (diff > 0 means dropping helped)")
print(result.round(4).to_string())
print()

all_weak = result.loc["drop all weak"] if "drop all weak" in result.index else None
if all_weak is not None:
    if all_weak["roc_auc_diff"] >= -all_weak["roc_auc_diff_std"]:
        print("Dropping all weak features does not hurt ROC AUC beyond noise -> safe to drop for a simpler model.")
    else:
        print("Dropping all weak features hurts ROC AUC -> keep them (they carry small but real signal together).")
