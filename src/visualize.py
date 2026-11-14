"""
Make charts for the churn data and the trained model.
Run train.py first (needs models/model.joblib). Saves PNGs to figures/.
"""
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import confusion_matrix, precision_recall_curve
from sklearn.model_selection import train_test_split

# --- Style (light theme) ---
SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_2 = "#52514e"
GRID = "#e4e3df"
BLUE = "#2a78d6"     # series 1
ORANGE = "#eb6834"   # series 2
BLUE_RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]

plt.rcParams.update({
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID,
    "axes.labelcolor": TEXT_2,
    "axes.titlecolor": TEXT,
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "axes.titlelocation": "left",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "axes.axisbelow": True,  # grid behind bars
    "grid.color": GRID,
    "grid.linewidth": 0.8,
    "xtick.color": TEXT_2,
    "ytick.color": TEXT_2,
    "font.size": 10,
})

ROOT = Path(__file__).resolve().parents[1]  # project root
OUT = ROOT / "figures"
OUT.mkdir(exist_ok=True)


def save(fig, name):
    fig.tight_layout()
    fig.savefig(OUT / name, dpi=150)
    plt.close(fig)
    print(f"Saved figures/{name}")


# --- Data and model (same split as train.py) ---
df = pd.read_csv(ROOT / "data" / "customer_churn.csv")
model = joblib.load(ROOT / "models" / "model.joblib")
features = list(model.feature_names_in_)

X = df[features]
y = (df["churn"] == "Yes").astype(int)
_, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

overall = y.mean()

# --- 1. Churn rate by contract ---
rate = (df.assign(churn=y).groupby("contract")["churn"].mean()
        .reindex(["Month-to-month", "One year", "Two year"]))

fig, ax = plt.subplots(figsize=(6.5, 4))
bars = ax.bar(rate.index, rate.values, color=BLUE, width=0.55)
ax.bar_label(bars, labels=[f"{v:.0%}" for v in rate.values], padding=4, color=TEXT)
ax.axhline(overall, color=TEXT_2, linewidth=1, linestyle="--")
ax.text(2.35, overall, f"All customers {overall:.0%}", va="bottom", ha="right", color=TEXT_2, fontsize=9)
ax.set_title("Month-to-month customers churn the most")
ax.set_ylabel("Churn rate")
ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
ax.grid(axis="x", visible=False)
ax.set_ylim(0, rate.max() * 1.2)
save(fig, "1_churn_by_contract.png")

# --- 2. Churn rate by tenure (why tenure gets a log) ---
bins = np.arange(0, 76, 4)
tenure_rate = (df.assign(churn=y, bin=pd.cut(df["tenure"], bins, right=False))
               .groupby("bin", observed=True)["churn"].mean())
mid = [b.left + 2 for b in tenure_rate.index]

fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(mid, tenure_rate.values, color=BLUE, linewidth=2, marker="o", markersize=5)
ax.annotate("Risk drops fast in the first year,\nthen levels off",
            xy=(mid[2], tenure_rate.iloc[2]), xytext=(22, tenure_rate.iloc[0] * 0.9),
            color=TEXT_2, fontsize=9, arrowprops=dict(arrowstyle="-", color=TEXT_2, linewidth=0.8))
ax.set_title("Churn rate by tenure")
ax.set_xlabel("Tenure (months, 4-month bins)")
ax.set_ylabel("Churn rate")
ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
ax.set_ylim(0, None)
save(fig, "2_churn_by_tenure.png")

# --- 3. Feature importance ---
result = permutation_importance(model, X_test, y_test, scoring="roc_auc",
                                n_repeats=10, random_state=42, n_jobs=-1)
imp = pd.Series(result.importances_mean, index=features).sort_values()
err = pd.Series(result.importances_std, index=features)[imp.index]

fig, ax = plt.subplots(figsize=(7, 4.2))
ax.barh(imp.index, imp.values, xerr=err.values, color=BLUE, height=0.6,
        error_kw=dict(ecolor=TEXT_2, elinewidth=1, capsize=2))
for i, v in enumerate(imp.values):
    ax.text(max(v, 0) + err.iloc[i] + 0.002, i, f"{v:.3f}", va="center", color=TEXT, fontsize=9)
ax.set_title("Contract and tenure drive the predictions")
ax.set_xlabel("Drop in test ROC AUC when the feature is shuffled")
ax.grid(axis="y", visible=False)
ax.set_xlim(0, imp.max() * 1.18)
save(fig, "3_feature_importance.png")

# --- 4. Precision and recall vs threshold ---
proba = model.predict_proba(X_test)[:, 1]
precision, recall, thresholds = precision_recall_curve(y_test, proba)
t = model.best_threshold_

fig, ax = plt.subplots(figsize=(7, 4.2))
ax.plot(thresholds, precision[:-1], color=BLUE, linewidth=2)
ax.plot(thresholds, recall[:-1], color=ORANGE, linewidth=2)
ax.axvline(t, color=TEXT_2, linewidth=1, linestyle="--")
ax.text(t - 0.01, 0.04, f"Chosen {t:.2f}", color=TEXT_2, fontsize=9, ha="right")
ax.axvline(0.5, color=TEXT_2, linewidth=0.8, linestyle=":")
ax.text(0.51, 0.04, "Default 0.50", color=TEXT_2, fontsize=9)
# Direct labels near the right end of each line
end = thresholds <= 0.78
# Precision rises to the right, so label below it; recall falls, so label above it
ax.text(0.79, precision[:-1][end][-1] - 0.05, "Precision", color=TEXT, fontsize=10, va="top")
ax.text(0.79, recall[:-1][end][-1] + 0.05, "Recall", color=TEXT, fontsize=10, va="bottom")
ax.legend(["Precision", "Recall"], loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2,
          frameon=False, labelcolor=TEXT)
ax.set_title("Precision and recall by threshold", pad=26)
ax.set_xlabel("Decision threshold (predicted churn probability)")
ax.set_ylabel("Score on test set")
ax.set_xlim(0, 0.9)  # few customers score above 0.9, so the curve gets noisy there
ax.set_ylim(0, 1.05)
save(fig, "4_threshold.png")

# --- 5. Confusion matrix ---
cm = confusion_matrix(y_test, model.predict(X_test))
labels = ["No churn", "Churn"]

fig, ax = plt.subplots(figsize=(5, 4.2))
cmap = plt.matplotlib.colors.LinearSegmentedColormap.from_list("blue", BLUE_RAMP)
ax.imshow(cm, cmap=cmap)
for i in range(2):
    for j in range(2):
        dark_cell = cm[i, j] > cm.max() * 0.5
        ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center", fontsize=14,
                color="white" if dark_cell else TEXT)
ax.set_xticks([0, 1], labels)
ax.set_yticks([0, 1], labels)
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
ax.grid(False)
for s in ax.spines.values():
    s.set_visible(False)
ax.set_title(f"Test set: {cm[1, 1]} of {cm[1].sum()} churners caught")
save(fig, "5_confusion_matrix.png")
