"""
Synthetic Customer Churn Dataset Generator
==========================================

Generates a fictional telecommunications customer churn dataset
(10,000 customers) for supervised binary classification practice.

*** THIS DATASET IS 100% SYNTHETIC. IT CONTAINS NO REAL CUSTOMER DATA. ***

Reproducibility: all randomness flows from a single NumPy Generator seeded
with RANDOM_STATE = 42. Running this script twice produces an identical CSV.

Usage:
    python src/generate_customer_churn.py            # writes data/customer_churn.csv
    python src/generate_customer_churn.py --out path.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

RANDOM_STATE = 42
N_CUSTOMERS = 10_000
MAX_TENURE = 72

YES, NO = "Yes", "No"
NO_PHONE = "No phone service"
NO_INTERNET = "No internet service"

INTERNET_ADDONS = [
    "online_security",
    "online_backup",
    "device_protection",
    "tech_support",
    "streaming_tv",
    "streaming_movies",
]

COLUMN_ORDER = [
    "customer_id", "gender", "senior_citizen", "partner", "dependents",
    "tenure", "phone_service", "multiple_lines", "internet_service",
    "online_security", "online_backup", "device_protection", "tech_support",
    "streaming_tv", "streaming_movies", "contract", "paperless_billing",
    "payment_method", "monthly_charges", "total_charges", "churn",
]


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def yes_no(mask: np.ndarray) -> np.ndarray:
    return np.where(mask, YES, NO)


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------
def generate(n: int = N_CUSTOMERS, seed: int = RANDOM_STATE) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    # ---- Customer information --------------------------------------------
    customer_num = rng.permutation(np.arange(100_000, 100_000 + n * 9))[:n]
    customer_id = np.array([f"CUST-{x:06d}" for x in customer_num])

    gender = rng.choice(["Male", "Female"], size=n, p=[0.50, 0.50])
    senior = (rng.random(n) < 0.16).astype(int)

    # Partners slightly less common among seniors? Keep mild.
    partner_p = np.where(senior == 1, 0.45, 0.49)
    partner = rng.random(n) < partner_p
    # Dependents strongly tied to partner, rarer for seniors
    dep_p = np.where(partner, 0.52, 0.10) * np.where(senior == 1, 0.35, 1.0)
    dependents = rng.random(n) < dep_p

    # ---- Contract (drives tenure distribution) ---------------------------
    # Families / partners lean toward longer contracts.
    base = np.column_stack([
        np.full(n, 0.0),                       # Month-to-month (reference)
        np.full(n, -0.55),                     # One year
        np.full(n, -0.60),                     # Two year
    ])
    base[:, 1] += 0.35 * partner + 0.25 * dependents
    base[:, 2] += 0.55 * partner + 0.45 * dependents - 0.40 * senior
    probs = np.exp(base)
    probs /= probs.sum(axis=1, keepdims=True)
    contract_idx = (probs.cumsum(axis=1) > rng.random((n, 1))).argmax(axis=1)
    contract = np.array(["Month-to-month", "One year", "Two year"])[contract_idx]

    # ---- Tenure (0–72), contract-dependent and overlapping ---------------
    # Beta distributions on [0,1], scaled to 0..72.
    a = np.select([contract_idx == 0, contract_idx == 1], [0.75, 1.6], 2.6)
    b = np.select([contract_idx == 0, contract_idx == 1], [2.2, 1.3], 0.75)
    tenure = np.floor(rng.beta(a, b) * (MAX_TENURE + 1)).astype(int)
    tenure = np.clip(tenure, 0, MAX_TENURE)
    # Brand-new customers (tenure 0) are only on fresh signups – ~1%.
    newbie = rng.random(n) < 0.012
    tenure[newbie] = 0

    # ---- Services --------------------------------------------------------
    phone = rng.random(n) < 0.90
    multiple_lines = np.where(
        ~phone, NO_PHONE,
        yes_no(rng.random(n) < sigmoid(-0.6 + 0.025 * tenure)),
    )

    internet_idx = rng.choice(3, size=n, p=[0.34, 0.44, 0.22])  # DSL, Fiber, No
    # Customers with no phone must have internet (otherwise no service at all).
    no_service = (~phone) & (internet_idx == 2)
    internet_idx[no_service] = rng.choice([0, 1], size=no_service.sum(), p=[0.7, 0.3])
    internet = np.array(["DSL", "Fiber optic", "No"])[internet_idx]
    has_net = internet_idx != 2

    # Add-on adoption: rises with tenure and longer contracts.
    loyalty = 0.02 * tenure + 0.35 * contract_idx
    addon_base = {
        "online_security":   -1.3,
        "online_backup":     -0.9,
        "device_protection": -0.9,
        "tech_support":      -1.15,
        "streaming_tv":      -0.5,
        "streaming_movies":  -0.45,
    }
    fiber_shift = {  # Fiber users stream more, buy security a bit less
        "online_security": -0.3, "online_backup": 0.0, "device_protection": 0.1,
        "tech_support": -0.3, "streaming_tv": 0.6, "streaming_movies": 0.6,
    }
    addons: dict[str, np.ndarray] = {}
    for col in INTERNET_ADDONS:
        logit = addon_base[col] + loyalty + fiber_shift[col] * (internet_idx == 1)
        if col == "streaming_movies":  # correlated with streaming_tv
            logit = logit + 1.2 * (addons["streaming_tv"] == YES) - 0.6
        val = yes_no(rng.random(n) < sigmoid(logit))
        addons[col] = np.where(has_net, val, NO_INTERNET)

    # ---- Billing ---------------------------------------------------------
    paperless = rng.random(n) < (0.48 + 0.18 * has_net - 0.10 * senior + 0.05 * (internet_idx == 1))

    pm_logits = np.column_stack([
        0.25 + 0.6 * paperless - 0.35 * contract_idx + 0.4 * senior,   # Electronic check
        0.10 - 0.8 * paperless + 0.05 * contract_idx,                  # Mailed check
        -0.25 + 0.30 * contract_idx,                                   # Bank transfer
        -0.25 + 0.30 * contract_idx,                                   # Credit card
    ])
    pm_p = np.exp(pm_logits)
    pm_p /= pm_p.sum(axis=1, keepdims=True)
    pm_idx = (pm_p.cumsum(axis=1) > rng.random((n, 1))).argmax(axis=1)
    payment = np.array(
        ["Electronic check", "Mailed check", "Bank transfer", "Credit card"]
    )[pm_idx]

    # Monthly charges: additive price list + small noise.
    price = np.zeros(n)
    price += np.where(phone, 20.0, 0.0)
    price += np.where(multiple_lines == YES, 5.0, 0.0)
    price += np.select([internet_idx == 0, internet_idx == 1], [25.0, 45.0], 0.0)
    addon_price = {
        "online_security": 5.0, "online_backup": 5.0, "device_protection": 5.0,
        "tech_support": 5.0, "streaming_tv": 10.0, "streaming_movies": 10.0,
    }
    for col, p in addon_price.items():
        price += np.where(addons[col] == YES, p, 0.0)
    price += rng.normal(0, 2.0, n)  # promos, taxes, rounding
    monthly = np.round(np.clip(price, 18.25, 120.0), 2)

    # Total charges: tenure * monthly with historical price drift (+/-).
    # Past prices differed (promos / increases), so ratio varies ~0.85–1.10.
    drift = np.clip(rng.normal(0.98, 0.05, n), 0.85, 1.10)
    total = np.round(tenure * monthly * drift, 2)
    # tenure 0 customers have not been billed yet -> 0.00
    total[tenure == 0] = 0.0

    # ---- Churn (latent logistic model + noise) ---------------------------
    z = np.full(n, -1.55)
    z += np.select([contract_idx == 0, contract_idx == 1], [0.9, -0.55], -1.55)
    # Tenure: strong early-life risk, decaying with log tenure
    z += -0.55 * (np.log1p(tenure) - np.log1p(12))
    z += 0.018 * (monthly - 65)                     # higher bill -> higher risk
    z += 0.35 * (internet_idx == 1)                 # fiber service-quality issues
    z += -0.30 * (internet_idx == 2)                # no-internet customers stable
    z += 0.35 * (addons["tech_support"] == NO)
    z += 0.30 * (addons["online_security"] == NO)
    z += 0.45 * (pm_idx == 0)                       # electronic check
    z += 0.15 * paperless
    z += 0.20 * senior
    z += -0.15 * partner - 0.15 * dependents
    z += rng.normal(0, 0.6, n)                      # unobserved heterogeneity
    p_churn = sigmoid(z)
    churn = yes_no(rng.random(n) < p_churn)

    df = pd.DataFrame({
        "customer_id": customer_id,
        "gender": gender,
        "senior_citizen": senior,
        "partner": yes_no(partner),
        "dependents": yes_no(dependents),
        "tenure": tenure,
        "phone_service": yes_no(phone),
        "multiple_lines": multiple_lines,
        "internet_service": internet,
        **addons,
        "contract": contract,
        "paperless_billing": yes_no(paperless),
        "payment_method": payment,
        "monthly_charges": monthly,
        "total_charges": total,
        "churn": churn,
    })
    return df[COLUMN_ORDER]


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
ALLOWED = {
    "gender": {"Male", "Female"},
    "senior_citizen": {0, 1},
    "partner": {YES, NO},
    "dependents": {YES, NO},
    "phone_service": {YES, NO},
    "multiple_lines": {YES, NO, NO_PHONE},
    "internet_service": {"DSL", "Fiber optic", "No"},
    **{c: {YES, NO, NO_INTERNET} for c in INTERNET_ADDONS},
    "contract": {"Month-to-month", "One year", "Two year"},
    "paperless_billing": {YES, NO},
    "payment_method": {"Electronic check", "Mailed check", "Bank transfer", "Credit card"},
    "churn": {YES, NO},
}


def tenure_group(t: pd.Series) -> pd.Series:
    return pd.cut(t, bins=[-1, 12, 24, 48, 72],
                  labels=["0-12", "13-24", "25-48", "49-72"])


def validate(df: pd.DataFrame) -> None:
    errors: list[str] = []

    def check(cond: bool, msg: str) -> None:
        if not cond:
            errors.append(msg)

    check(len(df) == N_CUSTOMERS, f"expected {N_CUSTOMERS} rows, got {len(df)}")
    check(list(df.columns) == COLUMN_ORDER, "column set/order mismatch")
    check(df["customer_id"].is_unique, "duplicate customer IDs")
    check(not df.isna().any().any(), "missing values present")

    for col, allowed in ALLOWED.items():
        bad = set(df[col].unique()) - allowed
        check(not bad, f"{col}: unexpected labels {bad}")

    check(df["tenure"].between(0, MAX_TENURE).all(), "tenure out of 0-72")
    check((df["monthly_charges"] > 0).all(), "non-positive monthly charges")
    check((df["total_charges"] >= 0).all(), "negative total charges")

    # Total-charges consistency
    billed = df["tenure"] > 0
    ratio = df.loc[billed, "total_charges"] / (df.loc[billed, "tenure"] * df.loc[billed, "monthly_charges"])
    check(ratio.between(0.80, 1.15).all(), "total_charges inconsistent with tenure*monthly")
    check((df.loc[~billed, "total_charges"] == 0).all(), "tenure 0 must have total_charges 0")

    # Structural consistency
    no_phone = df["phone_service"] == NO
    check((df.loc[no_phone, "multiple_lines"] == NO_PHONE).all(), "no-phone customers w/ bad multiple_lines")
    check((df.loc[~no_phone, "multiple_lines"] != NO_PHONE).all(), "phone customers labeled 'No phone service'")
    no_net = df["internet_service"] == "No"
    for c in INTERNET_ADDONS:
        check((df.loc[no_net, c] == NO_INTERNET).all(), f"{c}: no-internet customers w/ bad label")
        check((df.loc[~no_net, c] != NO_INTERNET).all(), f"{c}: internet customers labeled 'No internet service'")
    check(not (no_phone & no_net).any(), "customers with no services at all")

    # Statistical targets
    rate = (df["churn"] == YES).mean()
    check(0.20 <= rate <= 0.30, f"overall churn {rate:.3f} outside 20-30%")

    seg_cols = {
        "contract": df["contract"],
        "tenure_group": tenure_group(df["tenure"]),
        "payment_method": df["payment_method"],
        "internet_service": df["internet_service"],
        "senior_citizen": df["senior_citizen"],
        "gender": df["gender"],
    }
    for name, seg in seg_cols.items():
        rates = (df["churn"] == YES).groupby(seg, observed=True).mean()
        check(((rates > 0) & (rates < 1)).all(), f"{name}: a segment has 0% or 100% churn")
        if name in {"contract", "tenure_group", "payment_method"}:
            check(rates.max() - rates.min() > 0.05, f"{name}: churn rates do not differ")

    # No single feature should perfectly determine churn
    for col in df.columns.drop(["customer_id", "churn", "total_charges", "monthly_charges"]):
        r = (df["churn"] == YES).groupby(df[col]).mean()
        check(((r > 0) & (r < 1)).all(), f"{col}: some value perfectly determines churn")

    if errors:
        print("VALIDATION FAILED:")
        for e in errors:
            print("  -", e)
        sys.exit(1)
    print(f"Validation passed: {len(df)} rows, {df.shape[1]} columns, all checks OK.")


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def report(df: pd.DataFrame) -> None:
    pd.set_option("display.width", 120)
    churn01 = (df["churn"] == YES).astype(int)

    print("\n=== Summary statistics (numerical) ===")
    print(df[["senior_citizen", "tenure", "monthly_charges", "total_charges"]]
          .describe().round(2).to_string())

    print("\n=== Frequency distributions (categorical) ===")
    for col in df.columns.drop(["customer_id", "tenure", "monthly_charges", "total_charges"]):
        vc = df[col].value_counts()
        pct = (vc / len(df) * 100).round(1)
        print(f"\n{col}:")
        print(pd.DataFrame({"count": vc, "pct": pct}).to_string())

    print(f"\n=== Overall churn rate: {churn01.mean():.2%} ===")
    for name, seg in [
        ("contract", df["contract"]),
        ("tenure_group", tenure_group(df["tenure"])),
        ("payment_method", df["payment_method"]),
        ("internet_service", df["internet_service"]),
        ("tech_support", df["tech_support"]),
    ]:
        g = churn01.groupby(seg, observed=True).agg(["count", "mean"])
        g.columns = ["customers", "churn_rate"]
        g["churn_rate"] = (g["churn_rate"] * 100).round(1).astype(str) + "%"
        print(f"\nChurn rate by {name}:\n{g.to_string()}")


def main() -> None:
    ap = argparse.ArgumentParser()
    default_out = Path(__file__).resolve().parents[1] / "data" / "customer_churn.csv"
    ap.add_argument("--out", default=str(default_out))
    args = ap.parse_args()

    df = generate()
    validate(df)          # exits non-zero before export if anything fails
    df.to_csv(args.out, index=False)
    print(f"Wrote {args.out}")
    report(df)


if __name__ == "__main__":
    main()
