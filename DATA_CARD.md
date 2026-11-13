# Synthetic Telecom Customer Churn Dataset: Data Card

> **This dataset is entirely SYNTHETIC.** The generator script created every record from statistical rules and random draws. It contains no real customer data, and no record describes a real person or account. Any resemblance to real customers or companies is coincidental.

- **File:** `customer_churn.csv` (10,000 rows × 21 columns, UTF-8, comma-separated, header row)
- **Generator:** `generate_customer_churn.py` (Python 3, NumPy, Pandas), `random_state = 42`
- **Reproducibility:** re-running the script produces a byte-identical CSV (checked with md5).
- **Task:** binary classification. Target = `churn` (Yes/No)

---

## 1. Data dictionary

| # | Column | Type | Allowed values / range | Description |
|---|---|---|---|---|
| 1 | `customer_id` | string | `CUST-######` (unique) | Random, non-sequential identifier. Carries no information. Drop it before modelling. |
| 2 | `gender` | categorical | Male, Female | Customer gender. Has no effect on churn by design. |
| 3 | `senior_citizen` | integer (binary) | 0, 1 | 1 if the customer is a senior citizen |
| 4 | `partner` | categorical | Yes, No | Customer has a partner |
| 5 | `dependents` | categorical | Yes, No | Customer has dependents |
| 6 | `tenure` | integer | 0–72 | Months with the company. 0 = joined this billing cycle. |
| 7 | `phone_service` | categorical | Yes, No | Has phone service |
| 8 | `multiple_lines` | categorical | Yes, No, No phone service | Has more than one phone line. "No phone service" if and only if `phone_service = No` |
| 9 | `internet_service` | categorical | DSL, Fiber optic, No | Internet service type |
| 10 | `online_security` | categorical | Yes, No, No internet service | Online-security add-on |
| 11 | `online_backup` | categorical | Yes, No, No internet service | Online-backup add-on |
| 12 | `device_protection` | categorical | Yes, No, No internet service | Device-protection add-on |
| 13 | `tech_support` | categorical | Yes, No, No internet service | Premium tech-support add-on |
| 14 | `streaming_tv` | categorical | Yes, No, No internet service | Streaming-TV add-on |
| 15 | `streaming_movies` | categorical | Yes, No, No internet service | Streaming-movies add-on |
| 16 | `contract` | categorical | Month-to-month, One year, Two year | Current contract term |
| 17 | `paperless_billing` | categorical | Yes, No | Receives paperless bills |
| 18 | `payment_method` | categorical | Electronic check, Mailed check, Bank transfer, Credit card | Payment method |
| 19 | `monthly_charges` | float (USD, 2 dp) | 18.25–~120 | Current monthly bill |
| 20 | `total_charges` | float (USD, 2 dp) | ≥ 0 | Total billed over the customer's tenure. 0.00 when `tenure = 0` |
| 21 | `churn` | categorical (target) | Yes, No | Customer left the company. This is the label. |

For columns 10–15, "No internet service" appears if and only if `internet_service = No`.

---

## 2. Summary statistics (numerical features)

| Statistic | senior_citizen | tenure | monthly_charges | total_charges |
|---|---:|---:|---:|---:|
| count | 10,000 | 10,000 | 10,000 | 10,000 |
| mean | 0.15 | 35.51 | 68.81 | 2,512.20 |
| std | 0.36 | 23.09 | 27.29 | 2,094.37 |
| min | 0 | 0 | 18.25 | 0.00 |
| 25% | 0 | 14 | 52.48 | 696.14 |
| 50% | 0 | 36 | 75.24 | 1,857.88 |
| 75% | 0 | 56 | 90.20 | 4,157.11 |
| max | 1 | 72 | 116.16 | 8,540.58 |

`monthly_charges` is bimodal: one cluster of phone-only customers at roughly $20–25 and a broad mass of internet customers at $45–115. `total_charges` is right-skewed.

---

## 3. Frequency distributions (categorical features)

| Feature | Value: count (%) |
|---|---|
| gender | Female 5,028 (50.3) · Male 4,972 (49.7) |
| senior_citizen | 0: 8,484 (84.8) · 1: 1,516 (15.2) |
| partner | No 5,145 (51.4) · Yes 4,855 (48.6) |
| dependents | No 7,281 (72.8) · Yes 2,719 (27.2) |
| phone_service | Yes 8,974 (89.7) · No 1,026 (10.3) |
| multiple_lines | Yes 5,154 (51.5) · No 3,820 (38.2) · No phone service 1,026 (10.3) |
| internet_service | Fiber optic 4,501 (45.0) · DSL 3,588 (35.9) · No 1,911 (19.1) |
| online_security | No 4,837 (48.4) · Yes 3,252 (32.5) · No internet service 1,911 (19.1) |
| online_backup | Yes 4,183 (41.8) · No 3,906 (39.1) · No internet service 1,911 (19.1) |
| device_protection | Yes 4,383 (43.8) · No 3,706 (37.1) · No internet service 1,911 (19.1) |
| tech_support | No 4,576 (45.8) · Yes 3,513 (35.1) · No internet service 1,911 (19.1) |
| streaming_tv | Yes 5,541 (55.4) · No 2,548 (25.5) · No internet service 1,911 (19.1) |
| streaming_movies | Yes 5,823 (58.2) · No 2,266 (22.7) · No internet service 1,911 (19.1) |
| contract | Month-to-month 4,041 (40.4) · Two year 3,048 (30.5) · One year 2,911 (29.1) |
| paperless_billing | Yes 6,336 (63.4) · No 3,664 (36.6) |
| payment_method | Electronic check 3,543 (35.4) · Bank transfer 2,407 (24.1) · Credit card 2,290 (22.9) · Mailed check 1,760 (17.6) |
| churn | No 7,178 (71.8) · Yes 2,822 (28.2) |

---

## 4. Overall churn rate

**28.22%** (2,822 of 10,000 customers). This is inside the 20–30% target.

## 5. Churn rate by contract type

| Contract | Customers | Churn rate |
|---|---:|---:|
| Month-to-month | 4,041 | 53.4% |
| One year | 2,911 | 16.5% |
| Two year | 3,048 | 6.1% |

## 6. Churn rate by tenure group

| Tenure (months) | Customers | Churn rate |
|---|---:|---:|
| 0–12 | 2,292 | 58.9% |
| 13–24 | 1,461 | 37.4% |
| 25–48 | 2,737 | 21.7% |
| 49–72 | 3,510 | 9.4% |

### Additional segment views

| Payment method | Customers | Churn rate |
|---|---:|---:|
| Electronic check | 3,543 | 39.0% |
| Mailed check | 1,760 | 23.8% |
| Credit card | 2,290 | 22.1% |
| Bank transfer | 2,407 | 21.4% |

| Internet service | Customers | Churn rate |
|---|---:|---:|
| Fiber optic | 4,501 | 37.9% |
| DSL | 3,588 | 25.4% |
| No | 1,911 | 10.6% |

| Tech support | Customers | Churn rate |
|---|---:|---:|
| No | 4,576 | 41.3% |
| Yes | 3,513 | 20.8% |
| No internet service | 1,911 | 10.6% |

Every segment contains both churned and retained customers. No feature value has a churn rate of 0% or 100%, and the script validates this.

---

## 7. Generation assumptions

The generator builds features in causal order (demographics → contract → tenure → services → billing → churn), so features are correlated with each other as well as with the target.

1. **Demographics.** Gender is 50/50. About 16% of customers are seniors. About 49% have a partner. Dependents are much more likely with a partner and less likely for seniors.
2. **Contract.** The probability of each contract type comes from a multinomial-logit draw. Customers with partners or dependents lean toward longer contracts, and seniors lean toward month-to-month.
3. **Tenure.** Tenure is drawn from a contract-specific Beta distribution scaled to 0–72. Month-to-month tenure is skewed short, one-year tenure is flatter, and two-year tenure is skewed long. The distributions overlap heavily. About 4% of customers have tenure 0.
4. **Phone and internet.** 90% of customers have phone service. Internet is DSL 34% / Fiber 44% / None 22% before adjustment. A customer can't have neither service, so any such customer is given internet. The probability of multiple lines rises with tenure.
5. **Add-ons.** Each add-on is a logistic draw. Adoption rises with tenure and contract length. Fiber customers stream more and buy security/support a little less. Streaming movies is correlated with streaming TV. Customers without internet always get "No internet service".
6. **Billing.** Paperless billing is more common for internet and fiber customers and less common for seniors. Electronic check is favoured by paperless, month-to-month and senior customers. Mailed check is mostly used by customers without paperless billing. Bank transfer and credit card are favoured on long contracts.
7. **Monthly charges** come from an additive price list plus Gaussian noise (σ = $2) for promotions and taxes, clipped to $18.25–$120. The price list is: phone $20, extra lines $5, DSL $25, Fiber $45, security/backup/protection/support $5 each, streaming TV and movies $10 each.
8. **Total charges** = tenure × monthly charges × a per-customer price-drift factor drawn from N(0.98, 0.05) and clipped to 0.85–1.10. The drift stands for past promotions and price changes, so the totals are close to tenure × monthly but not exactly equal. Total charges are 0 when tenure is 0.
9. **Churn.** Churn comes from a latent logistic model. The log-odds include:
   - contract (month-to-month +0.9, one-year −0.55, two-year −1.55)
   - −0.55 × log(1 + tenure), centred at 12 months
   - +0.018 per dollar of monthly charge above $65
   - Fiber +0.35 (a proxy for service-quality issues), no internet −0.30
   - no tech support +0.35, no online security +0.30
   - electronic check +0.45, paperless +0.15
   - senior +0.20, partner −0.15, dependents −0.15

   Each customer also gets a random effect drawn from N(0, 0.6) for unobserved factors. Churn is then a Bernoulli draw from the resulting probability. **Gender has no effect.** It is a deliberate null feature.
10. **Leakage avoidance.** The target is drawn last and nothing is derived from it. There are no post-churn fields such as a churn reason or cancellation date. `customer_id` is a random permutation and carries no information about churn. Note that `total_charges` is highly correlated with `tenure` by construction, which is realistic multicollinearity rather than leakage.
11. **No missing values** in this version. The real Telco dataset has blank `total_charges` for tenure-0 customers. Here they are 0.00 instead.

## 8. Programmatic validation (runs before the CSV is written)

The script exits with an error and writes no file if any of these checks fails:

- exactly 10,000 rows
- the expected columns, in order
- unique IDs
- no missing values
- only the allowed labels in every categorical column
- tenure between 0 and 72
- charges positive or non-negative
- total / (tenure × monthly) between 0.80 and 1.15, and total = 0 when tenure = 0
- consistency between phone service and lines, and between internet and add-ons, in both directions
- no customer without any service
- overall churn between 20% and 30%
- churn rates that differ across contract, tenure group and payment method
- no feature value with a churn rate of 0% or 100%
