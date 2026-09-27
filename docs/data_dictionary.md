# Telecom Churn Dataset — Data Dictionary

## Purpose

This document defines the observed schema of the raw telecom churn dataset and records
the current interpretation and modeling treatment of each field.

The raw dataset is preserved unchanged in:

`data/raw/telecom_churn.csv`

Observed raw dimensions:

- Rows: 8,453
- Columns: 14

Target:

- `CHURN`

Important:
Business meanings below are based on the field names, observed values, dataset
documentation, and completed data-quality analysis. Where the source does not explicitly
define a field, the interpretation is marked as provisional rather than presented as fact.

---

## Field Reference

| Field | Raw Type | Role | Missing | Description / Interpretation | Modeling Treatment |
|---|---|---|---:|---|---|
| `PID` | Identifier-like | Identifier | 0 | Customer/account identifier present in the source data. Repeated PID values exist, so it is not a reliable unique customer key. | Exclude |
| `CRM_PID_Value_Segment` | Categorical | Feature | 5 | CRM/customer value segmentation field. Observed categories include SME, Platinum, Gold, Silver, Bronze, Iron, SE, Lead, and `Sliver`. | Retain; categorical imputation |
| `EffectiveSegment` | Categorical | Feature | 0 | Customer segmentation field. Observed categories are VSE, SME, Other, SOHO, SE, and LE. | Retain; categorical |
| `Billing_ZIP` | Numeric in raw CSV | Feature | 2 | Billing ZIP/postal-code field. Although numeric in the CSV, it represents a categorical geographic code rather than a continuous numeric measurement. | Exclude from initial model |
| `KA_name` | Categorical | Feature | 0 | Account-management/KA assignment field. The source does not explicitly define the abbreviation `KA`. | Exclude initially; retain for EDA |
| `Active_subscribers` | Numeric | Feature | 0 | Number of active subscribers associated with the record. | Retain |
| `Not_Active_subscribers` | Numeric | Feature | 4,149 | Number of non-active subscribers. Missing values are structurally recoverable as zero from the subscriber-total relationship. | Impute structural missing values to 0 |
| `Suspended_subscribers` | Numeric | Feature | 8,101 | Number of suspended subscribers. Missing values are structurally recoverable as zero from the subscriber-total relationship. | Impute structural missing values to 0 |
| `Total_SUBs` | Numeric | Feature | 0 | Total subscriber count. Verified to equal the sum of active, non-active, and suspended subscribers. | Retain initially |
| `AvgMobileRevenue` | Numeric | Feature | 0 | Average mobile revenue measure as supplied by the dataset. | Retain |
| `AvgFIXRevenue` | Numeric | Feature | 0 | Average fixed-service revenue measure as supplied by the dataset. | Retain |
| `TotalRevenue` | Numeric | Feature | 0 | Total revenue measure. Verified to equal `AvgMobileRevenue + AvgFIXRevenue` within floating-point precision for every row. | Exclude from initial baseline because it is deterministic |
| `ARPU` | Numeric | Feature | 1 | ARPU measure supplied by the dataset. The exact business definition is not explicitly established by the available documentation. | Retain; reconstruct missing value during preprocessing |
| `CHURN` | Categorical | Target | 0 | Customer churn outcome. Raw values are `Yes` and `No`. | Encode as binary target during modeling |

---

## Target Definition

### `CHURN`

Observed values:

- `No`: 7,904 rows (93.51%)
- `Yes`: 549 rows (6.49%)

There are no missing or unexpected target values.

The target is therefore a binary classification problem with substantial class
imbalance.

Model evaluation should prioritize:

- PR-AUC
- Recall
- Precision
- F1
- ROC-AUC
- Probability calibration

Accuracy should not be used as the primary model-selection metric.

---

## Identifier Policy

`PID` is retained in the raw dataset for traceability but is not used as a predictive
feature.

The dataset contains:

- 8,453 rows
- 8,436 unique PID values
- 6 repeated PID values
- 23 rows belonging to repeated PID groups

Some repeated PID groups have conflicting churn labels and other differing attributes.

Therefore, `PID` must not be treated as a unique customer key during modeling.

---

## Missing-Value Policy

### Structural subscriber fields

`Not_Active_subscribers` and `Suspended_subscribers` contain substantial missingness.

Data-quality analysis established that the subscriber relationship is satisfied when
missing subscriber components are treated as zero:

`Total_SUBs = Active_subscribers + Not_Active_subscribers + Suspended_subscribers`

All 8,453 rows satisfy this relationship.

Therefore:

- Missing `Not_Active_subscribers` → 0
- Missing `Suspended_subscribers` → 0

This transformation will occur during preprocessing, not by modifying the raw CSV.

### `CRM_PID_Value_Segment`

Five values are missing.

The missing values are not inferred from other fields during raw-data preparation.
They will be handled as an explicit categorical missing value during preprocessing.

### `Billing_ZIP`

Two values are missing.

Because `Billing_ZIP` is excluded from the initial model, no initial predictive
imputation is required.

### `ARPU`

One value is missing.

The record has:

- `TotalRevenue = 40.17`
- `Total_SUBs = 2`

A reconstructed value based on `TotalRevenue / Total_SUBs` is approximately `20.09`
when represented to two decimal places.

This is a preprocessing reconstruction, not a modification of the raw dataset.

---

## Feature Redundancy

Several deterministic relationships were identified.

### Subscriber relationship

```text
Total_SUBs =
    Active_subscribers
    + Not_Active_subscribers
    + Suspended_subscribers