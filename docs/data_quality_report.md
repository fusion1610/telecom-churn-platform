# Milestone 1 — Data Quality Report

## 1. Objective

This report consolidates the data-quality, schema, target, identifier, missing-value, and feature-relationship findings from Milestone 1.

The objective is to establish a reproducible understanding of the raw telecom churn dataset before feature engineering and model development.

---

## 2. Dataset Overview

| Property         | Value                                           |
| ---------------- | ----------------------------------------------- |
| Dataset          | Telecom Churn Dataset                           |
| Source           | Mendeley Data                                   |
| Source URL       | https://data.mendeley.com/datasets/nrb55gr66h/1 |
| License          | CC BY 4.0                                       |
| Local path       | `data/raw/telecom_churn.csv`                    |
| Observed rows    | 8,453                                           |
| Observed columns | 14                                              |
| Target           | `CHURN`                                         |
| Positive class   | `Yes`                                           |
| Negative class   | `No`                                            |
| Temporal field   | None                                            |

The publisher documentation describes 8,454 instances, while the locally observed CSV contains 8,453 rows. The local file is treated as the reproducible modeling input, and this discrepancy is explicitly recorded rather than silently corrected.

The raw CSV is immutable and must not be manually edited.

---

## 3. Reproducibility

### Raw File SHA-256

```text
36509d631e29a70e4a6e01c0deef42f47d35c112ea29d94b21b67b0a2e26c523
```

This hash identifies the exact raw CSV used during Milestone 1.

Any future change to the raw file should produce a different hash and should be treated as a new dataset version.

---

## 4. Schema Validation

The normalized schema contains 14 columns:

```text
PID
CRM_PID_Value_Segment
EffectiveSegment
Billing_ZIP
KA_name
Active_subscribers
Not_Active_subscribers
Suspended_subscribers
Total_SUBs
AvgMobileRevenue
AvgFIXRevenue
TotalRevenue
ARPU
CHURN
```

Schema validation completed successfully.

* Observed rows: 8,453
* Observed columns: 14
* Missing required columns: none
* Unexpected target values: none

The target is represented by the strings `Yes` and `No` in the raw data.

---

## 5. Missing-Value Assessment

| Column                   | Missing rows | Missing % | Treatment                                             |
| ------------------------ | -----------: | --------: | ----------------------------------------------------- |
| `Suspended_subscribers`  |        8,101 |    95.84% | Treat missing as structural zero during preprocessing |
| `Not_Active_subscribers` |        4,149 |    49.08% | Treat missing as structural zero during preprocessing |
| `CRM_PID_Value_Segment`  |            5 |     0.06% | Preserve missing; categorical imputation later        |
| `Billing_ZIP`            |            2 |     0.02% | Preserve missing; excluded from baseline model        |
| `ARPU`                   |            1 |     0.01% | Reconstruct programmatically during preprocessing     |
| All other columns        |            0 |        0% | No missing-value treatment required                   |

### Subscriber-Field Investigation

The following relationship holds for all 8,453 rows when missing subscriber fields are treated as zero:

```text
Total_SUBs =
    Active_subscribers
    + Not_Active_subscribers
    + Suspended_subscribers
```

There were zero mismatches.

For the 4,149 missing `Not_Active_subscribers` values, the value reconstructed from the subscriber relationship was zero for every row.

For the 8,101 missing `Suspended_subscribers` values, the reconstructed value was zero for every row.

No negative reconstructed values were found.

Therefore, these missing values are treated as structural zeros rather than conventional unknown or missing measurements.

### Remaining Missing Values

The five missing `CRM_PID_Value_Segment` values occur in records whose `EffectiveSegment` is `Other`. All five have `CHURN = No`.

The two missing `Billing_ZIP` values occur in `EffectiveSegment = SOHO` records. Both have `CHURN = No`.

One `ARPU` value is missing. Its corresponding record has:

```text
TotalRevenue = 40.17
Total_SUBs = 2
```

Therefore:

```text
40.17 / 2 = 20.085
```

Since ARPU is represented to two decimal places, preprocessing will reconstruct this value as `20.09`.

The raw CSV will not be modified.

---

## 6. Duplicate and Identifier Assessment

### Exact Duplicate Rows

Exact duplicate rows:

```text
0
```

### Repeated PID Values

The dataset contains:

* 8,436 unique PID values
* 6 repeated PID values
* 23 rows belonging to repeated PID groups
* 0.2721% of all rows belong to repeated PID groups

Repeated PID frequencies:

* 1 PID appears twice
* 1 PID appears three times
* 2 PIDs appear four times
* 2 PIDs appear five times

The repeated PID records are not identical when excluding the PID field.

Several repeated PID groups also contain conflicting values in business fields:

| Field                   | Repeated PIDs with conflicts |
| ----------------------- | ---------------------------: |
| `CHURN`                 |                            3 |
| `CRM_PID_Value_Segment` |                            6 |
| `EffectiveSegment`      |                            5 |
| `KA_name`               |                            6 |

### Identifier Policy

`PID` will not be used as a predictive feature.

Repeated PID records will be retained as separate observations rather than being dropped, aggregated, or reduced to first/last records.

This reflects the finding that PID cannot safely be treated as a unique customer key in this dataset.

---

## 7. Target Validation

The target column is `CHURN`.

### Target Quality

* Missing values: 0
* Unique values: `No`, `Yes`
* Unexpected values: none

### Target Distribution

| Target |  Rows | Percentage |
| ------ | ----: | ---------: |
| `No`   | 7,904 |   93.5053% |
| `Yes`  |   549 |    6.4947% |

The target is strongly imbalanced, with churn representing approximately 6.49% of observations.

Therefore, model evaluation will prioritize metrics that provide useful information for the minority class, including:

* PR-AUC
* Recall
* Precision
* F1
* ROC-AUC
* Calibration

Accuracy will not be treated as the primary model-selection metric.

Classification threshold selection will be treated as a business decision rather than automatically using a 0.50 threshold.

---

## 8. Feature Relationship Findings

### Deterministic Revenue Relationship

The following relationship holds for every row:

```text
TotalRevenue =
    AvgMobileRevenue + AvgFIXRevenue
```

Maximum floating-point difference:

```text
5.684341886080802e-14
```

Because `TotalRevenue` is deterministically derived from the two component revenue fields, it will be excluded from the baseline model.

### ARPU Relationship

`ARPU` does not exactly equal:

```text
TotalRevenue / Total_SUBs
```

throughout the dataset.

Observed differences:

* Rows with absolute difference greater than 0.01: 4,487
* Maximum absolute difference: 414.634146
* Mean absolute difference: 5.410935

Therefore, `ARPU` will be retained as an independently supplied feature rather than being replaced by a calculated version.

### Numeric Feature Relationships

Notable correlations include:

| Feature pair                              | Correlation |
| ----------------------------------------- | ----------: |
| `Not_Active_subscribers` / `Total_SUBs`   |       0.807 |
| `Active_subscribers` / `Total_SUBs`       |       0.722 |
| `AvgMobileRevenue` / `TotalRevenue`       |       0.994 |
| `Active_subscribers` / `AvgMobileRevenue` |       0.701 |
| `TotalRevenue` / `Total_SUBs`             |       0.553 |

The correlations indicate meaningful redundancy among subscriber and revenue variables.

The baseline model will retain the relevant subscriber variables initially so that alternative feature sets can later be evaluated empirically.

---

## 9. Categorical Feature Assessment

### `CRM_PID_Value_Segment`

* 9 observed categories
* 5 missing values
* Retain as a categorical feature
* Missing values will be handled during preprocessing
* `Sliver` appears as a one-row category and is preserved as observed

### `EffectiveSegment`

* 6 observed categories
* No missing values
* Retain as a categorical feature

### `Billing_ZIP`

* 456 unique values
* 2 missing values
* Many categories have very small sample sizes
* Sparse categories produce unstable apparent churn rates

`Billing_ZIP` will therefore be excluded from the initial model.

If investigated later, it must be treated as a categorical variable rather than as a continuous numeric measurement.

### `KA_name`

* 12 unique values
* No missing values

`KA_name` may reflect account-management assignment rather than an intrinsic customer characteristic.

It will therefore be excluded from the initial predictive model while remaining available for exploratory analysis.

---

## 10. Initial Feature Policy

### Retain Initially

```text
CRM_PID_Value_Segment
EffectiveSegment
Active_subscribers
Not_Active_subscribers
Suspended_subscribers
Total_SUBs
AvgMobileRevenue
AvgFIXRevenue
ARPU
```

### Exclude from Baseline

```text
PID
Billing_ZIP
KA_name
TotalRevenue
```

### Target

```text
CHURN
```

The feature policy is an initial modeling decision and may be revised after validation experiments.

---

## 11. Data-Quality Issues to Preserve and Document

The following observations are considered part of the dataset's data-quality profile:

1. Publisher-reported row count differs from the locally observed row count.
2. `PID` is not reliably unique.
3. `Not_Active_subscribers` contains substantial structural missingness.
4. `Suspended_subscribers` contains substantial structural missingness.
5. `CRM_PID_Value_Segment` contains five missing values.
6. `Billing_ZIP` contains two missing values.
7. One `ARPU` value is missing.
8. `ARPU` does not exactly reproduce from `TotalRevenue / Total_SUBs`.
9. `TotalRevenue` is deterministically related to the two revenue components.
10. `Billing_ZIP` has high cardinality and sparse categories.
11. `Sliver` is an observed category spelling anomaly.
12. The dataset has no temporal field.

These issues will be handled explicitly rather than silently corrected.

---

## 12. Leakage Assessment

No obvious direct target leakage was identified during Milestone 1.

However, feature engineering must not use target information calculated from the complete dataset.

Any target-dependent transformation, encoding, aggregation, or statistical calculation must be fitted using the training data only.

This rule will be enforced during the modeling pipeline.

---

## 13. Modeling Implications

The dataset is suitable for proceeding to feature engineering and baseline model development, subject to the documented limitations.

Key modeling considerations:

* Strong class imbalance requires appropriate evaluation metrics.
* `PID` must not be used as a predictive feature.
* High-cardinality `Billing_ZIP` is excluded from the baseline.
* `KA_name` is excluded from the baseline because it may represent account-management assignment.
* Structural subscriber missingness will be converted to zero during preprocessing.
* Categorical missing values will be handled within the preprocessing pipeline.
* `ARPU` will be reconstructed only for its missing observation.
* `TotalRevenue` is excluded because it is deterministically derived from mobile and fixed revenue.
* The lack of a temporal field prevents chronological train/test evaluation.
* A stratified train/test split will therefore be used initially.
* Any later temporal modeling would require a dataset containing suitable time information.

---

## 14. Milestone 1 Conclusion

Milestone 1 established the structure, provenance, reproducibility, missing-value behavior, identifier limitations, target distribution, feature relationships, and initial feature policy for the telecom churn dataset.

The raw dataset remains unchanged.

The next milestone can proceed to reproducible preprocessing and feature engineering while preserving the raw-data boundary established during Milestone 1.
