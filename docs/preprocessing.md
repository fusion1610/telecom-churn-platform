# Milestone 2 — Preprocessing Contract

## 1. Purpose

This document defines the reproducible preprocessing rules for the telecom churn prediction project.

The preprocessing pipeline must transform the immutable raw dataset into a model-ready feature matrix without modifying the raw data.

The same fitted preprocessing pipeline must be used during:

* model training
* validation
* testing
* batch inference
* API inference
* production deployment

All transformations that learn parameters from data must be fitted on the training data only.

---

## 2. Raw Data Boundary

Raw data is stored at:

```text
data/raw/telecom_churn.csv
```

The raw dataset must remain unchanged.

Preprocessing operates on an in-memory copy or generated processed artifact.

No manual edits should be made to the raw CSV.

---

## 3. Target Encoding

Raw target:

```text
CHURN
```

Raw values:

```text
Yes
No
```

Model target encoding:

```text
Yes -> 1
No  -> 0
```

The target is not passed through the feature preprocessing pipeline.

The target is separated from the feature matrix before feature transformations are fitted.

---

## 4. Identifier Handling

### `PID`

Treatment:

```text
Exclude
```

Reason:

`PID` behaves as an identifier rather than a meaningful predictive feature.

The dataset also contains repeated PID values with conflicting business attributes.

`PID` must therefore not be supplied to the model.

---

## 5. Categorical Features

The initial categorical feature set is:

```text
CRM_PID_Value_Segment
EffectiveSegment
```

### `CRM_PID_Value_Segment`

Treatment:

* Keep as categorical.
* Preserve missing values until preprocessing.
* Represent missing values explicitly as `"Missing"`.
* Encode using the categorical preprocessing pipeline.
* Preserve the observed `Sliver` category.

### `EffectiveSegment`

Treatment:

* Keep as categorical.
* No missing values were observed.
* Encode using the categorical preprocessing pipeline.

---

## 6. Excluded Categorical Features

### `Billing_ZIP`

Treatment:

```text
Exclude from baseline model
```

Reason:

* 456 unique values
* sparse categories
* two missing values
* ZIP is geographic/account-location information rather than a naturally continuous numeric variable
* sparse categories can produce unstable category-level patterns

It may be investigated in future experiments but is not part of the baseline feature set.

### `KA_name`

Treatment:

```text
Exclude from baseline model
```

Reason:

`KA_name` may represent account-management assignment rather than an intrinsic customer characteristic.

It remains available for exploratory analysis but is excluded from the initial predictive model.

---

## 7. Numeric Features

The initial numeric feature set is:

```text
Active_subscribers
Not_Active_subscribers
Suspended_subscribers
Total_SUBs
AvgMobileRevenue
AvgFIXRevenue
ARPU
```

---

## 8. Structural Missing Values

### `Not_Active_subscribers`

Observed missing values:

```text
4,149
```

Investigation established that the missing values can be reconstructed as zero using the subscriber relationship.

Treatment:

```text
Missing -> 0
```

### `Suspended_subscribers`

Observed missing values:

```text
8,101
```

Investigation established that the missing values can be reconstructed as zero using the subscriber relationship.

Treatment:

```text
Missing -> 0
```

These are treated as structural zeros rather than unknown values.

---

## 9. ARPU Missing Value

`ARPU` contains one missing value.

The corresponding record has:

```text
TotalRevenue = 40.17
Total_SUBs = 2
```

The reconstruction is:

```text
40.17 / 2 = 20.085
```

The preprocessing pipeline will reconstruct the missing value as:

```text
20.09
```

The raw CSV will not be modified.

Important:

`ARPU` will otherwise remain as the supplied dataset feature because `ARPU` does not exactly reproduce from `TotalRevenue / Total_SUBs` throughout the dataset.

---

## 10. Revenue Feature Policy

The raw dataset satisfies:

```text
TotalRevenue =
    AvgMobileRevenue + AvgFIXRevenue
```

for all observations, apart from negligible floating-point representation error.

Therefore:

```text
TotalRevenue
```

is excluded from the baseline feature matrix.

The baseline retains:

```text
AvgMobileRevenue
AvgFIXRevenue
```

This avoids feeding a deterministic duplicate of existing features into the baseline model.

---

## 11. Final Baseline Feature Set

### Numeric

```text
Active_subscribers
Not_Active_subscribers
Suspended_subscribers
Total_SUBs
AvgMobileRevenue
AvgFIXRevenue
ARPU
```

### Categorical

```text
CRM_PID_Value_Segment
EffectiveSegment
```

### Excluded

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

---

## 12. Numeric Transformation

The baseline preprocessing pipeline will:

1. Select the defined numeric features.
2. Apply structural-zero treatment where required.
3. Reconstruct the missing ARPU value.
4. Apply numeric imputation only if unexpected missing values remain.
5. Apply feature scaling where required by the selected estimator.

For the initial Logistic Regression model, numeric features will be standardized.

For tree-based models, scaling is not mathematically required, but the preprocessing architecture should remain reproducible and estimator-specific where appropriate.

---

## 13. Categorical Transformation

The baseline categorical preprocessing pipeline will:

1. Select the defined categorical features.
2. Replace missing categorical values with `"Missing"`.
3. Encode categorical values using an encoder fitted only on the training data.
4. Handle previously unseen categories during inference without failing.

The pipeline must therefore support production inference on records containing categorical values not observed during model training.

---

## 14. Train/Test Leakage Prevention

The complete preprocessing pipeline must be fitted only on the training split.

The correct sequence is:

```text
Raw dataset
    |
    v
Separate target
    |
    v
Train/Test split
    |
    +----> Training data ----> Fit preprocessing
    |                              |
    |                              v
    |                         Transform train
    |
    +----> Test data --------> Transform using
                               fitted preprocessing
```

The test set must never be used to calculate:

* imputation values
* category mappings
* scaling parameters
* feature-selection statistics
* target-dependent statistics

---

## 15. Target Leakage Prevention

No feature may use `CHURN` directly or indirectly.

Examples of prohibited transformations include:

* target encoding calculated before train/test splitting
* churn rate calculated using the complete dataset
* features derived from post-churn events
* target-dependent aggregation performed before splitting

Any target-dependent feature engineering must be performed inside a training-only pipeline or cross-validation procedure.

---

## 16. Reproducibility

Preprocessing must be implemented in Python rather than manually performed in a spreadsheet.

The transformation should eventually be represented by a serialized machine-learning pipeline so that the exact same preprocessing can be applied during inference.

The preprocessing implementation should be deterministic given:

* the same raw dataset
* the same feature configuration
* the same random seed where randomness is involved
* the same software environment

---

## 17. Validation Requirements

Before training a baseline model, preprocessing validation must confirm:

1. Raw data is not modified.
2. Expected columns exist.
3. Target contains only `Yes` and `No`.
4. Structural subscriber missing values are converted to zero.
5. The missing ARPU value is reconstructed correctly.
6. No unexpected missing values remain in the model matrix.
7. Categorical values are encoded successfully.
8. Train and test transformations use the same fitted preprocessing object.
9. Test data does not influence preprocessing fitting.
10. The resulting feature matrix contains only model-approved features.

---

## 18. Future Extensions

The baseline preprocessing contract may be extended after empirical evaluation.

Possible future experiments include:

* alternative subscriber feature representations
* revenue-derived features
* interaction features
* categorical grouping
* alternative encoding strategies
* feature selection
* nonlinear transformations
* model-specific preprocessing
* additional business-value features

Any change must be documented as a new experiment rather than silently replacing the baseline.

---

## 19. Baseline Preprocessing Contract

The initial model-ready feature contract is therefore:

```text
Categorical:
    CRM_PID_Value_Segment
    EffectiveSegment

Numeric:
    Active_subscribers
    Not_Active_subscribers
    Suspended_subscribers
    Total_SUBs
    AvgMobileRevenue
    AvgFIXRevenue
    ARPU

Target:
    CHURN
```

Excluded from baseline:

```text
PID
Billing_ZIP
KA_name
TotalRevenue
```

This contract is the starting point for Milestone 2 preprocessing implementation.
