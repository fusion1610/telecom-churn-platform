# Business Risk & Revenue Prioritization

## Overview

The business-risk layer converts model churn probabilities into operational
risk and revenue-exposure signals for retention prioritization.

The business layer is applied after the final calibrated churn model produces
customer-level churn probabilities.

The primary outputs are:

- Churn Probability
- Revenue at Risk
- Retention Flag
- Risk Band
- Priority Tier
- Retention Action
- Retention Urgency

### Important terminology

Revenue at Risk is defined as:

Revenue at Risk = Churn Probability × TotalRevenue

This represents modeled revenue exposure associated with predicted churn risk.
It is not realized revenue loss, predicted savings, or revenue that will
necessarily be lost.

The dataset does not provide a reliable revenue time period, so TotalRevenue
is not described as monthly, annual, or lifetime revenue.

---

## Observation Lineage

### Data-lineage issue identified and repaired

The train/test split intentionally resets the pandas index:

```python
X_train.reset_index(drop=True)
X_test.reset_index(drop=True)
y_train.reset_index(drop=True)
y_test.reset_index(drop=True)