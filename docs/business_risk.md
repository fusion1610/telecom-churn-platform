# Business Risk & Revenue Layer

## Checkpoint 41 — Observation-Level Risk & Revenue

The final calibrated HistGradientBoosting model produces a churn
probability for each held-out test observation.

An observation-level revenue-at-risk proxy is calculated as:

Revenue at Risk = Churn Probability × TotalRevenue

The operating threshold selected from out-of-fold training predictions
is 0.07.

### Held-out test results

- Test observations: 1,691
- Retention-flagged observations: 431
- Flagged coverage: 25.49%
- Total observed revenue: 116,713.13
- Total modeled revenue at risk: 7,599.40
- Revenue in flagged observations: 29,914.84
- Modeled revenue at risk in flagged observations: 2,356.27

The thresholded observations therefore contain approximately 31.0% of
the total modeled revenue-at-risk.

Revenue at Risk is an expected-exposure proxy, not realized revenue
loss or revenue saved.

The available dataset does not establish the time period represented
by TotalRevenue. Therefore, the metric is not labeled as monthly
revenue, annual revenue, ARR, or another time-specific financial
measure.

The current output is intentionally observation-level because PID was
previously found not to be a guaranteed unique identifier. Customer-
level aggregation requires an explicit policy for repeated PID
observations.

## Risk Bands & Operating Segments

Risk bands were introduced to convert continuous churn probabilities into operational customer segments.

### Risk Band Definitions

| Risk Band | Churn Probability |
|---|---:|
| Low | < 0.05 |
| Moderate | 0.05–<0.07 |
| High | >= 0.07 |

The 0.07 boundary corresponds to the frozen retention operating threshold selected using out-of-fold training predictions. The 0.05 boundary is a descriptive segmentation boundary and was not separately optimized.

### Held-Out Test Results

| Risk Band | Observations | Mean Churn Probability | Observed Churn Rate | Revenue at Risk | Revenue-at-Risk Share | Coverage |
|---|---:|---:|---:|---:|---:|---:|
| Low | 102 | 4.71% | 2.94% | 321.92 | 4.24% | 6.03% |
| Moderate | 1,158 | 6.15% | 4.75% | 4,921.21 | 64.76% | 68.48% |
| High | 431 | 7.88% | 3.25% | 2,356.27 | 31.01% | 25.49% |

### Interpretation

The risk bands provide an operational segmentation of the model's predicted churn probability.

The High-risk band contains 431 observations, representing 25.49% of the held-out test population. Because the frozen operating threshold is 0.07, these 431 observations are the current retention-flagged population.

Although the High-risk band represents 25.49% of observations, it contains 31.01% of the modeled revenue-at-risk. This indicates that the flagged population represents a disproportionately large share of the modeled risk exposure.

The observed churn rates are 2.94% for Low, 4.75% for Moderate, and 3.25% for High. Therefore, the held-out sample does not demonstrate a monotonic relationship between risk band and observed churn rate. This should not be interpreted as strong empirical separation between the bands.

### Business Caveat

Risk bands represent model predictions, not confirmed customer outcomes.

`Revenue at Risk = Churn Probability × TotalRevenue`

The resulting revenue-at-risk measure is an expected-exposure proxy rather than realized revenue loss. The dataset does not establish the revenue time period represented by `TotalRevenue`.

The current risk table is observation-level because `PID` is not a reliable unique identifier in this dataset.

Risk bands and operating segments have been implemented and evaluated on the untouched held-out test set.