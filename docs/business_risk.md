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

## Revenue-at-Risk Prioritization

Revenue-at-risk prioritization extends the churn-risk framework by combining predicted churn probability with observed revenue exposure.

### Prioritization Formula

Revenue at Risk is defined as:

`Revenue at Risk = Churn Probability × TotalRevenue`

Observations were ranked in descending order of Revenue at Risk.

Because the distribution of Revenue at Risk is dataset-specific, percentile-based prioritization was used instead of arbitrary fixed revenue thresholds.

### Priority Tier Definitions

| Priority Tier | Revenue-at-Risk Boundary |
|---|---:|
| Standard | <= 75th percentile |
| Priority | > 75th and <= 90th percentile |
| Critical | > 90th percentile |

For the held-out test population:

| Priority Tier | Observations | Mean Churn Probability | Mean Revenue at Risk | Total Revenue at Risk |
|---|---:|---:|---:|---:|
| Standard | 1,268 | 6.22% | 3.907 | 4,954.46 |
| Priority | 254 | 6.95% | 5.781 | 1,468.39 |
| Critical | 169 | 8.01% | 6.962 | 1,176.55 |

### Interpretation

The prioritization tiers show increasing mean churn probability and increasing mean revenue-at-risk from Standard to Critical.

The Critical tier contains 169 observations, approximately 10% of the held-out population, while accounting for approximately 15.48% of total modeled revenue-at-risk.

Priority and Critical observations together contain 423 observations, approximately 25% of the held-out population, and approximately 34.8% of total modeled revenue-at-risk.

The resulting ranking is intended to support retention prioritization. It does not identify customers who will definitely churn and does not represent realized revenue loss.

### Relationship to Risk Bands

Risk Band and Priority Tier represent different concepts:

- Risk Band describes predicted churn probability.
- Priority Tier describes modeled revenue exposure.
- Revenue at Risk combines the two through the formula above.

The current prioritization remains observation-level because PID was previously found not to be a reliably unique identifier.

### Business Caveats

Revenue at Risk is an expected-exposure proxy:

`Churn Probability × TotalRevenue`

It should not be interpreted as confirmed future revenue loss or revenue that can necessarily be saved through intervention.

The dataset does not establish the time period represented by TotalRevenue.

Revenue-at-risk observations have been ranked and segmented into percentile-based operational priority tiers.