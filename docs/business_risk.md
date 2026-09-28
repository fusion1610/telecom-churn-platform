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