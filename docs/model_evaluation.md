## Candidate Model

The HistGradientBoosting family was selected for continued evaluation based on
cross-validation performance, probability calibration, and held-out
generalization evidence.

The tuned HistGradientBoosting configuration achieved:

- CV PR-AUC: 0.091899
- OOF Brier score: 0.060758
- Test PR-AUC: 0.087748
- Test ROC-AUC: 0.581634
- Test F1: 0.137441

The untuned HistGradientBoosting model remains an important benchmark:

- CV PR-AUC: 0.079645
- OOF Brier score: 0.065957
- Test PR-AUC: 0.089013
- Test ROC-AUC: 0.592640
- Test F1: 0.157687

The held-out test set is treated as a generalization assessment rather than
as a repeated tuning target.

Random Forest showed substantially weaker probability calibration, while
Logistic Regression produced competitive calibration but lower held-out
discrimination in this evaluation.

## Final Candidate Model

The final candidate carried forward is a tuned
HistGradientBoostingClassifier with sigmoid probability calibration.

### Final Configuration

- Model: HistGradientBoostingClassifier
- Hyperparameter tuning metric: Average Precision (PR-AUC)
- Calibration: Sigmoid
- Calibration CV: 5-fold
- Operating threshold: 0.07
- Threshold selection: OOF training predictions
- Final evaluation: untouched held-out test set
- Random state: 42

### Final Held-Out Test Results

| Metric | Result |
|---|---:|
| PR-AUC | 0.086978 |
| ROC-AUC | 0.578495 |
| Precision | 0.085847 |
| Recall | 0.336364 |
| F1 | 0.136784 |
| Customers Flagged | 431 |
| Test Customers | 1691 |
| Coverage | 0.254879 |

### Confusion Matrix

| | Predicted No Churn | Predicted Churn |
|---|---:|---:|
| Actual No Churn | 1187 | 394 |
| Actual Churn | 73 | 37 |

### Interpretation

At the frozen threshold of 0.07, the model flags approximately
25.5% of test customers and identifies approximately 33.6% of
observed churners.

The model provides a modest ranking signal rather than highly
accurate individual churn prediction. Therefore, it is intended
to support customer retention prioritization rather than make
fully automated retention decisions.

The 0.07 threshold was selected using out-of-fold predictions
from the training data and was not optimized using the held-out
test set.

The held-out test set was used only for final generalization
assessment.