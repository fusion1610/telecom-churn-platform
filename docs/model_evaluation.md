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