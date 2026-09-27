from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

RANDOM_STATE = 42
N_SPLITS = 5

def create_stratified_cv(n_splits: int = N_SPLITS, random_state: int = RANDOM_STATE) -> StratifiedKFold:
    """
    Create a reproducible stratified cross-validation splitter.
    """

    return StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state,
    )

def generate_oof_predictions(model, X: pd.DataFrame, y: pd.DataFrame, cv: StratifiedKFold | None = None) -> np.ndarray:
    """
    Generate out-of-fold predicted probabilities.

    Each observation receives a probability from a model that was not
    trained on that observation.
    """

    if cv is None:
        cv = create_stratified_cv()

    oof_predictions = np.zeros(len(X), dtype=float)

    for train_idx, validation_idx in cv.split(X, y):
        X_train = X.iloc[train_idx]
        X_validation = X.iloc[validation_idx]

        y_train = y.iloc[train_idx]

        fold_model = clone(model)

        fold_model.fit(X_train, y_train)

        oof_predictions[validation_idx] = (
            fold_model.predict_proba(X_validation)[:, 1]
        )

    return oof_predictions

def evaluate_probabilities(y_true: pd.Series, y_proba: pd.ndarry) -> pd.Series:
    """
    Evaluate probability predictions using ranking metrics.
    """

    return pd.Series(
        {
            "PR-AUC": average_precision_score(y_true, y_proba),
            "ROC-AUC": roc_auc_score(y_true, y_proba),
        }
    )