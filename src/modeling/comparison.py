from __future__ import annotations

import pandas as pd
from sklearn.base import clone
from sklearn.metrics import average_precision_score, roc_auc_score

from src.modeling.evaluation import evaluate_test_predictions
from src.modeling.validation import (
    create_stratified_cv,
    generate_oof_predictions,
)


def evaluate_model_cv(
    model,
    X: pd.DataFrame,
    y: pd.Series,
    cv=None,
) -> dict:
    """Evaluate a model using stratified out-of-fold predictions."""

    if cv is None:
        cv = create_stratified_cv()

    oof_proba = generate_oof_predictions(
        model=model,
        X=X,
        y=y,
        cv=cv,
    )

    metrics = {
        "PR-AUC": average_precision_score(y, oof_proba),
        "ROC-AUC": roc_auc_score(y, oof_proba),
    }

    return {
        "oof_predictions": oof_proba,
        "metrics": metrics,
    }


def fit_and_evaluate_test(
    model,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    threshold: float,
) -> dict:
    """Fit a model on training data and evaluate it on held-out test data."""

    fitted_model = clone(model)
    fitted_model.fit(X_train, y_train)

    test_proba = fitted_model.predict_proba(X_test)[:, 1]

    test_metrics = evaluate_test_predictions(
        y_true=y_test,
        y_proba=test_proba,
        threshold=threshold,
    )

    return {
        "model": fitted_model,
        "test_probabilities": test_proba,
        "metrics": test_metrics,
    }