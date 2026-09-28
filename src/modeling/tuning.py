from __future__ import annotations

import pandas as pd

from sklearn.model_selection import GridSearchCV

from src.modeling.validation import create_stratified_cv


SCORING = "average_precision"


def tune_model(
    model,
    param_grid: dict,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cv=None,
) -> GridSearchCV:
    """Tune a model using stratified CV and PR-AUC."""

    if cv is None:
        cv = create_stratified_cv()

    search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        scoring=SCORING,
        cv=cv,
        refit=True,
        n_jobs=-1,
        return_train_score=False,
    )

    search.fit(X_train, y_train)

    return search