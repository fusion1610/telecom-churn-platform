import numpy as np
import pandas as pd

from src.data.preprocessing import build_preprocessor
from src.modeling.comparison import (
    evaluate_model_cv,
    fit_and_evaluate_test,
)
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression


def make_test_data(n_rows=20):
    X = pd.DataFrame(
        {
            "Active_subscribers": np.arange(1, n_rows + 1),
            "Not_Active_subscribers": np.ones(n_rows),
            "Suspended_subscribers": np.zeros(n_rows),
            "Total_SUBs": np.arange(2, n_rows + 2),
            "AvgMobileRevenue": np.linspace(50, 100, n_rows),
            "AvgFIXRevenue": np.linspace(1, 5, n_rows),
            "ARPU": np.linspace(20, 30, n_rows),
            "CRM_PID_Value_Segment": ["Gold", "Silver"] * (n_rows // 2),
            "EffectiveSegment": ["SME", "SOHO"] * (n_rows // 2),
        }
    )

    y = pd.Series(
        [0, 1] * (n_rows // 2),
        name="CHURN",
    )

    return X, y


def make_model():
    return Pipeline(
        steps=[
            (
                "preprocessor",
                build_preprocessor(scale_numeric=True),
            ),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1000,
                    random_state=42,
                ),
            ),
        ]
    )


def test_evaluate_model_cv_returns_oof_predictions_and_metrics():
    X, y = make_test_data()
    model = make_model()

    result = evaluate_model_cv(
        model=model,
        X=X,
        y=y,
    )

    assert "oof_predictions" in result
    assert "metrics" in result

    assert len(result["oof_predictions"]) == len(y)

    assert "PR-AUC" in result["metrics"]
    assert "ROC-AUC" in result["metrics"]

    assert np.all(
        (result["oof_predictions"] >= 0)
        & (result["oof_predictions"] <= 1)
    )


def test_fit_and_evaluate_test_returns_fitted_model_and_metrics():
    X, y = make_test_data()

    X_train = X.iloc[:16].reset_index(drop=True)
    y_train = y.iloc[:16].reset_index(drop=True)

    X_test = X.iloc[16:].reset_index(drop=True)
    y_test = y.iloc[16:].reset_index(drop=True)

    model = make_model()

    result = fit_and_evaluate_test(
        model=model,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        threshold=0.06,
    )

    assert "model" in result
    assert "test_probabilities" in result
    assert "metrics" in result

    assert len(result["test_probabilities"]) == len(y_test)

    assert "PR-AUC" in result["metrics"]
    assert "ROC-AUC" in result["metrics"]
    assert "Precision" in result["metrics"]
    assert "Recall" in result["metrics"]
    assert "F1" in result["metrics"]