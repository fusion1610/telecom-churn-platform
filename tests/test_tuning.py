import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.data.preprocessing import build_preprocessor
from src.modeling.tuning import tune_model


def make_test_data(n_rows: int = 40):
    X = pd.DataFrame(
        {
            "CRM_PID_Value_Segment": ["Gold", "Silver"] * (n_rows // 2),
            "EffectiveSegment": ["SME", "SOHO"] * (n_rows // 2),
            "Active_subscribers": [5, 10] * (n_rows // 2),
            "Not_Active_subscribers": [2, 4] * (n_rows // 2),
            "Suspended_subscribers": [1, 0] * (n_rows // 2),
            "Total_SUBs": [8, 14] * (n_rows // 2),
            "AvgMobileRevenue": [100.0, 200.0] * (n_rows // 2),
            "AvgFIXRevenue": [5.0, 10.0] * (n_rows // 2),
            "ARPU": [12.5, 15.0] * (n_rows // 2),
        }
    )

    y = pd.Series(
        [0, 1] * (n_rows // 2),
        name="CHURN",
    )

    return X, y


def test_tune_model_returns_fitted_search():
    X, y = make_test_data()

    model = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(scale_numeric=True)),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1000,
                    random_state=42,
                ),
            ),
        ]
    )

    search = tune_model(
        model=model,
        param_grid={
            "classifier__C": [0.1, 1.0],
        },
        X_train=X,
        y_train=y,
    )

    assert search.best_estimator_ is not None
    assert search.best_params_["classifier__C"] in [0.1, 1.0]
    assert search.best_score_ >= 0.0
    assert len(search.cv_results_["mean_test_score"]) == 2