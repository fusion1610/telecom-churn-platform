from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.data.preprocessing import build_preprocessor
from src.modeling.calibration import build_calibrated_model


def test_build_calibrated_model_sigmoid():
    estimator = Pipeline(
        steps=[
            (
                "preprocessor",
                build_preprocessor(scale_numeric=True),
            ),
            (
                "classifier",
                LogisticRegression(max_iter=1000),
            ),
        ]
    )

    calibrated_model = build_calibrated_model(
        estimator=estimator,
        method="sigmoid",
        cv=3,
    )

    assert calibrated_model.method == "sigmoid"
    assert calibrated_model.cv == 3


def test_build_calibrated_model_isotonic():
    estimator = Pipeline(
        steps=[
            (
                "preprocessor",
                build_preprocessor(scale_numeric=True),
            ),
            (
                "classifier",
                LogisticRegression(max_iter=1000),
            ),
        ]
    )

    calibrated_model = build_calibrated_model(
        estimator=estimator,
        method="isotonic",
        cv=3,
    )

    assert calibrated_model.method == "isotonic"


def test_build_calibrated_model_rejects_invalid_method():
    estimator = LogisticRegression(max_iter=1000)

    try:
        build_calibrated_model(
            estimator=estimator,
            method="invalid",
        )
    except ValueError:
        return

    raise AssertionError("Expected ValueError for invalid calibration method.")