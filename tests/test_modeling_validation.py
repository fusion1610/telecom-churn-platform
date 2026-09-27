import numpy as np

from src.data.preprocessing import (
    build_preprocessor,
    load_raw_data,
    prepare_dataset,
)
from src.data.split import create_train_test_split
from src.modeling.validation import (
    create_stratified_cv,
    evaluate_probabilities,
    generate_oof_predictions,
)
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


def test_stratified_cv_is_reproducible():
    cv_1 = create_stratified_cv()
    cv_2 = create_stratified_cv()

    assert cv_1.n_splits == 5
    assert cv_1.shuffle is True
    assert cv_1.random_state == cv_2.random_state


def test_oof_predictions_have_correct_length():
    df = load_raw_data()
    X, y = prepare_dataset(df)

    X_train, _, y_train, _ = create_train_test_split(X, y)

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

    cv = create_stratified_cv()

    oof_predictions = generate_oof_predictions(
        model=model,
        X=X_train,
        y=y_train,
        cv=cv,
    )

    assert len(oof_predictions) == len(X_train)
    assert np.all(oof_predictions >= 0)
    assert np.all(oof_predictions <= 1)


def test_oof_predictions_are_reproducible():
    df = load_raw_data()
    X, y = prepare_dataset(df)

    X_train, _, y_train, _ = create_train_test_split(X, y)

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

    predictions_1 = generate_oof_predictions(
        model=model,
        X=X_train,
        y=y_train,
    )

    predictions_2 = generate_oof_predictions(
        model=model,
        X=X_train,
        y=y_train,
    )

    assert np.allclose(predictions_1, predictions_2)


def test_probability_evaluation_returns_expected_metrics():
    y_true = np.array([0, 0, 1, 1])
    y_proba = np.array([0.1, 0.2, 0.8, 0.9])

    metrics = evaluate_probabilities(y_true, y_proba)

    assert "PR-AUC" in metrics
    assert "ROC-AUC" in metrics
    assert metrics["ROC-AUC"] > 0.5