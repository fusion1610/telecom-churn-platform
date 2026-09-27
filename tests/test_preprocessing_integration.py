from src.data.preprocessing import (
    build_preprocessor,
    load_raw_data,
    prepare_dataset,
)
from src.data.split import create_train_test_split


def test_preprocessor_is_fitted_only_on_training_data():
    df = load_raw_data()

    X, y = prepare_dataset(df)

    X_train, X_test, y_train, y_test = create_train_test_split(
        X,
        y,
    )

    preprocessor = build_preprocessor()

    X_train_transformed = preprocessor.fit_transform(X_train)
    X_test_transformed = preprocessor.transform(X_test)

    assert X_train_transformed.shape[0] == len(X_train)
    assert X_test_transformed.shape[0] == len(X_test)

    assert (
        X_train_transformed.shape[1]
        == X_test_transformed.shape[1]
    )