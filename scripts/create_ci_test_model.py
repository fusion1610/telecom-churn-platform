from pathlib import Path

import mlflow.sklearn
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = PROJECT_ROOT / ".ci" / "test_model"

CATEGORICAL_FEATURES = [
    "CRM_PID_Value_Segment",
    "EffectiveSegment",
]

NUMERIC_FEATURES = [
    "Active_subscribers",
    "Not_Active_subscribers",
    "Suspended_subscribers",
    "Total_SUBs",
    "AvgMobileRevenue",
    "AvgFIXRevenue",
    "ARPU",
]


def main() -> None:
    MODEL_DIR.parent.mkdir(parents=True, exist_ok=True)

    X = pd.DataFrame(
        {
            "CRM_PID_Value_Segment": ["Bronze", "Silver", "Gold", "Bronze"],
            "EffectiveSegment": ["SOHO", "SME", "VSE", "SOHO"],
            "Active_subscribers": [6, 10, 15, 4],
            "Not_Active_subscribers": [2, 1, 3, 0],
            "Suspended_subscribers": [0, 0, 1, 0],
            "Total_SUBs": [8, 11, 19, 4],
            "AvgMobileRevenue": [53.67, 120.0, 250.0, 40.0],
            "AvgFIXRevenue": [0.0, 20.0, 50.0, 0.0],
            "ARPU": [8.95, 12.73, 15.79, 10.0],
        }
    )

    y = [0, 1, 0, 1]

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                CATEGORICAL_FEATURES,
            ),
            (
                "numeric",
                "passthrough",
                NUMERIC_FEATURES,
            ),
        ]
    )

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                DummyClassifier(strategy="prior"),
            ),
        ]
    )

    model.fit(X, y)

    if MODEL_DIR.exists():
        import shutil

        shutil.rmtree(MODEL_DIR)

    mlflow.sklearn.save_model(
        sk_model=model,
        path=str(MODEL_DIR),
    )

    print(f"CI test model created at: {MODEL_DIR}")


if __name__ == "__main__":
    main()