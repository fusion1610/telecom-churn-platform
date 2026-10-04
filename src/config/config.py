import os


MODEL_URI = os.getenv(
    "MODEL_URI",
    "models:/telecom-churn-hgb@production",
)

MLFLOW_TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
)

MODEL_THRESHOLD = float(
    os.getenv("MODEL_THRESHOLD", "0.07")
)