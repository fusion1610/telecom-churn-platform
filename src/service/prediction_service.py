import mlflow
import pandas as pd

from src.business.risk import (
    assign_priority_tier,
    assign_retention_action,
    assign_retention_urgency,
    assign_risk_band,
    calculate_revenue_at_risk,
)

from src.config.config import MODEL_THRESHOLD, MODEL_URI

import os

DEFAULT_MODEL_URI = MODEL_URI
DEFAULT_THRESHOLD = MODEL_THRESHOLD

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")

if MLFLOW_TRACKING_URI:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

MODEL_FEATURES = [
    "CRM_PID_Value_Segment",
    "EffectiveSegment",
    "Active_subscribers",
    "Not_Active_subscribers",
    "Suspended_subscribers",
    "Total_SUBs",
    "AvgMobileRevenue",
    "AvgFIXRevenue",
    "ARPU",
]


class ChurnPredictionService:
    """Prediction service backed by one loaded MLflow model."""

    def __init__(
        self,
        model_uri: str = DEFAULT_MODEL_URI,
        threshold: float = DEFAULT_THRESHOLD,
    ):
        if not 0 <= threshold <= 1:
            raise ValueError("threshold must be between 0 and 1.")

        self.model_uri = model_uri
        self.threshold = threshold

        # Model is loaded once when the service is initialized.
        self.model = mlflow.sklearn.load_model(model_uri)

    def predict(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate churn and business-risk predictions."""

        if not isinstance(data, pd.DataFrame):
            raise TypeError("data must be a pandas DataFrame.")

        if data.empty:
            raise ValueError("data cannot be empty.")

        missing_columns = [
            column
            for column in MODEL_FEATURES
            if column not in data.columns
        ]

        if missing_columns:
            raise ValueError(
                f"Missing required model features: {missing_columns}"
            )

        model_input = data[MODEL_FEATURES].copy()

        probabilities = self.model.predict_proba(model_input)[:, 1]

        result = data.copy()

        result["Churn Probability"] = probabilities
        result["Retention Flag"] = (
            result["Churn Probability"] >= self.threshold
        )

        result["Risk Band"] = assign_risk_band(
            result["Churn Probability"]
        )

        # Revenue-risk calculations are downstream business logic.
        if "TotalRevenue" in result.columns:
            result["Revenue at Risk"] = calculate_revenue_at_risk(
                result["Churn Probability"].to_numpy(),
                result["TotalRevenue"].to_numpy(),
            )

            result["Priority Tier"] = assign_priority_tier(
                result["Revenue at Risk"]
            )

            result["Retention Action"] = assign_retention_action(
                result["Risk Band"],
                result["Priority Tier"],
            )

            result["Retention Urgency"] = assign_retention_urgency(
                result["Risk Band"],
                result["Priority Tier"],
            )

        return result