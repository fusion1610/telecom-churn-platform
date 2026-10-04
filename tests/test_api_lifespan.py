import os

from fastapi.testclient import TestClient

from src.api.app import app


def test_prediction_service_loaded_at_startup():
    with TestClient(app) as client:
        service = client.app.state.prediction_service

        expected_model_uri = os.getenv(
            "MODEL_URI",
            "models:/telecom-churn-hgb@production",
        )

        assert service is not None
        assert service.model is not None
        assert service.model_uri == expected_model_uri
        assert service.threshold == 0.07