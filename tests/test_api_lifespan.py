from fastapi.testclient import TestClient

from src.api.app import app


def test_prediction_service_loaded_at_startup():
    with TestClient(app) as client:
        service = client.app.state.prediction_service

        assert service is not None
        assert service.model is not None
        assert service.model_uri == (
            "models:/telecom-churn-hgb@production"
        )
        assert service.threshold == 0.07