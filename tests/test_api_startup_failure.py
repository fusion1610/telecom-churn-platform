import pytest
from fastapi.testclient import TestClient

from src.api.app import app
from src.service.prediction_service import ChurnPredictionService


def test_application_fails_startup_when_model_cannot_load(monkeypatch):
    def fail_to_load_model(*args, **kwargs):
        raise RuntimeError("Simulated model loading failure")

    monkeypatch.setattr(
        "src.api.app.ChurnPredictionService",
        fail_to_load_model,
    )

    with pytest.raises(RuntimeError, match="Simulated model loading failure"):
        with TestClient(app):
            pass