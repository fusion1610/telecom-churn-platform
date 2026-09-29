from fastapi.testclient import TestClient

from src.api.app import app


VALID_PAYLOAD = {
    "CRM_PID_Value_Segment": "Bronze",
    "EffectiveSegment": "SOHO",
    "Active_subscribers": 6,
    "Not_Active_subscribers": 2,
    "Suspended_subscribers": 0,
    "Total_SUBs": 8,
    "AvgMobileRevenue": 53.67,
    "AvgFIXRevenue": 0.0,
    "ARPU": 8.95,
    "TotalRevenue": 53.67,
}


def test_prediction_failure_returns_safe_500_response(monkeypatch):
    def fail_prediction(*args, **kwargs):
        raise RuntimeError("Internal model failure")

    with TestClient(app, raise_server_exceptions=False) as client:
        monkeypatch.setattr(
            client.app.state.prediction_service,
            "predict",
            fail_prediction,
        )

        response = client.post(
            "/predictions/churn",
            json=VALID_PAYLOAD,
        )

    assert response.status_code == 500
    assert response.json() == {
        "detail": "Prediction service unavailable."
    }


def test_health_endpoint_survives_prediction_failure(monkeypatch):
    def fail_prediction(*args, **kwargs):
        raise RuntimeError("Internal model failure")

    with TestClient(app) as client:
        monkeypatch.setattr(
            client.app.state.prediction_service,
            "predict",
            fail_prediction,
        )

        response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}