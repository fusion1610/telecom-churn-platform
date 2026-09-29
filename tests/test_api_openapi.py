from fastapi.testclient import TestClient

from src.api.app import app


def test_openapi_documentation_is_available():
    with TestClient(app) as client:
        response = client.get("/openapi.json")

    assert response.status_code == 200

    schema = response.json()

    assert schema["info"]["title"] == "Telecom Churn Prediction API"
    assert schema["info"]["version"] == "1.0.0"


def test_churn_prediction_endpoint_is_documented():
    with TestClient(app) as client:
        response = client.get("/openapi.json")

    schema = response.json()

    assert "/predictions/churn" in schema["paths"]

    endpoint = schema["paths"]["/predictions/churn"]["post"]

    assert endpoint["summary"] == "Predict customer churn risk"
    assert "requestBody" in endpoint
    assert "responses" in endpoint


def test_health_endpoint_is_documented():
    with TestClient(app) as client:
        response = client.get("/openapi.json")

    schema = response.json()

    assert "/health" in schema["paths"]
    assert "get" in schema["paths"]["/health"]