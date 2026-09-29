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


def test_predict_returns_prediction():
    with TestClient(app) as client:
        response = client.post("/predictions/churn", json=VALID_PAYLOAD)

    assert response.status_code == 200

    body = response.json()

    assert 0 <= body["Churn_Probability"] <= 1
    assert isinstance(body["Retention_Flag"], bool)
    assert body["Risk_Band"] in {"Low", "Moderate", "High"}

    assert body["Revenue_at_Risk"] >= 0
    assert body["Priority_Tier"] in {"Standard", "Priority", "Critical"}
    assert isinstance(body["Retention_Action"], str)
    assert body["Retention_Urgency"] in {
        "Low",
        "Medium",
        "High",
        "Very High",
    }


def test_predict_rejects_target_column():
    payload = {
        **VALID_PAYLOAD,
        "CHURN": "Yes",
    }

    with TestClient(app) as client:
        response = client.post("/predictions/churn", json=payload)

    assert response.status_code == 422


def test_predict_rejects_negative_subscribers():
    payload = {
        **VALID_PAYLOAD,
        "Active_subscribers": -1,
    }

    with TestClient(app) as client:
        response = client.post("/predictions/churn", json=payload)

    assert response.status_code == 422