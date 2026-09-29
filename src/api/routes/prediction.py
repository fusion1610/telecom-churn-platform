import pandas as pd

from fastapi import APIRouter, Request

from src.api.models.schemas import (
    ChurnPredictionRequest,
    ChurnPredictionResponse,
)


router = APIRouter(
    prefix="/predictions",
    tags=["predictions"],
)


@router.post(
    "/churn",
    response_model=ChurnPredictionResponse,
)
def predict_churn(
    payload: ChurnPredictionRequest,
    request: Request,
) -> ChurnPredictionResponse:
    service = request.app.state.prediction_service

    input_data = pd.DataFrame(
        [payload.model_dump()]
    )

    result = service.predict(input_data).iloc[0]

    response = {
        "Churn_Probability": float(
            result["Churn Probability"]
        ),
        "Retention_Flag": bool(
            result["Retention Flag"]
        ),
        "Risk_Band": str(
            result["Risk Band"]
        ),
    }

    if "Revenue at Risk" in result:
        response["Revenue_at_Risk"] = float(
            result["Revenue at Risk"]
        )
        response["Priority_Tier"] = str(
            result["Priority Tier"]
        )
        response["Retention_Action"] = str(
            result["Retention Action"]
        )
        response["Retention_Urgency"] = str(
            result["Retention Urgency"]
        )

    return ChurnPredictionResponse(**response)