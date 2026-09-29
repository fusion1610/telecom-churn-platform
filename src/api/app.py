from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from src.api.routes.prediction import router as prediction_router
from src.service.prediction_service import ChurnPredictionService


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.prediction_service = ChurnPredictionService()
    yield
    app.state.prediction_service = None


app = FastAPI(
    title="Telecom Churn Prediction API",
    description=(
        "Production API for B2B telecom customer churn prediction "
        "and retention risk prioritization."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


@app.exception_handler(Exception)
async def unhandled_exception_handler(
    request: Request,
    exc: Exception,
):
    return JSONResponse(
        status_code=500,
        content={
            "detail": "Prediction service unavailable.",
        },
    )


@app.get("/health", tags=["health"])
def health():
    return {"status": "healthy"}


app.include_router(prediction_router)