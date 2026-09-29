from contextlib import asynccontextmanager

from fastapi import FastAPI

from src.service.prediction_service import ChurnPredictionService


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize application resources at startup."""

    app.state.prediction_service = ChurnPredictionService()

    yield

    # Release application resources during shutdown.
    app.state.prediction_service = None


app = FastAPI(
    title="Telecom Churn Prediction API",
    version="1.0.0",
    lifespan=lifespan,
)