from contextlib import asynccontextmanager

from fastapi import FastAPI

from src.api.routes.prediction import router as prediction_router
from src.service.prediction_service import ChurnPredictionService


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.prediction_service = ChurnPredictionService()

    yield

    app.state.prediction_service = None


app = FastAPI(
    title="Telecom Churn Prediction API",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health", tags=["health"])
def health():
    return {"status": "healthy"}


app.include_router(prediction_router)