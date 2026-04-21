from contextlib import asynccontextmanager

import mlflow
import mlflow.pyfunc
import numpy as np
from fastapi import FastAPI, BackgroundTasks

from app.schemas import PredictRequest, PredictResponse, InfoResponse
from app.config import settings
from app.db import SessionLocal, ModelResult


class AppState:
    model = None


state = AppState()


@asynccontextmanager
async def lifespan(app: FastAPI):
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    
    model_uri = f"models:/{settings.model_name}/{settings.model_version}"
    state.model = mlflow.pyfunc.load_model(model_uri)

    yield


app = FastAPI(title="MLOps Platform", lifespan=lifespan)


def save_result_to_db(features: list[float], prediction_int: int):
    db = SessionLocal()
    try:
        row = ModelResult(
            feature_1=features[0],
            feature_2=features[1],
            feature_3=features[2],
            feature_4=features[3],
            prediction=prediction_int,
        )
        db.add(row)
        db.commit()
    finally:
        db.close()


@app.get("/info", response_model=InfoResponse)
def get_info():
    return InfoResponse(
        model_name=settings.model_name,
        model_version=settings.model_version
    )


@app.post("/predict", response_model=PredictResponse)
def predict(payload: PredictRequest, background_tasks: BackgroundTasks):
    features = np.array([payload.features])
    prediction = state.model.predict(features)[0]
    prediction_int = int(prediction)

    response = PredictResponse(prediction=prediction_int)

    background_tasks.add_task(
        save_result_to_db,
        payload.features,
        prediction_int
    )

    return response
