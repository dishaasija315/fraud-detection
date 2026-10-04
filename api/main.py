import json
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import List, Literal
import joblib
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from src.db import get_predictions, is_db_connected, save_prediction
from src.features import create_features

model = None
model_meta = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, model_meta
    try:
        model = joblib.load("models/fraud_model.joblib")
        with open("models/model_meta.json", "r") as f:
            model_meta = json.load(f)
    except Exception as e:
        print(f"Failed to load model/metadata: {e}")
    yield


app = FastAPI(
    title="Fraud Detection API",
    description="FastAPI service for predicting financial transaction fraud with MongoDB logging",
    version="1.0.0",
    lifespan=lifespan,
)


class TransactionInput(BaseModel):
    step: int = Field(
        ..., ge=0, description="Time step in hours (non-negative integer)"
    )
    type: Literal["TRANSFER", "CASH_OUT"] = Field(
        ..., description="Type of transaction (TRANSFER or CASH_OUT)"
    )
    amount: float = Field(
        ..., ge=0.0, description="Transaction amount (non-negative)"
    )
    oldbalanceOrg: float = Field(
        ..., ge=0.0, description="Initial balance of sender (non-negative)"
    )
    newbalanceOrig: float = Field(
        ..., ge=0.0, description="New balance of sender (non-negative)"
    )
    oldbalanceDest: float = Field(
        ..., ge=0.0, description="Initial balance of recipient (non-negative)"
    )
    newbalanceDest: float = Field(
        ..., ge=0.0, description="New balance of recipient (non-negative)"
    )


class PredictionResponse(BaseModel):
    fraud_probability: float = Field(
        ..., description="Predicted probability of fraud (0.0 to 1.0)"
    )
    is_fraud: bool = Field(
        ..., description="Classification result using the threshold"
    )
    threshold_used: float = Field(
        ..., description="Decision threshold used for prediction"
    )
    logged: bool = Field(
        ..., description="Whether prediction was successfully saved to database"
    )


class PredictionRecord(BaseModel):
    id: str = Field(..., alias="_id", description="MongoDB record ID")
    transaction: TransactionInput = Field(..., description="Transaction input data")
    fraud_probability: float = Field(..., description="Predicted fraud probability")
    is_fraud: bool = Field(..., description="Fraud classification result")
    threshold_used: float = Field(..., description="Threshold used")
    timestamp: datetime = Field(..., description="UTC timestamp of prediction")

    class Config:
        populate_by_name = True


class HealthResponse(BaseModel):
    status: str = Field(..., description="API operational status")
    model_loaded: bool = Field(
        ..., description="Whether model and metadata are loaded"
    )
    db_connected: bool = Field(
        ..., description="Whether MongoDB database connection is active"
    )


@app.get("/health", response_model=HealthResponse)
def health_check():
    return {
        "status": "ok",
        "model_loaded": model is not None and model_meta is not None,
        "db_connected": is_db_connected(),
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(transaction: TransactionInput):
    if model is None or model_meta is None:
        raise HTTPException(status_code=500, detail="Model or metadata not loaded.")

    features_df = create_features(transaction, model_meta["features"])

    # Predict fraud probability (probability of class 1)
    probabilities = model.predict_proba(features_df)
    fraud_prob = float(probabilities[0][1])

    threshold = float(model_meta["threshold"])
    is_fraud = bool(fraud_prob >= threshold)
    timestamp = datetime.now(timezone.utc)

    tx_dict = (
        transaction.model_dump()
        if hasattr(transaction, "model_dump")
        else transaction.dict()
    )

    logged = False
    try:
        logged = save_prediction(
            transaction=tx_dict,
            fraud_probability=round(fraud_prob, 4),
            is_fraud=is_fraud,
            threshold_used=threshold,
            timestamp=timestamp,
        )
    except Exception as e:
        print(f"Failed to log prediction to MongoDB: {e}")
        logged = False

    return PredictionResponse(
        fraud_probability=round(fraud_prob, 4),
        is_fraud=is_fraud,
        threshold_used=threshold,
        logged=logged,
    )


@app.get("/predictions", response_model=List[PredictionRecord])
def list_predictions(
    limit: int = Query(
        20, ge=1, le=100, description="Maximum number of predictions to fetch (1-100)"
    )
):
    try:
        return get_predictions(limit=limit)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database query failed: {str(e)}")
