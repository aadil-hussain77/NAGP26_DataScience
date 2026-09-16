from pathlib import Path
from typing import Literal

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


MODEL_PATH = Path("model/churn_pipeline.joblib")

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Saved model not found at: {MODEL_PATH.resolve()}"
    )

model_pipeline = joblib.load(MODEL_PATH)

app = FastAPI(
    title="Customer Churn Prediction API",
    description=(
        "Predicts whether a telecom customer is likely to churn "
        "using the saved Decision Tree pipeline."
    ),
    version="1.0.0"
)


class CustomerInput(BaseModel):
    gender: Literal["Female", "Male"]
    SeniorCitizen: int = Field(..., ge=0, le=1)
    Partner: Literal["Yes", "No"]
    Dependents: Literal["Yes", "No"]
    tenure: int = Field(..., ge=0, le=72)
    PhoneService: Literal["Yes", "No"]
    MultipleLines: Literal["Yes", "No", "No phone service"]
    InternetService: Literal["DSL", "Fiber optic", "No"]
    OnlineSecurity: Literal["Yes", "No", "No internet service"]
    OnlineBackup: Literal["Yes", "No", "No internet service"]
    DeviceProtection: Literal["Yes", "No", "No internet service"]
    TechSupport: Literal["Yes", "No", "No internet service"]
    StreamingTV: Literal["Yes", "No", "No internet service"]
    StreamingMovies: Literal["Yes", "No", "No internet service"]
    Contract: Literal["Month-to-month", "One year", "Two year"]
    PaperlessBilling: Literal["Yes", "No"]
    PaymentMethod: Literal[
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)"
    ]
    MonthlyCharges: float = Field(..., ge=0)
    TotalCharges: float = Field(..., ge=0)


def create_engineered_features(data: pd.DataFrame) -> pd.DataFrame:
    """Create the same engineered features used during model training."""
    processed_data = data.copy()

    tenure_bins = [-1, 12, 24, 48, 72]
    tenure_labels = [
        "0-12 months",
        "13-24 months",
        "25-48 months",
        "49-72 months"
    ]

    processed_data["TenureGroup"] = pd.cut(
        processed_data["tenure"],
        bins=tenure_bins,
        labels=tenure_labels
    )

    service_columns = [
        "PhoneService",
        "MultipleLines",
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies"
    ]

    processed_data["ServiceCount"] = (
        processed_data[service_columns]
        .eq("Yes")
        .sum(axis=1)
    )

    support_protection_columns = [
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport"
    ]

    processed_data["SupportProtectionCount"] = (
        processed_data[support_protection_columns]
        .eq("Yes")
        .sum(axis=1)
    )

    return processed_data


@app.get("/")
def home():
    return {
        "message": "Customer Churn Prediction API is running.",
        "documentation": "/docs"
    }


@app.post("/predict")
def predict_churn(customer: CustomerInput):
    try:
        customer_data = pd.DataFrame([customer.model_dump()])

        customer_data = create_engineered_features(customer_data)

        prediction = model_pipeline.predict(customer_data)[0]

        churn_probability_index = list(
            model_pipeline.classes_
        ).index("Yes")

        churn_probability = float(
            model_pipeline.predict_proba(customer_data)[0][
                churn_probability_index
            ]
        )

        return {
            "prediction": prediction,
            "churn_probability": round(churn_probability, 4)
        }

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(error)}"
        )