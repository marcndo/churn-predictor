"""
FastAPI application entry point.
Serves the train churn prediction model as REST API

Run locally with: univcorn main:app --reload
"""

from fastapi import FastAPI, HttpException
from pydantic import BaseModel, Field
from typing import Literal
import joblib
import pandas as pd
import os

# ===============================================
# APP INITIALIZATION
# ===============================================

app = FastAPI(
    title = "Customer Churn Prediction API",
    description="Predicts the probability that a telecom customer will churn",
    version = "1.0.0"
)

# ============================================================
# LOAD THE TRAINED MODEL AT STARTUP
# ============================================================

MODEL_PATH = "models/best_model.joblib"

if not os.path.exits(MODEL_PATH):
    raise FileNotFoundError(
        f"Model file not found at {MODEL_PATH}. "
        f"Make sure base_model.joblib is commited to repo."
    )

pipeline = joblib.load(MODEL_PATH)
print(f"Successfully loaded the model at {MODEL_PATH}")

# ============================================================
# REQUEST SCHEMA — Pydantic model
# ============================================================
class CustomData:
    """
    Defines the shape of the data the API takes
    """
    gender: Literal["Male", "Female"]
    SeniorCitizen: Literal[0, 1] = Field(..., description="0=No, 1=Yes")
    Partner: Literal["Yes", "No"]
    Dependents: Literal["Yes", "No"]
    tenure: int = Field(..., ge=0, le=100, description="Months as a customer")
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
        "Electronic check", "Mailed check",
        "Bank transfer (automatic)", "Credit card (automatic)"
    ]
    MonthlyCharges: float = Field(..., ge=0, le=200)
    
    class Config:
    json_schema_extra = {
        "example": {
            "gender": "Female",
            "SeniorCitizen": 0,
            "Partner": "Yes",
            "Dependents": "No",
            "tenure": 12,
            "PhoneService": "Yes",
            "MultipleLines": "No",
            "InternetService": "Fiber optic",
            "OnlineSecurity": "No",
            "OnlineBackup": "Yes",
            "DeviceProtection": "No",
            "TechSupport": "No",
            "StreamingTV": "Yes",
            "StreamingMovies": "No",
            "Contract": "Month-to-month",
            "PaperlessBilling": "Yes",
            "PaymentMethod": "Electronic check",
            "MonthlyCharges": 70.35
        }
    }

# ============================================================
# RESPONSE SCHEMA
# ============================================================

"""
main.py

FastAPI application entry point.
Serves the trained churn prediction model as a REST API.

Run locally with: uvicorn main:app --reload
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Literal
import joblib
import pandas as pd
import os


# ============================================================
# APP INITIALIZATION
# ============================================================

app = FastAPI(
    title="Customer Churn Prediction API",
    description="Predicts the probability that a telecom customer will churn",
    version="1.0.0"
)


# ============================================================
# LOAD THE TRAINED MODEL AT STARTUP
# ============================================================

MODEL_PATH = "models/best_model.joblib"

# Why load once at module level, not inside the endpoint function:
# Loading a model from disk takes time (deserializing the joblib file).
# If you loaded it inside the predict function, every single request
# would reload the model from disk — wasting time on every call.
# Loading once at startup means the model stays in memory and every
# request just calls .predict() on the already-loaded object.

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Model file not found at {MODEL_PATH}. "
        f"Make sure best_model.joblib is committed to the repo."
    )

pipeline = joblib.load(MODEL_PATH)
print(f"Model loaded successfully from {MODEL_PATH}")


# ============================================================
# REQUEST SCHEMA — Pydantic model
# ============================================================

class CustomerData(BaseModel):
    """
    Defines the exact shape of data this API accepts.

    Why Literal types for categorical fields:
    Literal restricts a field to only the exact values your model
    was trained on. If someone sends Contract="Quarterly" — a value
    your model never saw during training — FastAPI rejects the
    request with a clear 422 error before it ever reaches your model.
    Without this, your OneHotEncoder's handle_unknown='ignore' would
    silently encode it as all zeros, giving a prediction based on
    incomplete information without anyone knowing something was wrong.

    Why Field(...) with examples:
    The example values appear in FastAPI's auto-generated docs at
    /docs, making your API self-documenting for anyone who visits it.
    """
    gender: Literal["Male", "Female"]
    SeniorCitizen: Literal[0, 1] = Field(..., description="0=No, 1=Yes")
    Partner: Literal["Yes", "No"]
    Dependents: Literal["Yes", "No"]
    tenure: int = Field(..., ge=0, le=100, description="Months as a customer")
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
        "Electronic check", "Mailed check",
        "Bank transfer (automatic)", "Credit card (automatic)"
    ]
    MonthlyCharges: float = Field(..., ge=0, le=200)

    class Config:
        json_schema_extra = {
            "example": {
                "gender": "Female",
                "SeniorCitizen": 0,
                "Partner": "Yes",
                "Dependents": "No",
                "tenure": 12,
                "PhoneService": "Yes",
                "MultipleLines": "No",
                "InternetService": "Fiber optic",
                "OnlineSecurity": "No",
                "OnlineBackup": "Yes",
                "DeviceProtection": "No",
                "TechSupport": "No",
                "StreamingTV": "Yes",
                "StreamingMovies": "No",
                "Contract": "Month-to-month",
                "PaperlessBilling": "Yes",
                "PaymentMethod": "Electronic check",
                "MonthlyCharges": 70.35
            }
        }


# ============================================================
# RESPONSE SCHEMA
# ============================================================

class ChurnPrediction(BaseModel):
    """
    Defines the exact shape of data this API returns.

    Why a response schema too, not just returning a raw dict:
    Pydantic validates your OWN output before sending it, catching
    bugs where your code accidentally returns the wrong type. It
    also generates accurate API documentation automatically.
    """
    churn_prediction: Literal["Yes", "No"]
    churn_probability: float = Field(..., ge=0, le=1)
    risk_level: Literal["Low", "Medium", "High"]


# ============================================================
# HEALTH CHECK ENDPOINT
# ============================================================

@app.get("/health")
def health_check():
    """
    Why a health endpoint exists:
    Render and other hosting platforms periodically ping this
    endpoint to confirm your API is alive and responding. If it
    fails, the platform knows to restart your service. This is
    standard practice for any production API — never skip it.
    """
    return {"status": "healthy", "model_loaded": pipeline is not None}


# ============================================================
# PREDICTION ENDPOINT
# ============================================================

@app.post("/predict", response_model=ChurnPrediction)
def predict_churn(customer: CustomerData):
    """
    Predict whether a customer will churn.

    Why we convert the Pydantic object to a DataFrame:
    Your trained pipeline was fit on pandas DataFrames during
    training. It expects the same structure at prediction time —
    same column names, same data types. customer.dict() converts
    the validated Pydantic object into a plain dictionary, and
    pd.DataFrame([...]) wraps it as a single-row DataFrame matching
    the shape your pipeline was trained on.

    Why wrapped in try/except:
    Even with Pydantic validation, unexpected errors can occur
    (e.g., a model file issue). Returning a clean 500 error with
    a message is better than an unhandled crash that exposes
    internal stack traces to API consumers.
    """
    try:
        # Convert validated request into DataFrame matching training format
        input_df = pd.DataFrame([customer.dict()])

        # Run prediction through the full pipeline
        # (preprocessing happens automatically inside .predict())
        prediction = pipeline.predict(input_df)[0]
        probability = pipeline.predict_proba(input_df)[0][1]

        # Convert numeric prediction to human-readable label
        churn_label = "Yes" if prediction == 1 else "No"

        # Bucket probability into risk levels for business usability
        if probability >= 0.7:
            risk = "High"
        elif probability >= 0.4:
            risk = "Medium"
        else:
            risk = "Low"

        return ChurnPrediction(
            churn_prediction=churn_label,
            churn_probability=round(float(probability), 4),
            risk_level=risk
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )


# ============================================================
# RESPONSE SCHEMA
# ============================================================

class ChurnPrediction(BaseModel):
    """
    Defines the exact shape of data this API returns.
    """
    churn_prediction: Literal["Yes", "No"]
    churn_probability: float = Field(..., ge=0, le=1)
    risk_level: Literal["Low", "Medium", "High"]


# ============================================================
# PREDICTION ENDPOINT
# ============================================================
@app.post("/predict", response_model=ChurnPrediction)
def predict_churn(customer: CustomerData):
    """
    Predict whether a customer will churn.
    """
    try:
        input_df = pd.DataFrame([customer.dict()])

        prediction = pipeline.predict(input_df)[0]
        probability = pipeline.predict_proba(input_df)[0][1]

        churn_label = "Yes" if prediction == 1 else "No"

        if probability >= 0.7:
            risk = "High"
        elif probability >= 0.4:
            risk = "Medium"
        else:
            risk = "Low"

        return ChurnPrediction(
            churn_prediction=churn_label,
            churn_probability=round(float(probability), 4),
            risk_level=risk
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )


# ============================================================
# ROOT ENDPOINT
# ============================================================

@app.get("/")
def root():
    """Simple landing message confirming the API is running."""
    return {
        "message": "Customer Churn Prediction API",
        "docs": "/docs",
        "health": "/health"
    }
