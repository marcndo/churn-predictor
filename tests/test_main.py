"""
Tests for main.py — the FastAPI application.
Uses FastAPI's TestClient which simulates HTTP requests
without needing a running server.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi.testclient import TestClient
from main import app

client = TestClient(app)


def test_health_check():
    """Health endpoint should return healthy status."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_root_endpoint():
    """Root endpoint should return a welcome message."""
    response = client.get("/")
    assert response.status_code == 200
    assert "message" in response.json()


def test_predict_valid_input():
    """A valid customer payload should return a prediction."""
    valid_customer = {
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
    response = client.post("/predict", json=valid_customer)
    assert response.status_code == 200
    data = response.json()
    assert data["churn_prediction"] in ["Yes", "No"]
    assert 0 <= data["churn_probability"] <= 1
    assert data["risk_level"] in ["Low", "Medium", "High"]


def test_predict_invalid_contract_value():
    """An invalid Contract value should be rejected with 422."""
    invalid_customer = {
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
        "Contract": "Quarterly",  # invalid — model never saw this
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 70.35
    }
    response = client.post("/predict", json=invalid_customer)
    assert response.status_code == 422


def test_predict_missing_field():
    """A request missing a required field should be rejected."""
    incomplete_customer = {
        "gender": "Female",
        "tenure": 12
        # missing all other required fields
    }
    response = client.post("/predict", json=incomplete_customer)
    assert response.status_code == 422