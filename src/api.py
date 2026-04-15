"""
FastAPI Inference Server for Real Estate Rent Predictions
==========================================================
Exposes a high-performance REST API for XGBoost-based apartment rent predictions.
Loads serialized ML pipeline at startup and processes prediction requests with <100ms latency.

Architecture:
- Single-threaded model loading via FastAPI lifespan context manager
- Pydantic schema validation on inbound requests (type safety)
- Fast vectorized inference using pre-loaded model
- Graceful error handling for edge cases and API failures

Deployment:
  $ uvicorn src.api:app --host 0.0.0.0 --port 8000 --reload
  
API Endpoints:
  POST /predict      - Rent prediction for apartment features
  GET /health        - Health check (model readiness verification)

Expected Inference Latency:
- Cold start (first request): 2-5 seconds (model load + preprocessing)
- Warm requests: <50ms (in-memory model access)
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional
import os
import pandas as pd
import joblib
from contextlib import asynccontextmanager


# ────────────────────────────────────────────────────────────────────────────
# GLOBAL STATE: Model Repository
# ────────────────────────────────────────────────────────────────────────────

# Global dictionary to store the dynamically loaded model Pipeline
# Key: 'rent_predictor' -> Value: sklearn.pipeline.Pipeline (preprocessor + XGBoost model)
# Motivation: Avoid re-loading model from disk on every request (5-10x latency cost)
ml_models = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI lifespan context manager: Load/unload resources at server startup/shutdown.
    
    **Why use lifespan?**
    - Model files (~2-5 MB) loaded once at server startup, not per-request
    - Avoids disk I/O bottleneck (10-50ms per request)
    - Gracefully cleans up resources on shutdown (memory deallocation)
    
    **Execution Flow:**
    1. Server startup: Load model.joblib from disk into ml_models['rent_predictor']
    2. Request handling: All requests access in-memory model (low latency)
    3. Server shutdown: Clear ml_models dictionary (memory cleanup)
    
    Raises:
        RuntimeError: If model file not found at expected path.
    
    Yields:
        (None) - Resumes server startup after resource initialization.
    """
    # ─── STARTUP: Load Model ───
    # Dynamically locate the assets folder irrespective of exact working directory
    # Rationale: Supports both direct script execution and docker container paths
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = os.path.join(base_dir, 'assets', 'model.joblib')
    
    if not os.path.exists(model_path):
        raise RuntimeError(
            f"Serialized model file missing at {model_path}. "
            f"Execute training (python src/model_training.py) to generate model."
        )
    
    print(f"📦 Loading binary machine learning pipeline from: {model_path}")
    ml_models["rent_predictor"] = joblib.load(model_path)
    print("✅ Model successfully initialized into memory.")
    
    yield  # ─── READY FOR REQUESTS ───
    
    # ─── SHUTDOWN: Cleanup ───
    # Memory cleanup triggered safely on server shutdown
    ml_models.clear()
    print("🧹 Model unloaded from memory.")


# ────────────────────────────────────────────────────────────────────────────
# FASTAPI APPLICATION SETUP
# ────────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Timișoara Real Estate Inference API",
    description="Mathematical prediction endpoints for apartment rents utilizing memory-resident XGBoost Decision Trees.",
    version="1.0.0",
    lifespan=lifespan  # Attach startup/shutdown handlers
)


# ────────────────────────────────────────────────────────────────────────────
# PYDANTIC MODELS: Request/Response Schemas
# ────────────────────────────────────────────────────────────────────────────

class ApartmentFeatures(BaseModel):
    """
    Request schema: Apartment characteristics for rent prediction.
    
    Pydantic automatically:
    - Validates data types (returns 422 if types mismatch)
    - Enforces constraints (rooms >= 1)
    - Generates OpenAPI documentation at /docs
    
    Attributes:
        neighborhood: High-cardinality categorical feature (15-30 unique values).
                     Examples: "Centru", "Complexul Studentesc", "Periphery"
        rooms: Integer room count. Constraint: ge=1 (greater than or equal to 1).
               Validation rejects queries like rooms=0 or rooms=-1.
        is_pet_friendly: Optional binary feature. Defaults to False if omitted.
    """
    neighborhood: str = Field(
        ...,
        description="Target property's geographical neighborhood"
    )
    rooms: int = Field(
        ...,
        description="Mathematical quantity of rooms",
        ge=1  # Constraint: greater than or equal to 1
    )
    is_pet_friendly: Optional[bool] = Field(
        False,
        description="Whether the host accepts animals"
    )


class PredictionResponse(BaseModel):
    """
    Response schema: Model prediction with metadata.
    
    Attributes:
        predicted_rent_eur: Algorithmically determined monthly rent in EUR.
                           Output of XGBoost regressor (post-processing pipeline).
    """
    predicted_rent_eur: float = Field(
        ...,
        description="Algorithmically determined monthly rent equivalent in EUR"
    )


# ────────────────────────────────────────────────────────────────────────────
# INFERENCE ENDPOINT
# ────────────────────────────────────────────────────────────────────────────

@app.post("/predict", response_model=PredictionResponse)
async def predict_rent_price(features: ApartmentFeatures):
    """
    Predict monthly apartment rent from structured features using XGBoost model.
    
    **Request Validation Pipeline:**
    1. Parse JSON body
    2. Validate types and constraints (Pydantic)
    3. Check model availability in memory
    4. Execute preprocessing (target encoding, binary encoding)
    5. Run inference (XGBoost prediction)
    6. Return rounded prediction
    
    **Expected Input Shape:** (1, 3)
    - Row: Single apartment instance
    - Columns: ['neighborhood', 'rooms', 'is_pet_friendly'] (order preserved)
    
    **Expected Output:** Float32 scalar (EUR/month)
    - Typical range: €200-€800 (depending on training data)
    - Outlier predictions possible if input features are atypical
    
    **Processing Latency:** <50ms (warm inference from memory)
    
    Args:
        features: Pydantic-validated ApartmentFeatures object containing:
                 - neighborhood (str): Categorical location
                 - rooms (int): Room count (≥1)
                 - is_pet_friendly (bool): Pet policy indicator
    
    Returns:
        PredictionResponse: JSON object with predicted_rent_eur field.
    
    Raises:
        HTTPException 503: Model not loaded (ml_models["rent_predictor"] missing).
                         Indicates server startup failure.
        HTTPException 400: Inference failure (preprocessing or prediction error).
                         Rare; indicates data-model mismatch or data type issues.
    
    Example Request:
        POST /predict HTTP/1.1
        Content-Type: application/json
        
        {
          "neighborhood": "Centru",
          "rooms": 2,
          "is_pet_friendly": true
        }
    
    Example Response:
        HTTP/1.1 200 OK
        Content-Type: application/json
        
        {
          "predicted_rent_eur": 525.50
        }
    """
    # ─── Validation ───
    if "rent_predictor" not in ml_models:
        raise HTTPException(
            status_code=503,
            detail="Prediction models unavailable. Server initialization failed."
        )
    
    try:
        # ─── Data Preparation ───
        # Convert Pydantic model to dict, then to DataFrame for pipeline consumption
        # Expected shape: (1, 3) - single row, three features
        data_dict = features.model_dump()
        input_frame = pd.DataFrame([data_dict])
        
        # ─── Inference ───
        # Execute full preprocessing pipeline:
        # 1. Target encode neighborhood (string -> float, mean rent per area)
        # 2. Binary encode is_pet_friendly (bool -> int)
        # 3. Pass through rooms (int, unchanged)
        # 4. XGBoost prediction (ensemble of decision trees)
        prediction = ml_models["rent_predictor"].predict(input_frame)[0]
        
        # ─── Post-Processing ───
        # Round to 2 decimal places for currency representation
        prediction_rounded = round(float(prediction), 2)
        
        return PredictionResponse(predicted_rent_eur=prediction_rounded)
    
    except Exception as e:
        # ─── Error Handling ───
        # Catch preprocessing failures (e.g., unexpected feature values)
        # Return informative 400 Bad Request to client
        raise HTTPException(
            status_code=400,
            detail=f"Inference failure: {str(e)}"
        )


# ────────────────────────────────────────────────────────────────────────────
# HEALTH CHECK ENDPOINT
# ────────────────────────────────────────────────────────────────────────────

@app.get("/health")
async def health_status():
    """
    Health check endpoint for monitoring and orchestration.
    
    **Purpose:**
    - Kubernetes/Docker health probes (verify container readiness)
    - Load balancer checks (ensure model is loaded before routing traffic)
    - Monitoring systems (confirm API is responsive)
    
    **Response:**
    - status: "operational" if server is running
    - model_resident: True if model is loaded in memory, False otherwise
    
    Returns:
        dict: Health status with keys:
            - status (str): "operational" if API is ready
            - model_resident (bool): True if model in memory, False otherwise
    
    Example Response:
        {
          "status": "operational",
          "model_resident": true
        }
    
    FIXME: Consider adding model_version and inference_latency metrics for production monitoring.
    """
    return {
        "status": "operational",
        "model_resident": "rent_predictor" in ml_models
    }
