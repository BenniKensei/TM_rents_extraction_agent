from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional
import os
import pandas as pd
import joblib
from contextlib import asynccontextmanager

# Global dictionary to store the dynamically loaded model Pipeline
ml_models = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifecycle event handler allowing FastAPI to load the serialized pipeline securely
    into memory strictly once per server instantiation. Circumvents disk I/O bottlenecks.
    """
    # Dynamically locate the assets folder irrespective of exact working directory
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = os.path.join(base_dir, 'assets', 'model.joblib')
    
    if not os.path.exists(model_path):
        raise RuntimeError(f"Serialized model file missing at {model_path}. Execute Step 2 training first.")
    
    print(f"Loading binary machine learning pipeline from: {model_path}")
    ml_models["rent_predictor"] = joblib.load(model_path)
    print("✅ Model successfully initialized into memory.")
    
    yield
    # Memory cleanup triggered safely on server shutdown
    ml_models.clear()

app = FastAPI(
    title="Timișoara Real Estate Inference API",
    description="Mathematical prediction endpoints for apartment rents utilizing memory-resident XGBoost Decision Trees.",
    version="1.0.0",
    lifespan=lifespan
)

# 1. API Design & Strict Schema Validation
class ApartmentFeatures(BaseModel):
    neighborhood: str = Field(..., description="Target property's geographical neighborhood")
    rooms: int = Field(..., description="Mathematical quantity of rooms", ge=1)
    is_pet_friendly: Optional[bool] = Field(False, description="Whether the host accepts animals")

class PredictionResponse(BaseModel):
    predicted_rent_eur: float = Field(..., description="Algorithmically determined monthly rent equivalent in EUR")

# 3. Inference Endpoint Execution
@app.post("/predict", response_model=PredictionResponse)
async def predict_rent_price(features: ApartmentFeatures):
    """
    Accepts Pydantic validated structured JSON containing apartment characteristics, 
    dynamically evaluates the data directly through the pre-loaded XGBoost pipeline, and 
    returns a mathematical regression extraction.
    """
    if "rent_predictor" not in ml_models:
        raise HTTPException(status_code=503, detail="Prediction models unavailable.")
    
    try:
        # Structuring standard tabular DataFrame preserving pipeline column alignments
        data_dict = features.model_dump()
        input_frame = pd.DataFrame([data_dict])
        
        # Activating SKLearn model processing chain
        prediction = ml_models["rent_predictor"].predict(input_frame)[0]
        
        return PredictionResponse(predicted_rent_eur=round(float(prediction), 2))
    
    except Exception as e:
        # 4. Edge Case Handling (Type mismatches / Structural issues gracefully blocked)
        raise HTTPException(status_code=400, detail=f"Inference failure: {str(e)}")

@app.get("/health")
async def health_status():
    """Confirms persistent model bindings in production instances."""
    return {"status": "operational", "model_resident": "rent_predictor" in ml_models}
