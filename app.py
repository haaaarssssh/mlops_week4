from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle
import numpy as np
import uvicorn
from typing import List
import os

app = FastAPI(
    title="IRIS Classifier API",
    description="ML Model API for IRIS species classification",
    version="1.0.0"
)

# Load model at startup
MODEL_PATH = os.getenv("MODEL_PATH", "models/iris_model.pkl")
model = None

@app.on_event("startup")
async def load_model():
    global model
    try:
        with open(MODEL_PATH, 'rb') as f:
            model = pickle.load(f)
        print(f"✅ Model loaded successfully from {MODEL_PATH}")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        raise

class IrisFeatures(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float
    
    class Config:
        json_schema_extra = {
            "example": {
                "sepal_length": 5.1,
                "sepal_width": 3.5,
                "petal_length": 1.4,
                "petal_width": 0.2
            }
        }

class PredictionResponse(BaseModel):
    prediction: str
    prediction_id: int
    confidence: float

@app.get("/")
async def root():
    return {
        "message": "IRIS Classifier API",
        "health": "OK",
        "endpoints": ["/predict", "/health", "/docs"]
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model_loaded": model is not None
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict(features: IrisFeatures):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Prepare input
        input_data = np.array([[
            features.sepal_length,
            features.sepal_width,
            features.petal_length,
            features.petal_width
        ]])
        
        # Make prediction
        prediction_id = int(model.predict(input_data)[0])
        
        # Get probability scores
        probabilities = model.predict_proba(input_data)[0]
        confidence = float(probabilities[prediction_id])
        
        # Map to species name
        species_map = {0: "setosa", 1: "versicolor", 2: "virginica"}
        species_name = species_map.get(prediction_id, "unknown")
        
        return PredictionResponse(
            prediction=species_name,
            prediction_id=prediction_id,
            confidence=confidence
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
