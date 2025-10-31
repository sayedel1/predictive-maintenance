# ==========================
# predictive_maintenance_api.py
# FastAPI Application for Predictive Maintenance Model
# ==========================

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import pandas as pd
import numpy as np
import joblib
from typing import List, Optional, Dict, Any
import logging
from datetime import datetime
import uvicorn

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ==========================
# FastAPI Application Setup
# ==========================

app = FastAPI(
    title="Predictive Maintenance API",
    description="API for predicting machine failures using ML model",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# ==========================
# Data Models
# ==========================

class MachineData(BaseModel):
    """Single machine data for prediction"""
    UDI: Optional[int] = None
    Product_ID: str = Field(..., description="Product ID")
    Type: str = Field(..., description="Machine Type (L, M, H)")
    Air_temperature_K: float = Field(..., description="Air temperature in Kelvin")
    Process_temperature_K: float = Field(..., description="Process temperature in Kelvin")
    Rotational_speed_rpm: float = Field(..., description="Rotational speed in RPM")
    Torque_Nm: float = Field(..., description="Torque in Nm")
    Tool_wear_min: float = Field(..., description="Tool wear in minutes")
    
    class Config:
        schema_extra = {
            "example": {
                "Product_ID": "M14860",
                "Type": "M",
                "Air_temperature_K": 298.1,
                "Process_temperature_K": 308.6,
                "Rotational_speed_rpm": 1551,
                "Torque_Nm": 42.8,
                "Tool_wear_min": 0
            }
        }

class BatchPredictionRequest(BaseModel):
    """Batch prediction request with multiple machines"""
    machines: List[MachineData]
    
    class Config:
        schema_extra = {
            "example": {
                "machines": [
                    {
                        "Product_ID": "M14860",
                        "Type": "M",
                        "Air_temperature_K": 298.1,
                        "Process_temperature_K": 308.6,
                        "Rotational_speed_rpm": 1551,
                        "Torque_Nm": 42.8,
                        "Tool_wear_min": 0
                    },
                    {
                        "Product_ID": "L47181",
                        "Type": "L",
                        "Air_temperature_K": 298.2,
                        "Process_temperature_K": 308.7,
                        "Rotational_speed_rpm": 1580,
                        "Torque_Nm": 38.9,
                        "Tool_wear_min": 210
                    }
                ]
            }
        }

class PredictionResult(BaseModel):
    """Single prediction result"""
    machine_id: Optional[int]
    product_id: str
    prediction: str
    confidence: float
    failure_probability: float
    normal_probability: float
    alert_level: str
    recommendation: str
    timestamp: str

class BatchPredictionResponse(BaseModel):
    """Batch prediction response"""
    predictions: List[PredictionResult]
    total_machines: int
    failures_predicted: int
    timestamp: str

class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    model_loaded: bool
    timestamp: str
    version: str

# ==========================
# Model Loader
# ==========================

class PredictiveMaintenanceModel:
    """Model loader and predictor class"""
    
    def __init__(self):
        self.model = None
        self.preprocessor = None
        self.is_loaded = False
        
    def load_model(self):
        """Load the trained model and preprocessor"""
        try:
            self.model = joblib.load("production_predictive_maintenance_model.pkl")
            self.preprocessor = joblib.load("preprocessor.pkl")
            self.is_loaded = True
            logger.info("✅ Model and preprocessor loaded successfully")
        except Exception as e:
            logger.error(f"❌ Error loading model: {e}")
            self.is_loaded = False
    
    def preprocess_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Preprocess incoming data with feature engineering"""
        df = data.copy()
        
        # Create additional features (same as training)
        df['temp_difference'] = df['Process_temperature_K'] - df['Air_temperature_K']
        df['power'] = df['Torque_Nm'] * df['Rotational_speed_rpm']
        df['wear_to_torque_ratio'] = df['Tool_wear_min'] / (df['Torque_Nm'] + 1)
        
        return df
    
    def predict(self, data: pd.DataFrame) -> tuple:
        """Make predictions"""
        if not self.is_loaded:
            raise ValueError("Model not loaded")
        
        # Preprocess data
        processed_data = self.preprocess_data(data)
        
        # Make predictions
        predictions = self.model.predict(processed_data)
        probabilities = self.model.predict_proba(processed_data)
        
        return predictions, probabilities

# Initialize model
model_predictor = PredictiveMaintenanceModel()

# ==========================
# API Endpoints
# ==========================

@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    model_predictor.load_model()

@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint"""
    return {
        "message": "Predictive Maintenance API",
        "description": "Machine Learning API for predicting equipment failures",
        "version": "1.0.0",
        "docs": "/docs"
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy" if model_predictor.is_loaded else "unhealthy",
        model_loaded=model_predictor.is_loaded,
        timestamp=datetime.now().isoformat(),
        version="1.0.0"
    )

@app.post("/predict", response_model=PredictionResult)
async def predict_single(machine_data: MachineData):
    """Predict failure for a single machine"""
    if not model_predictor.is_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Convert to DataFrame
        input_data = pd.DataFrame([machine_data.dict()])
        
        # Rename columns to match training data format
        column_mapping = {
            'Air_temperature_K': 'Air temperature [K]',
            'Process_temperature_K': 'Process temperature [K]', 
            'Rotational_speed_rpm': 'Rotational speed [rpm]',
            'Torque_Nm': 'Torque [Nm]',
            'Tool_wear_min': 'Tool wear [min]'
        }
        input_data = input_data.rename(columns=column_mapping)
        
        # Make prediction
        predictions, probabilities = model_predictor.predict(input_data)
        
        # Process results
        prediction = predictions[0]
        failure_prob = probabilities[0][1]
        normal_prob = probabilities[0][0]
        confidence = max(probabilities[0])
        
        # Determine alert level and recommendation
        if prediction == 1:
            alert_level = "HIGH"
            recommendation = "Immediate maintenance required. Schedule inspection ASAP."
        else:
            if failure_prob > 0.3:  # Warning threshold
                alert_level = "MEDIUM"
                recommendation = "Monitor closely. Schedule maintenance in near future."
            else:
                alert_level = "LOW" 
                recommendation = "Normal operation. Continue routine monitoring."
        
        return PredictionResult(
            machine_id=machine_data.UDI,
            product_id=machine_data.Product_ID,
            prediction="Failure" if prediction == 1 else "Normal",
            confidence=confidence,
            failure_probability=failure_prob,
            normal_probability=normal_prob,
            alert_level=alert_level,
            recommendation=recommendation,
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/predict-batch", response_model=BatchPredictionResponse)
async def predict_batch(batch_request: BatchPredictionRequest):
    """Predict failures for multiple machines"""
    if not model_predictor.is_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Convert to DataFrame
        machines_data = [machine.dict() for machine in batch_request.machines]
        input_data = pd.DataFrame(machines_data)
        
        # Rename columns to match training data format
        column_mapping = {
            'Air_temperature_K': 'Air temperature [K]',
            'Process_temperature_K': 'Process temperature [K]',
            'Rotational_speed_rpm': 'Rotational speed [rpm]',
            'Torque_Nm': 'Torque [Nm]', 
            'Tool_wear_min': 'Tool wear [min]'
        }
        input_data = input_data.rename(columns=column_mapping)
        
        # Make predictions
        predictions, probabilities = model_predictor.predict(input_data)
        
        # Process results
        results = []
        failure_count = 0
        
        for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
            failure_prob = prob[1]
            normal_prob = prob[0]
            confidence = max(prob)
            
            # Determine alert level and recommendation
            if pred == 1:
                alert_level = "HIGH"
                recommendation = "Immediate maintenance required"
                failure_count += 1
            else:
                if failure_prob > 0.3:
                    alert_level = "MEDIUM"
                    recommendation = "Monitor closely"
                else:
                    alert_level = "LOW"
                    recommendation = "Normal operation"
            
            results.append(
                PredictionResult(
                    machine_id=machines_data[i].get('UDI'),
                    product_id=machines_data[i]['Product_ID'],
                    prediction="Failure" if pred == 1 else "Normal",
                    confidence=confidence,
                    failure_probability=failure_prob,
                    normal_probability=normal_prob,
                    alert_level=alert_level,
                    recommendation=recommendation,
                    timestamp=datetime.now().isoformat()
                )
            )
        
        return BatchPredictionResponse(
            predictions=results,
            total_machines=len(results),
            failures_predicted=failure_count,
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")

@app.get("/model-info")
async def get_model_info():
    """Get information about the loaded model"""
    if not model_predictor.is_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        model = model_predictor.model
        model_type = type(model.named_steps['classifier']).__name__
        
        if hasattr(model.named_steps['classifier'], 'feature_importances_'):
            feature_importance = True
        else:
            feature_importance = False
        
        return {
            "model_type": model_type,
            "model_loaded": True,
            "has_feature_importance": feature_importance,
            "pipeline_steps": list(model.named_steps.keys()),
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting model info: {str(e)}")

# ==========================
# Main Execution
# ==========================

if __name__ == "__main__":
    uvicorn.run(
        "predictive_maintenance_api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )