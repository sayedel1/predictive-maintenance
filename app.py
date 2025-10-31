# ==========================
# 1. Import Libraries
# ==========================
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
import joblib
from typing import List, Optional
import uvicorn
from datetime import datetime
import os
import time

# ==========================
# 2. Initialize FastAPI App
# ==========================
app = FastAPI(
    title="Predictive Maintenance API",
    description="Advanced API for predicting machine failures and maintenance needs using ML",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# ==========================
# 3. Define Data Models
# ==========================
class MachineData(BaseModel):
    Type: str
    Air_temperature_K: float
    Process_temperature_K: float
    Rotational_speed_rpm: float
    Torque_Nm: float
    Tool_wear_min: float

class PredictionRequest(BaseModel):
    machines: List[MachineData]

class PredictionResponse(BaseModel):
    machine_id: int
    status: str
    prediction: str
    confidence: float
    alert_level: str
    maintenance_needed: bool
    recommendation: str
    timestamp: str

class BatchPredictionResponse(BaseModel):
    predictions: List[PredictionResponse]
    total_machines: int
    machines_need_maintenance: int
    overall_risk_level: str
    maintenance_percentage: float

class HealthCheck(BaseModel):
    status: str
    model_loaded: bool
    timestamp: str
    version: str

class APIInfo(BaseModel):
    name: str
    version: str
    status: str
    endpoints: list
    documentation: str

# ==========================
# 4. Global Variables
# ==========================
model = None
preprocessor = None
model_loaded = False

# ==========================
# 5. Load Model on Startup with Retries
# ==========================
@app.on_event("startup")
async def load_model():
    global model, preprocessor, model_loaded
    max_retries = 3
    retry_delay = 2  # seconds
    
    for attempt in range(max_retries):
        try:
            print(f"🔄 Loading model... Attempt {attempt + 1}/{max_retries}")
            model = joblib.load("production_predictive_maintenance_model.pkl")
            preprocessor = joblib.load("preprocessor.pkl")
            model_loaded = True
            print("✅ Model and preprocessor loaded successfully!")
            break
        except Exception as e:
            print(f"❌ Attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                print(f"⏳ Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
            else:
                print("💥 All attempts to load model failed!")
                model_loaded = False

# ==========================
# 6. Helper Functions
# ==========================
def create_features(input_data: pd.DataFrame) -> pd.DataFrame:
    """Create additional engineered features"""
    input_data = input_data.copy()
    input_data['temp_difference'] = input_data['Process temperature [K]'] - input_data['Air temperature [K]']
    input_data['power'] = input_data['Torque [Nm]'] * input_data['Rotational speed [rpm]']
    input_data['wear_to_torque_ratio'] = input_data['Tool wear [min]'] / (input_data['Torque [Nm]'] + 1e-5)
    return input_data

def get_recommendation(prediction: int, confidence: float, tool_wear: float) -> str:
    """Generate maintenance recommendations"""
    if prediction == 1:
        if confidence > 0.8:
            return "Immediate maintenance required! High failure risk detected."
        else:
            return "Schedule maintenance soon. Moderate failure risk."
    else:
        if tool_wear > 150:
            return "Monitor closely. High tool wear detected."
        elif confidence < 0.7:
            return "Regular monitoring recommended."
        else:
            return "Normal operation. Continue routine checks."

def get_alert_level(prediction: int, confidence: float) -> str:
    """Determine alert level based on prediction and confidence"""
    if prediction == 1:
        if confidence > 0.8:
            return "CRITICAL"
        elif confidence > 0.6:
            return "HIGH"
        else:
            return "MEDIUM"
    else:
        if confidence > 0.8:
            return "LOW"
        else:
            return "MONITOR"

# ==========================
# 7. API Routes
# ==========================
@app.get("/", response_model=APIInfo)
async def root():
    """Root endpoint with API information"""
    return {
        "name": "Predictive Maintenance API",
        "version": "2.0.0",
        "status": "active",
        "endpoints": [
            {"path": "/docs", "method": "GET", "description": "Interactive API documentation"},
            {"path": "/health", "method": "GET", "description": "API health check"},
            {"path": "/predict", "method": "POST", "description": "Single machine prediction"},
            {"path": "/predict-batch", "method": "POST", "description": "Batch predictions"},
            {"path": "/model-info", "method": "GET", "description": "Model information"},
            {"path": "/example-request", "method": "GET", "description": "Example requests"}
        ],
        "documentation": "Visit /docs for interactive API testing"
    }

@app.get("/health", response_model=HealthCheck)
async def health_check():
    """Check API health and model status"""
    return {
        "status": "healthy" if model_loaded else "degraded",
        "model_loaded": model_loaded,
        "timestamp": datetime.now().isoformat(),
        "version": "2.0.0"
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict_single(machine_data: MachineData):
    """
    Predict maintenance need for a single machine
    
    - **Type**: Machine type (L, M, H)
    - **Air_temperature_K**: Air temperature in Kelvin
    - **Process_temperature_K**: Process temperature in Kelvin  
    - **Rotational_speed_rpm**: Rotational speed in RPM
    - **Torque_Nm**: Torque in Newton-meters
    - **Tool_wear_min**: Tool wear in minutes
    """
    if not model_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded. Please try again later.")
    
    try:
        # Convert to DataFrame with correct column names
        input_data = pd.DataFrame([{
            'Type': machine_data.Type,
            'Air temperature [K]': machine_data.Air_temperature_K,
            'Process temperature [K]': machine_data.Process_temperature_K,
            'Rotational speed [rpm]': machine_data.Rotational_speed_rpm,
            'Torque [Nm]': machine_data.Torque_Nm,
            'Tool wear [min]': machine_data.Tool_wear_min
        }])
        
        # Create additional features
        input_data = create_features(input_data)
        
        # Make prediction
        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0]
        
        # Prepare response
        if prediction == 1:
            status = "FAILURE_RISK"
            prediction_text = "Maintenance Required"
            maintenance_needed = True
            confidence = probability[1]
        else:
            status = "NORMAL"
            prediction_text = "Normal Operation"
            maintenance_needed = False
            confidence = probability[0]
        
        alert_level = get_alert_level(prediction, confidence)
        recommendation = get_recommendation(prediction, confidence, machine_data.Tool_wear_min)
        
        return PredictionResponse(
            machine_id=1,
            status=status,
            prediction=prediction_text,
            confidence=round(confidence, 4),
            alert_level=alert_level,
            maintenance_needed=maintenance_needed,
            recommendation=recommendation,
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.post("/predict-batch", response_model=BatchPredictionResponse)
async def predict_batch(request: PredictionRequest):
    """
    Predict maintenance needs for multiple machines at once
    
    Provide a list of machine data objects for batch processing
    """
    if not model_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded. Please try again later.")
    
    try:
        predictions = []
        machines_need_maintenance = 0
        
        for i, machine_data in enumerate(request.machines, 1):
            # Convert to DataFrame format
            input_data = pd.DataFrame([{
                'Type': machine_data.Type,
                'Air temperature [K]': machine_data.Air_temperature_K,
                'Process temperature [K]': machine_data.Process_temperature_K,
                'Rotational speed [rpm]': machine_data.Rotational_speed_rpm,
                'Torque [Nm]': machine_data.Torque_Nm,
                'Tool wear [min]': machine_data.Tool_wear_min
            }])
            
            # Create additional features
            input_data = create_features(input_data)
            
            # Make prediction
            prediction = model.predict(input_data)[0]
            probability = model.predict_proba(input_data)[0]
            
            # Determine response
            if prediction == 1:
                status = "FAILURE_RISK"
                prediction_text = "Maintenance Required"
                maintenance_needed = True
                confidence = probability[1]
                machines_need_maintenance += 1
            else:
                status = "NORMAL"
                prediction_text = "Normal Operation"
                maintenance_needed = False
                confidence = probability[0]
            
            alert_level = get_alert_level(prediction, confidence)
            recommendation = get_recommendation(prediction, confidence, machine_data.Tool_wear_min)
            
            predictions.append(
                PredictionResponse(
                    machine_id=i,
                    status=status,
                    prediction=prediction_text,
                    confidence=round(confidence, 4),
                    alert_level=alert_level,
                    maintenance_needed=maintenance_needed,
                    recommendation=recommendation,
                    timestamp=datetime.now().isoformat()
                )
            )
        
        # Determine overall risk level
        total_machines = len(request.machines)
        maintenance_percentage = (machines_need_maintenance / total_machines) * 100
        
        if maintenance_percentage > 30:
            overall_risk = "HIGH"
        elif maintenance_percentage > 10:
            overall_risk = "MEDIUM"
        else:
            overall_risk = "LOW"
        
        return BatchPredictionResponse(
            predictions=predictions,
            total_machines=total_machines,
            machines_need_maintenance=machines_need_maintenance,
            overall_risk_level=overall_risk,
            maintenance_percentage=round(maintenance_percentage, 2)
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch prediction error: {str(e)}")

@app.get("/model-info")
async def get_model_info():
    """Get detailed information about the loaded ML model"""
    if not model_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        model_type = type(model.named_steps['classifier']).__name__
        
        # Get feature information
        numeric_features = model.named_steps['preprocessor'].transformers_[0][2]
        categorical_features = model.named_steps['preprocessor'].transformers_[1][1].get_feature_names_out().tolist()
        all_features = numeric_features + categorical_features
        
        feature_importance = None
        if hasattr(model.named_steps['classifier'], 'feature_importances_'):
            importances = model.named_steps['classifier'].feature_importances_
            # Create sorted feature importance list
            feature_importance = [
                {"feature": feature, "importance": round(importance, 4)}
                for feature, importance in zip(all_features, importances)
            ]
            feature_importance.sort(key=lambda x: x["importance"], reverse=True)
        
        return {
            "model_type": model_type,
            "model_loaded": model_loaded,
            "total_features": len(all_features),
            "feature_importance": feature_importance[:10] if feature_importance else None,  # Top 10 features
            "timestamp": datetime.now().isoformat(),
            "version": "2.0.0"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting model info: {str(e)}")

@app.get("/example-request")
async def get_example_request():
    """Get example request structures for testing the API"""
    return {
        "single_prediction": {
            "endpoint": "POST /predict",
            "example_body": {
                "Type": "M",
                "Air_temperature_K": 298.1,
                "Process_temperature_K": 308.6,
                "Rotational_speed_rpm": 1551,
                "Torque_Nm": 42.8,
                "Tool_wear_min": 0
            },
            "description": "Predict maintenance need for a single machine"
        },
        "batch_prediction": {
            "endpoint": "POST /predict-batch", 
            "example_body": {
                "machines": [
                    {
                        "Type": "L",
                        "Air_temperature_K": 298.0,
                        "Process_temperature_K": 308.0,
                        "Rotational_speed_rpm": 1500,
                        "Torque_Nm": 40.0,
                        "Tool_wear_min": 50
                    },
                    {
                        "Type": "H",
                        "Air_temperature_K": 302.0,
                        "Process_temperature_K": 315.0, 
                        "Rotational_speed_rpm": 2500,
                        "Torque_Nm": 60.0,
                        "Tool_wear_min": 200
                    },
                    {
                        "Type": "M",
                        "Air_temperature_K": 299.5,
                        "Process_temperature_K": 310.0,
                        "Rotational_speed_rpm": 1800,
                        "Torque_Nm": 45.0,
                        "Tool_wear_min": 120
                    }
                ]
            },
            "description": "Predict maintenance needs for multiple machines"
        }
    }

@app.get("/stats")
async def get_api_stats():
    """Get API usage statistics and performance metrics"""
    return {
        "api_status": "running",
        "model_status": "loaded" if model_loaded else "not_loaded",
        "startup_time": "on_startup",
        "endpoints_available": 6,
        "supported_operations": ["single_prediction", "batch_prediction", "health_check", "model_info"],
        "timestamp": datetime.now().isoformat()
    }

# ==========================
# 8. Railway & Production Configuration
# ==========================
def get_port():
    """Get port from environment variable for cloud compatibility"""
    return int(os.environ.get("PORT", 8000))

def start_server():
    """Start the server with production settings"""
    port = get_port()
    
    print("🚀" * 50)
    print("🤖 Predictive Maintenance API Starting...")
    print(f"📊 Model Loaded: {model_loaded}")
    print(f"🌐 Port: {port}")
    print(f"📚 Docs: http://0.0.0.0:{port}/docs")
    print(f"❤️  Health: http://0.0.0.0:{port}/health")
    print("🚀" * 50)
    
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=port,
        reload=False,  # Disable in production
        access_log=True,
        log_level="info"
    )

if __name__ == "__main__":
    start_server()
