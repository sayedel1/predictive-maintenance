from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator, model_validator
from pydantic_core import ValidationError
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
from typing import List, Optional, Dict, Any
import uvicorn
import logging

# -------------------------------------
# Setup Logging
# -------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -------------------------------------
# FastAPI Application
# -------------------------------------
app = FastAPI(
    title="Predictive Maintenance API",
    description="API for predicting machine failures using TensorFlow Lite with data validation",
    version="2.0.0"
)

# -------------------------------------
# CORS Middleware
# -------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------------------
# Global variables for loaded models
# -------------------------------------
preprocessor = None
interpreter = None
input_details = None
output_details = None

# -------------------------------------
# Configuration - REALISTIC RANGES
# -------------------------------------
VALID_RANGES = {
    "Air temperature [K]": (290.0, 310.0),      # ~17°C to 37°C (Realistic room temp)
    "Process temperature [K]": (300.0, 320.0),  # ~27°C to 47°C (Realistic process temp)
    "Rotational speed [rpm]": (1000.0, 3000.0), # Realistic RPM range
    "Torque [Nm]": (10.0, 80.0),               # Realistic torque range
    "Tool wear [min]": (0.0, 300.0)            # Realistic tool wear
}

# ABSOLUTE MAXIMUM RANGES (for critical errors)
ABSOLUTE_LIMITS = {
    "Air temperature [K]": (273.0, 500.0),     # 0°C to 227°C (Absolute min/max)
    "Process temperature [K]": (273.0, 600.0), # 0°C to 327°C
    "Rotational speed [rpm]": (0.0, 10000.0),
    "Torque [Nm]": (0.0, 500.0),
    "Tool wear [min]": (0.0, 1000.0)
}

# -------------------------------------
# Pydantic V2 Models with Validation
# -------------------------------------
class MaintenanceFeatures(BaseModel):
    Type: Optional[str] = Field(None, alias="Type")
    Air_temperature_K: Optional[float] = Field(None, alias="Air temperature [K]")
    Process_temperature_K: Optional[float] = Field(None, alias="Process temperature [K]")
    Rotational_speed_rpm: Optional[float] = Field(None, alias="Rotational speed [rpm]")
    Torque_Nm: Optional[float] = Field(None, alias="Torque [Nm]")
    Tool_wear_min: Optional[float] = Field(None, alias="Tool wear [min]")

    model_config = {
        "populate_by_name": True,
        "json_schema_extra": {
            "example": {
                "Type": "M",
                "Air temperature [K]": 298.5,
                "Process temperature [K]": 308.8,
                "Rotational speed [rpm]": 1450,
                "Torque [Nm]": 42.3,
                "Tool wear [min]": 45
            }
        }
    }

    @field_validator('Type')
    @classmethod
    def validate_type(cls, v):
        if v is not None and v.upper() not in ['L', 'M', 'H']:
            raise ValueError('Type must be L, M, or H')
        return v.upper() if v else v

    @field_validator('Air_temperature_K', 'Process_temperature_K', 'Rotational_speed_rpm', 'Torque_Nm', 'Tool_wear_min')
    @classmethod
    def validate_absolute_limits(cls, v, info):
        if v is None:
            return v
            
        field_name = info.field_name
        field_map = {
            'Air_temperature_K': 'Air temperature [K]',
            'Process_temperature_K': 'Process temperature [K]', 
            'Rotational_speed_rpm': 'Rotational speed [rpm]',
            'Torque_Nm': 'Torque [Nm]',
            'Tool_wear_min': 'Tool wear [min]'
        }
        
        display_name = field_map.get(field_name, field_name)
        
        if display_name in ABSOLUTE_LIMITS:
            min_val, max_val = ABSOLUTE_LIMITS[display_name]
            if not (min_val <= v <= max_val):
                raise ValueError(f'{display_name} ({v}) is outside possible physical range ({min_val}-{max_val})')
        
        return v

class PredictionResponse(BaseModel):
    prediction: float
    failure_probability: float
    prediction_class: int
    status: str
    warning: Optional[str] = None
    data_quality: str = "good"  # good, warning, error

class HealthResponse(BaseModel):
    status: str
    models_loaded: bool
    message: str

class ValidationResponse(BaseModel):
    valid: bool
    errors: List[str] = []
    warnings: List[str] = []
    data_quality: str = "good"

# -------------------------------------
# Load Models at Startup
# -------------------------------------
@app.on_event("startup")
async def load_models():
    """Load the preprocessor and TensorFlow Lite model"""
    global preprocessor, interpreter, input_details, output_details

    try:
        preprocessor = joblib.load('preprocessor.pkl')
        logger.info("✅ Preprocessor loaded successfully")

        interpreter = tf.lite.Interpreter(model_path='predictive_maintenance_model.tflite')
        interpreter.allocate_tensors()

        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        logger.info("✅ TensorFlow Lite model loaded successfully")
        logger.info(f"📋 Model Input: {input_details[0]['shape']}, {input_details[0]['dtype']}")
        logger.info(f"📋 Model Output: {output_details[0]['shape']}, {output_details[0]['dtype']}")

    except Exception as e:
        logger.error(f"❌ Error loading models: {str(e)}")
        raise e

# -------------------------------------
# Enhanced Helper Functions
# -------------------------------------
def validate_data_quality(features_dict: dict) -> Dict[str, Any]:
    """
    Validate data quality and return warnings/errors
    """
    warnings = []
    data_quality = "good"
    
    for field_name, (min_val, max_val) in VALID_RANGES.items():
        if field_name in features_dict and features_dict[field_name] is not None:
            value = features_dict[field_name]
            
            # Check if value is within realistic range
            if not (min_val <= value <= max_val):
                warning_msg = f"{field_name} ({value}) is outside realistic range ({min_val}-{max_val})"
                warnings.append(warning_msg)
                data_quality = "warning"
                
                # If value is extremely unrealistic, mark as error
                abs_min, abs_max = ABSOLUTE_LIMITS[field_name]
                if not (abs_min <= value <= abs_max):
                    data_quality = "error"
    
    # Special check for process temperature being much higher than air temperature
    if ('Air temperature [K]' in features_dict and 
        'Process temperature [K]' in features_dict and
        features_dict['Process temperature [K]'] > features_dict['Air temperature [K]'] + 50.0):
        warnings.append("Process temperature is unusually high compared to air temperature")
        data_quality = "warning"
    
    return {
        "warnings": warnings,
        "data_quality": data_quality,
        "is_realistic": data_quality == "good"
    }

def convert_to_preprocessor_format(features_dict: dict) -> dict:
    """Convert input features to match preprocessor's expected format"""
    converted = {}
    
    if features_dict.get("Type"):
        converted['Type'] = features_dict["Type"]
    if features_dict.get("Air temperature [K]") is not None:
        converted['Air temperature [K]'] = features_dict["Air temperature [K]"]
    if features_dict.get("Process temperature [K]") is not None:
        converted['Process temperature [K]'] = features_dict["Process temperature [K]"]
    if features_dict.get("Rotational speed [rpm]") is not None:
        converted['Rotational speed [rpm]'] = features_dict["Rotational speed [rpm]"]
    if features_dict.get("Torque [Nm]") is not None:
        converted['Torque [Nm]'] = features_dict["Torque [Nm]"]
    if features_dict.get("Tool wear [min]") is not None:
        converted['Tool wear [min]'] = features_dict["Tool wear [min]"]
    
    return converted

def preprocess_input(features_dict: dict) -> np.ndarray:
    """Preprocess input features"""
    try:
        if preprocessor is None:
            raise RuntimeError("Preprocessor not loaded")

        # Convert to preprocessor's expected format
        converted_features = convert_to_preprocessor_format(features_dict)
        
        # Check for missing required fields
        required_fields = ['Type', 'Air temperature [K]', 'Process temperature [K]', 
                          'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]']
        
        missing_fields = [field for field in required_fields if field not in converted_features]
        if missing_fields:
            raise HTTPException(status_code=422, detail=f"Missing required fields: {missing_fields}")

        # Create DataFrame and preprocess
        input_data = pd.DataFrame([converted_features])
        logger.info(f"🔧 Input data: {converted_features}")
        
        processed_data = preprocessor.transform(input_data)

        # Convert to dense array if sparse
        if hasattr(processed_data, 'toarray'):
            processed_data = processed_data.toarray()

        processed_data = processed_data.astype(np.float32)
        logger.info(f"🔧 Processed data shape: {processed_data.shape}")

        return processed_data

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error in preprocess_input: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Preprocessing error: {str(e)}")

def predict_with_tflite(processed_data: np.ndarray) -> float:
    """Make prediction using TensorFlow Lite model"""
    global interpreter, input_details, output_details

    try:
        if interpreter is None:
            raise RuntimeError("Interpreter not loaded")

        interpreter.set_tensor(input_details[0]['index'], processed_data)
        interpreter.invoke()
        prediction = interpreter.get_tensor(output_details[0]['index'])

        # Extract scalar value
        if prediction.ndim == 0:
            return float(prediction)
        else:
            return float(prediction.reshape(-1)[0])

    except Exception as e:
        logger.error(f"❌ Error in predict_with_tflite: {str(e)}")
        raise e

# -------------------------------------
# API Routes
# -------------------------------------
@app.get("/")
async def root():
    return {
        "message": "Predictive Maintenance API v2.0",
        "version": "2.0.0",
        "status": "active",
        "features": ["data_validation", "quality_check", "realistic_ranges"],
        "endpoints": {
            "docs": "/docs",
            "health": "/health",
            "predict": "/predict",
            "model-info": "/model-info",
            "validate-data": "/validate-data",
            "test-unrealistic": "/test-unrealistic-data"
        }
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    if preprocessor is not None and interpreter is not None:
        return HealthResponse(
            status="healthy",
            models_loaded=True,
            message="API is ready for predictions"
        )
    else:
        return HealthResponse(
            status="unhealthy",
            models_loaded=False,
            message="Models not loaded properly"
        )

@app.post("/validate-data", response_model=ValidationResponse)
async def validate_data(features: MaintenanceFeatures):
    """Validate data without making prediction"""
    try:
        features_dict = features.model_dump(by_alias=True)
        features_dict = {k: v for k, v in features_dict.items() if v is not None}
        
        quality_check = validate_data_quality(features_dict)
        
        return ValidationResponse(
            valid=True,
            warnings=quality_check["warnings"],
            data_quality=quality_check["data_quality"]
        )
        
    except ValidationError as e:
        return ValidationResponse(
            valid=False,
            errors=[str(err) for err in e.errors()],
            data_quality="error"
        )
    except Exception as e:
        return ValidationResponse(
            valid=False,
            errors=[str(e)],
            data_quality="error"
        )

@app.post("/predict", response_model=PredictionResponse)
async def predict(features: MaintenanceFeatures):
    """Main prediction endpoint with data quality checks"""
    try:
        # Convert to dictionary using Pydantic V2 method
        features_dict = features.model_dump(by_alias=True)
        features_dict = {k: v for k, v in features_dict.items() if v is not None}
        
        logger.info(f"🎯 Received prediction request: {features_dict}")
        
        # Validate data quality
        quality_check = validate_data_quality(features_dict)
        
        # If data quality is error, we might not want to proceed
        if quality_check["data_quality"] == "error":
            warning_msg = "Data contains unrealistic values. Prediction may not be reliable."
        elif quality_check["data_quality"] == "warning":
            warning_msg = "; ".join(quality_check["warnings"])
        else:
            warning_msg = None
        
        # Process and predict
        processed_data = preprocess_input(features_dict)
        prediction_prob = predict_with_tflite(processed_data)
        prediction_class = 1 if prediction_prob >= 0.5 else 0
        
        # Enhanced status based on data quality
        if quality_check["data_quality"] == "error":
            status = "Invalid input data - check values"
        elif quality_check["data_quality"] == "warning":
            status = f"{'Failure' if prediction_class == 1 else 'No failure'} predicted (data quality warning)"
        else:
            status = "Failure predicted" if prediction_class == 1 else "No failure predicted"
        
        logger.info(f"✅ Prediction completed: {prediction_prob:.6f} -> {status}")

        return PredictionResponse(
            prediction=prediction_prob,
            failure_probability=prediction_prob,
            prediction_class=prediction_class,
            status=status,
            warning=warning_msg,
            data_quality=quality_check["data_quality"]
        )

    except HTTPException:
        raise
    except Exception as e:
        error_msg = f"Prediction error: {str(e)}"
        logger.error(f"❌ {error_msg}")
        raise HTTPException(status_code=500, detail=error_msg)

@app.get("/model-info")
async def model_info():
    if interpreter is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    return {
        "model_type": "TensorFlow Lite",
        "input_shape": input_details[0]['shape'].tolist(),
        "input_type": str(input_details[0]['dtype']),
        "output_shape": output_details[0]['shape'].tolist(),
        "output_type": str(output_details[0]['dtype']),
        "note": "Model expects 8 features after preprocessing"
    }

@app.get("/test-unrealistic-data")
async def test_unrealistic_data():
    """Test endpoint for unrealistic data"""
    test_cases = [
        {
            "name": "Extremely High Temperature",
            "data": {
                "Type": "M",
                "Air temperature [K]": 308335464.8,  # Unrealistic
                "Process temperature [K]": 308.8,
                "Rotational speed [rpm]": 1450,
                "Torque [Nm]": 42.3,
                "Tool wear [min]": 45
            }
        },
        {
            "name": "Normal Data",
            "data": {
                "Type": "M",
                "Air temperature [K]": 298.5,
                "Process temperature [K]": 308.8,
                "Rotational speed [rpm]": 1450,
                "Torque [Nm]": 42.3,
                "Tool wear [min]": 45
            }
        },
        {
            "name": "Extreme Values",
            "data": {
                "Type": "M",
                "Air temperature [K]": 1000.0,  # Very hot
                "Process temperature [K]": 2000.0,  # Extremely hot
                "Rotational speed [rpm]": 50000,  # Very high RPM
                "Torque [Nm]": 1000.0,  # Very high torque
                "Tool wear [min]": 5000.0  # Very high wear
            }
        }
    ]
    
    results = []
    for test_case in test_cases:
        try:
            quality_check = validate_data_quality(test_case["data"])
            processed_data = preprocess_input(test_case["data"])
            prediction_prob = predict_with_tflite(processed_data)
            
            results.append({
                "test_case": test_case["name"],
                "input_data": test_case["data"],
                "data_quality": quality_check["data_quality"],
                "warnings": quality_check["warnings"],
                "prediction_probability": prediction_prob,
                "prediction_class": 1 if prediction_prob >= 0.5 else 0,
                "status": "Failure predicted" if prediction_prob >= 0.5 else "No failure predicted"
            })
        except Exception as e:
            results.append({
                "test_case": test_case["name"],
                "error": str(e)
            })
    
    return {"test_results": results}

# -------------------------------------
# Run the Application
# -------------------------------------
if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
