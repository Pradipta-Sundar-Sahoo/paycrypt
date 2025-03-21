from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
from typing import Optional, List, Dict, Any
import os
from datetime import datetime
import logging
from openai import OpenAI

# Initialize logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

app = FastAPI(title="Toll Plaza Management API",
             description="API for toll plaza predictions and optimizations",
             version="1.0.0")

# Global variables for models and components
models = {}
openai_client = OpenAI(api_key=os.getenv("OPENAI_KEY", "default-key"))

class TrafficPredictionInput(BaseModel):
    hour: int
    day_of_week: str
    selected_merchant: str
    selected_vehicle: str
    lag_1: int
    lag_2: int
    lag_3: int
    lag_4: int
    lag_5: int
    rolling_mean_3: float
    rolling_std_3: float
    rolling_mean_6: float
    rolling_min_6: float
    rolling_max_6: float
    tx_diff_1: int
    tx_diff_2: int
    tx_per_second: float
    avg_amount_per_tx: float

class VehicleClassInput(BaseModel):
    hour: int
    day_of_week: str
    selected_merchant: str
    temperature: Optional[float] = None
    precipitation: Optional[float] = None
    direction_n: Optional[float] = None
    direction_s: Optional[float] = None
    direction_e: Optional[float] = None
    direction_w: Optional[float] = None

class AnomalyDetectionInput(BaseModel):
    transaction_count: float
    hour: int
    day_of_week: str
    is_weekend: bool
    month: int
    day: int

class LaneOptimizationInput(BaseModel):
    merchant_name: str
    hour: int
    expected_traffic: Optional[int] = None

@app.on_event("startup")
async def load_models_and_data():
    """Load ML models and preprocessing artifacts"""
    try:
        # Load dataset
        df = pd.read_csv('data/preprocessed_tollplaza_data.csv')
        df['initiated_time'] = pd.to_datetime(df['initiated_time'])
        models['dataset'] = df

        # Load ML models
        models.update({
            'traffic_model': joblib.load('models/traffic_prediction_model.pkl'),
            'anomaly_model': joblib.load('models/traffic_anomaly_models.pkl')[0],
            'anomaly_scaler': joblib.load('models/traffic_anomaly_models.pkl')[1],
            'vc_model': joblib.load('models/vehicle_class_prediction_model.pkl'),
            'vehicle_classifier': joblib.load('models/vehicle_behavior_classifier.pkl'),
            'lane_optimizer': EnhancedLaneOptimizationSystem(df, openai_client)
        })
        logger.info("All models loaded successfully")
        
    except Exception as e:
        logger.error(f"Error during startup: {str(e)}")
        raise HTTPException(status_code=500, detail="Initialization failed")

def preprocess_traffic_input(data: TrafficPredictionInput) -> pd.DataFrame:
    """Replicate Streamlit's feature engineering logic"""
    # Cyclical feature encoding
    hour_sin = np.sin(2 * np.pi * data.hour / 24)
    hour_cos = np.cos(2 * np.pi * data.hour / 24)
    day_num = ["Monday", "Tuesday", "Wednesday", "Thursday", 
              "Friday", "Saturday", "Sunday"].index(data.day_of_week)
    day_sin = np.sin(2 * np.pi * day_num / 7)
    day_cos = np.cos(2 * np.pi * day_num / 7)
    
    # Merchant encoding
    merchant_features = {}
    all_merchants = ["ATTIBELLE ", "Devanahalli Toll Plaza", 
                    "ELECTRONIC  CITY Phase 1", 
                    "Banglaore-Nelamangala Plaza", 
                    "Hoskote Toll Plaza"]
    
    for merchant in all_merchants:
        merchant_key = f"merchant_{merchant.replace(' ', '_')}"
        merchant_features[merchant_key] = 100 if merchant == data.selected_merchant else 0
    
    # Vehicle class encoding
    vehicle_classes = ["VC4", "VC5", "VC6", "VC7", "VC8", "VC9",
                      "VC10", "VC11", "VC12", "VC13", "VC14", "VC15", "VC16", "VC20"]
    vehicle_features = {f"vc_{vc}": 100 if vc == data.selected_vehicle else 0 
                       for vc in vehicle_classes}
    
    # Create feature dictionary
    features = {
        'hour': data.hour,
        'day_of_week': day_num,
        'is_weekend': int(data.day_of_week in ["Saturday", "Sunday"]),
        'quarter_of_day': data.hour // 6,
        'hour_sin': hour_sin,
        'hour_cos': hour_cos,
        'day_sin': day_sin,
        'day_cos': day_cos,
        'lag_1': data.lag_1,
        'lag_2': data.lag_2,
        'lag_3': data.lag_3,
        'lag_4': data.lag_4,
        'lag_5': data.lag_5,
        'rolling_mean_3': data.rolling_mean_3,
        'rolling_std_3': data.rolling_std_3,
        'rolling_mean_6': data.rolling_mean_6,
        'rolling_max_6': data.rolling_max_6,
        'rolling_min_6': data.rolling_min_6,
        'tx_diff_1': data.tx_diff_1,
        'tx_diff_2': data.tx_diff_2,
        'tx_per_second': data.tx_per_second,
        'avg_amount_per_tx': data.avg_amount_per_tx,
        **merchant_features,
        **vehicle_features
    }
    
    return pd.DataFrame([features])

@app.post("/predict/traffic", response_model=Dict[str, float])
async def predict_traffic(data: TrafficPredictionInput):
    """Endpoint for traffic volume prediction"""
    try:
        features = preprocess_traffic_input(data)
        model = models['traffic_model']
        prediction = model.predict(features)
        return {"predicted_traffic": float(prediction[0])}
    except Exception as e:
        logger.error(f"Traffic prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/detect/anomalies", response_model=Dict[str, Any])
async def detect_anomalies(data: AnomalyDetectionInput):
    """Endpoint for traffic pattern anomaly detection"""
    try:
        # Convert to numerical features
        day_num = ["Monday", "Tuesday", "Wednesday", "Thursday",
                  "Friday", "Saturday", "Sunday"].index(data.day_of_week)
        
        features = np.array([[
            data.transaction_count,
            data.hour,
            day_num,
            int(data.is_weekend),
            data.month,
            data.day
        ]])
        
        # Scale features
        scaled_features = models['anomaly_scaler'].transform(features)
        
        # Get predictions
        if_model = models['anomaly_model']['isolation_forest']
        dbscan = models['anomaly_model']['dbscan']
        
        if_pred = if_model.predict(scaled_features)
        dbscan_pred = dbscan.fit_predict(scaled_features)
        
        # Determine final anomaly status
        is_anomaly = if_pred[0] == -1 or dbscan_pred[0] == -1
        
        return {
            "is_anomaly": bool(is_anomaly),
            "isolation_forest_pred": int(if_pred[0]),
            "dbscan_pred": int(dbscan_pred[0]),
            "anomaly_score": float(if_model.score_samples(scaled_features)[0])
        }
    except Exception as e:
        logger.error(f"Anomaly detection error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/optimize/lanes", response_model=Dict[str, Any])
async def optimize_lanes(data: LaneOptimizationInput):
    """Endpoint for lane optimization recommendations"""
    try:
        optimizer = models['lane_optimizer']
        recommendations = optimizer.get_lane_recommendations(
            data.merchant_name,
            data.hour,
            data.expected_traffic
        )
        return recommendations
    except Exception as e:
        logger.error(f"Lane optimization error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate/insights", response_model=Dict[str, str])
async def generate_insights():
    """Endpoint for automated insights generation"""
    try:
        generator = AutomatedInsightsGenerator(models['dataset'], openai_client)
        report = generator.generate_insights_report()
        return {"insights": report['narrative']}
    except Exception as e:
        logger.error(f"Insights generation error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)