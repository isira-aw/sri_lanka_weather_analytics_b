from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from app.db.database import get_db, Weather, Location
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import pickle
import os

router = APIRouter()

# Global model storage
model_storage = {
    'precipitation_hours_model': None,
    'sunshine_model': None,
    'wind_speed_model': None,
    'scaler_params': None
}

def prepare_data(db: Session):
    """
    Prepare data for ML training
    
    Features:
    - temperature_2m_max, temperature_2m_min, temperature_2m_mean
    - apparent_temperature_max, apparent_temperature_min
    - daylight_duration
    - precipitation_sum
    - wind_gusts_10m_max
    - shortwave_radiation_sum
    - month, day_of_year
    
    Targets:
    - precipitation_hours
    - sunshine_duration
    - wind_speed_10m_max
    """
    query = db.query(Weather).filter(
        Weather.temperature_2m_max.isnot(None),
        Weather.precipitation_hours.isnot(None),
        Weather.sunshine_duration.isnot(None),
        Weather.wind_speed_10m_max.isnot(None)
    )
    
    df = pd.read_sql(query.statement, db.bind)
    
    if df.empty:
        return None, None, None
    
    # Convert date to datetime
    df['date'] = pd.to_datetime(df['date'])
    df['month'] = df['date'].dt.month
    df['day_of_year'] = df['date'].dt.dayofyear
    
    # Select features
    feature_columns = [
        'temperature_2m_max', 'temperature_2m_min', 'temperature_2m_mean',
        'apparent_temperature_max', 'apparent_temperature_min', 'apparent_temperature_mean',
        'daylight_duration', 'precipitation_sum', 'wind_gusts_10m_max',
        'shortwave_radiation_sum', 'month', 'day_of_year', 'location_id'
    ]
    
    target_columns = ['precipitation_hours', 'sunshine_duration', 'wind_speed_10m_max']
    
    # Drop rows with missing values
    df_clean = df[feature_columns + target_columns].dropna()
    
    X = df_clean[feature_columns]
    y = df_clean[target_columns]
    
    return X, y, df_clean

@router.post("/train")
async def train_models(db: Session = Depends(get_db)):
    """
    Train ML models to predict precipitation_hours, sunshine_duration, and wind_speed
    Goal: Predict conditions where evapotranspiration < 1.5mm in May
    """
    try:
        # Prepare data
        X, y, df_clean = prepare_data(db)
        
        if X is None:
            return {"error": "Insufficient data for training"}
        
        # Split data: 80% training, 20% validation
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        
        results = {
            "data_preparation": {
                "total_samples": len(X),
                "training_samples": len(X_train),
                "validation_samples": len(X_val),
                "features": list(X.columns),
                "targets": list(y.columns)
            },
            "models": {}
        }
        
        # Train separate models for each target
        for target in y.columns:
            # Model Selection: Random Forest Regressor
            # Reason: Handles non-linear relationships, robust to outliers, good for weather data
            model = RandomForestRegressor(
                n_estimators=100,
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            )
            
            # Train model
            model.fit(X_train, y_train[target])
            
            # Predictions
            y_pred_train = model.predict(X_train)
            y_pred_val = model.predict(X_val)
            
            # Evaluation metrics
            train_rmse = np.sqrt(mean_squared_error(y_train[target], y_pred_train))
            val_rmse = np.sqrt(mean_squared_error(y_val[target], y_pred_val))
            train_r2 = r2_score(y_train[target], y_pred_train)
            val_r2 = r2_score(y_val[target], y_pred_val)
            val_mae = mean_absolute_error(y_val[target], y_pred_val)
            
            # Feature importance
            feature_importance = dict(zip(X.columns, model.feature_importances_))
            top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:5]
            
            results["models"][target] = {
                "model_type": "Random Forest Regressor",
                "training_rmse": round(train_rmse, 4),
                "validation_rmse": round(val_rmse, 4),
                "training_r2": round(train_r2, 4),
                "validation_r2": round(val_r2, 4),
                "validation_mae": round(val_mae, 4),
                "top_5_features": [{"feature": f, "importance": round(imp, 4)} for f, imp in top_features]
            }
            
            # Store model
            model_storage[f'{target}_model'] = model
        
        # Store scaler params (mean and std for normalization)
        model_storage['scaler_params'] = {
            'mean': X_train.mean().to_dict(),
            'std': X_train.std().to_dict()
        }
        
        # Model explanation
        results["explanation"] = {
            "data_preparation": "Weather data was cleaned and features were extracted including temperature metrics, daylight duration, precipitation, wind, radiation, and temporal features (month, day of year).",
            "feature_selection": "Selected 13 features that directly impact weather patterns: temperature variables (max, min, mean, apparent), daylight duration, precipitation, wind gusts, shortwave radiation, and temporal indicators.",
            "model_choice": "Random Forest Regressor was chosen for its ability to capture non-linear relationships in weather data, handle multiple correlated features, and provide feature importance rankings. It's robust to outliers and doesn't require feature scaling.",
            "evaluation_metrics": "RMSE (Root Mean Squared Error) measures prediction accuracy in original units. R² score indicates how well the model explains variance (1.0 is perfect). MAE (Mean Absolute Error) shows average prediction error."
        }
        
        return results
        
    except Exception as e:
        return {"error": str(e)}

@router.post("/predict-may-2026")
async def predict_may_2026(db: Session = Depends(get_db)):
    """
    Predict weather conditions for May 2026 to achieve evapotranspiration < 1.5mm
    """
    try:
        if model_storage['precipitation_hours_model'] is None:
            return {"error": "Models not trained. Please train models first using /train endpoint"}
        
        # Get average May conditions from historical data
        query = db.query(Weather).filter(
            Weather.date >= '2020-05-01',
            Weather.date < '2024-06-01'
        )
        df = pd.read_sql(query.statement, db.bind)
        df['date'] = pd.to_datetime(df['date'])
        df_may = df[df['date'].dt.month == 5]
        
        if df_may.empty:
            return {"error": "No historical May data found"}
        
        # Get locations
        locations = db.query(Location).all()
        
        predictions_by_location = []
        
        for location in locations[:5]:  # Predict for first 5 locations
            # Create feature vector for May 2026
            # Use average historical May values as baseline
            may_features = df_may[df_may['location_id'] == location.location_id].mean()
            
            if pd.isna(may_features['temperature_2m_max']):
                continue
            
            # Prepare input features for May 15, 2026
            X_pred = pd.DataFrame([{
                'temperature_2m_max': may_features['temperature_2m_max'],
                'temperature_2m_min': may_features['temperature_2m_min'],
                'temperature_2m_mean': may_features['temperature_2m_mean'],
                'apparent_temperature_max': may_features['apparent_temperature_max'],
                'apparent_temperature_min': may_features['apparent_temperature_min'],
                'apparent_temperature_mean': may_features['apparent_temperature_mean'],
                'daylight_duration': may_features['daylight_duration'],
                'precipitation_sum': may_features['precipitation_sum'] * 0.7,  # Reduce precipitation for low ET
                'wind_gusts_10m_max': may_features['wind_gusts_10m_max'],
                'shortwave_radiation_sum': may_features['shortwave_radiation_sum'] * 0.8,  # Reduce radiation
                'month': 5,
                'day_of_year': 135,  # May 15
                'location_id': location.location_id
            }])
            
            # Make predictions
            pred_precip_hours = model_storage['precipitation_hours_model'].predict(X_pred)[0]
            pred_sunshine = model_storage['sunshine_model'].predict(X_pred)[0]
            pred_wind = model_storage['wind_speed_model'].predict(X_pred)[0]
            
            # Estimate evapotranspiration (simplified formula)
            # ET0 is influenced by temperature, radiation, wind, and humidity
            estimated_et0 = (
                0.0023 * (may_features['temperature_2m_mean'] + 17.8) * 
                np.sqrt(may_features['temperature_2m_max'] - may_features['temperature_2m_min']) *
                (may_features['shortwave_radiation_sum'] * 0.8) / 20
            )
            
            predictions_by_location.append({
                'location_id': location.location_id,
                'city_name': location.city_name,
                'date': '2026-05-15',
                'predicted_precipitation_hours': round(pred_precip_hours, 2),
                'predicted_sunshine_duration': round(pred_sunshine, 2),
                'predicted_wind_speed': round(pred_wind, 2),
                'estimated_et0': round(estimated_et0, 2),
                'target_et0': 1.5,
                'meets_target': estimated_et0 < 1.5,
                'input_features': {
                    'temperature_2m_max': round(may_features['temperature_2m_max'], 2),
                    'precipitation_sum_reduced': round(may_features['precipitation_sum'] * 0.7, 2),
                    'shortwave_radiation_reduced': round(may_features['shortwave_radiation_sum'] * 0.8, 2)
                }
            })
        
        return {
            "message": "Predictions for May 2026 with target evapotranspiration < 1.5mm",
            "prediction_date": "2026-05-15",
            "target": "Evapotranspiration < 1.5mm",
            "strategy": "Reduced precipitation and shortwave radiation inputs to achieve lower evapotranspiration",
            "predictions": predictions_by_location
        }
        
    except Exception as e:
        return {"error": str(e)}

@router.get("/model-info")
async def get_model_info():
    """Get information about trained models"""
    if model_storage['precipitation_hours_model'] is None:
        return {"status": "Models not trained", "trained": False}
    
    return {
        "status": "Models trained and ready",
        "trained": True,
        "models": [
            "precipitation_hours_model",
            "sunshine_duration_model", 
            "wind_speed_10m_max_model"
        ],
        "model_type": "Random Forest Regressor"
    }

@router.get("/feature-importance")
async def get_feature_importance(db: Session = Depends(get_db)):
    """Get feature importance from trained models"""
    if model_storage['precipitation_hours_model'] is None:
        return {"error": "Models not trained"}
    
    # Get feature names
    X, _, _ = prepare_data(db)
    if X is None:
        return {"error": "No data available"}
    
    feature_names = list(X.columns)
    
    importance_data = {}
    for target in ['precipitation_hours', 'sunshine_duration', 'wind_speed_10m_max']:
        model = model_storage[f'{target}_model']
        if model:
            importances = dict(zip(feature_names, model.feature_importances_))
            sorted_importances = sorted(importances.items(), key=lambda x: x[1], reverse=True)
            importance_data[target] = [
                {"feature": f, "importance": round(imp, 4)} 
                for f, imp in sorted_importances
            ]
    
    return {
        "message": "Feature importance for all models",
        "data": importance_data
    }
