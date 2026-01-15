from fastapi import FastAPI
import joblib
import pandas as pd
from pydantic import BaseModel

# --- 1. Initialize FastAPI ---
app = FastAPI(title="AgriPredict API")

# --- 2. Load trained model with Memory Mapping ---
model = joblib.load('crop_yield_model.pkl', mmap_mode='r')

# --- 3. Define input schema ---
class FarmData(BaseModel):
    Crop_Type: int
    Soil_Type: int
    Soil_pH: float
    Temperature: float
    Humidity: float
    Wind_Speed: float
    N: float
    P: float
    K: float
    Soil_Quality: float
    month: int
    year: int

# --- 4. Prediction endpoint ---
@app.post("/predict")
def get_prediction(data: FarmData):
    try:
        # Prepare input DataFrame (Ensure keys match training column names)
        input_dict = {
            'Crop_Type': int(data.Crop_Type),
            'Soil_Type': int(data.Soil_Type),
            'Soil_pH': float(data.Soil_pH),
            'Temperature': float(data.Temperature),
            'Humidity': float(data.Humidity),
            'Wind_Speed': float(data.Wind_Speed),
            'N': float(data.N),
            'P': float(data.P),
            'K': float(data.K),
            'Soil_Quality': float(data.Soil_Quality),
            'month': int(data.month),
            'year': int(data.year)
        }
        
        input_df = pd.DataFrame([input_dict])

        # Make prediction
        raw_prediction = model.predict(input_df)[0]

        # --- FEATURE IMPORTANCE EXTRACTION ---
        # Get importance scores from the RandomForest model
        importances = model.feature_importances_
        feature_names = input_df.columns.tolist()
        
        # Convert NumPy floats to Python floats for JSON compatibility
        fi_dict = {name: float(imp) for name, imp in zip(feature_names, importances)}

        # --- DATA CONVERSION ---
        final_prediction = float(raw_prediction)
        threshold = 0.15 * 136.711982
        alert_status = bool(final_prediction < threshold)

        message = "Yield is within healthy range." if not alert_status else \
                  "ALERT: Predicted yield is critically low! Check soil nutrients."

        return {
            "predicted_yield": round(final_prediction, 2),
            "unit": "Metric Tons per Hectare",
            "low_yield_alert": alert_status,
            "recommendation": message,
            "feature_importance": fi_dict  # <--- Dashboard now gets this data
        }

    except Exception as e:
        return {"error": str(e)}