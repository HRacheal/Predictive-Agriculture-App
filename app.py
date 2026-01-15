import streamlit as st
import datetime
import pandas as pd
import joblib
import os
import matplotlib.pyplot as plt

st.set_page_config(page_title="AgriPredict", page_icon="🌱", layout="wide")

# --- MODEL LOADING ---
MODEL_PATH = "crop_yield_model.pkl"

@st.cache_resource 
def load_active_model():
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)
    return None

model = load_active_model()

# --- SIDEBAR UI ---
with st.sidebar:
    if model:
        st.success(" Model Ready")
    
    st.header(" Field Conditions")
    crop_mapping = {"Wheat": 9, "Corn": 1, "Rice": 7, "Soybean": 8, "Cotton": 2}
    soil_mapping = {"Clay": 0, "Silt": 4, "Sandy": 3, "Loamy": 2}
    
    date = st.date_input("Observation Date", datetime.date.today())
    crop_name = st.selectbox("Select Crop", list(crop_mapping.keys()))
    soil_name = st.selectbox("Soil Type", list(soil_mapping.keys()))
    temp = st.slider("Temperature (°C)", -10, 50, 25)
    ph = st.slider("Soil pH", 4.0, 9.0, 6.5)
    hum = st.slider("Humidity (%)", 0, 100, 60)
    
    st.divider()
    n_val = st.number_input("Nitrogen (N)", 0.0, 200.0, 50.0)
    p_val = st.number_input("Phosphorus (P)", 0.0, 200.0, 40.0)
    k_val = st.number_input("Potassium (K)", 0.0, 200.0, 30.0)
    sq_val = st.slider("Soil Quality Index", 0.0, 100.0, 80.0)

# --- MAIN DASHBOARD ---
st.title("🌾 Smart Farm: Yield Prediction Dashboard")

if st.button(" Analyze & Generate Prediction"):
    if model:
        # Prepare Feature Data
        features = ["Crop_Type", "Soil_Type", "Soil_pH", "Temperature", "Humidity", 
                    "Wind_Speed", "N", "P", "K", "Soil_Quality", "month", "year"]
        
        input_df = pd.DataFrame([[
            crop_mapping[crop_name], soil_mapping[soil_name], ph, temp, float(hum),
            12.0, n_val, p_val, k_val, sq_val, date.month, date.year
        ]], columns=features)
        
        try:
            prediction = model.predict(input_df)[0]
            
            # Layout Columns
            col_text, col_chart = st.columns([1, 1.2])
            
            with col_text:
                st.subheader("Predicted Yield")
                st.title(f"{prediction:.2f} MT/Ha")
                
                # --- YOUR CUSTOM STATUS ALERT ---
                if prediction < 25.0: # Set threshold for 'critically low'
                    st.error("Status: ALERT: Predicted yield is critically low! Check soil nutrients.")
                else:
                    st.success("Status: SUCCESS: Predicted yield is within a healthy range.")

            with col_chart:
                st.subheader(" Factor Impact Analysis")
                if hasattr(model, 'feature_importances_'):
                    importances = model.feature_importances_
                    feat_imp = pd.Series(importances, index=features).sort_values(ascending=True)
                    
                    fig, ax = plt.subplots(figsize=(8, 5))
                    feat_imp.tail(5).plot(kind='barh', ax=ax, color='#2e7d32')
                    ax.set_title("Top 5 Impacting Factors")
                    st.pyplot(fig)
                else:
                    st.info("Factor analysis is currently unavailable.")
                    
        except Exception as e:
            st.error(f"Prediction Error: {e}")
