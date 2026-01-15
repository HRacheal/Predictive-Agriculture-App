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
        st.success("✅ System Online: Model Ready")
    
    st.header("📍 Field Conditions")
    date = st.date_input("Observation Date", datetime.date.today())
    crop_mapping = {"Wheat": 9, "Corn": 1, "Rice": 7, "Soybean": 8, "Cotton": 2}
    soil_mapping = {"Clay": 0, "Silt": 4, "Sandy": 3, "Loamy": 2}
    
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

if st.button("🚀 Analyze & Generate Prediction"):
    if model:
        # Prepare Feature Data
        features = ["Crop_Type", "Soil_Type", "Soil_pH", "Temperature", "Humidity", 
                    "Wind_Speed", "N", "P", "K", "Soil_Quality", "month", "year"]
        
        input_data = [
            crop_mapping[crop_name], soil_mapping[soil_name], ph, temp, float(hum),
            12.0, n_val, p_val, k_val, sq_val, date.month, date.year
        ]
        input_df = pd.DataFrame([input_data], columns=features)
        
        try:
            prediction = model.predict(input_df)[0]
            
            # Create Layout Columns
            col_text, col_chart = st.columns([1, 1.2])
            
            with col_text:
                st.subheader("Predicted Yield")
                st.title(f"{prediction:.2f} MT/Ha")
                
                # --- UPDATED STATUS MESSAGES ---
                if prediction < 20.0:
                    st.error(f"**Status:** ALERT: Predicted yield is critically low! Check soil nutrients (N, P, K) and pH balance.")
                elif 20.0 <= prediction < 30.0:
                    st.warning(f"**Status:** ATTENTION: Predicted yield is below average. Consider optimizing irrigation or fertilizer timing.")
                else:
                    st.success(f"**Status:** SUCCESS: Predicted yield is within healthy range. Conditions are optimal for {crop_name}.")

            with col_chart:
                st.subheader("📊 Factor Impact Analysis")
                if hasattr(model, 'feature_importances_'):
                    importances = model.feature_importances_
                    feat_imp = pd.Series(importances, index=features).sort_values(ascending=True)
                    
                    # Create the Bar Chart
                    fig, ax = plt.subplots(figsize=(8, 6))
                    feat_imp.plot(kind='barh', ax=ax, color='#2e7d32')
                    ax.set_title("Which factors affected this prediction most?")
                    st.pyplot(fig)
                else:
                    st.info("Factor analysis is only available for tree-based models.")
                    
        except Exception as e:
            st.error(f"Prediction Error: {e}")
