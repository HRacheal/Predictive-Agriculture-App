import streamlit as st
import requests
import datetime
import pandas as pd

st.set_page_config(page_title="AgriPredict", page_icon="🌱", layout="wide")

# Custom CSS to make it look professional
st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
    .stButton>button { width: 100%; border-radius: 5px; height: 3em; background-color: #2e7d32; color: white; }
    </style>
    """, unsafe_allow_html=True)

st.title("🌾 Smart Farm: Yield Prediction Dashboard")

# 1. Mappings
crop_mapping = {"Wheat": 9, "Corn": 1, "Rice": 7, "Soybean": 8, "Cotton": 2}
soil_mapping = {"Clay": 0, "Silt": 4, "Sandy": 3, "Loamy": 2}

# Sidebar Inputs
with st.sidebar:
    st.header("📍 Field Conditions")
    date = st.date_input("Observation Date", datetime.date.today())
    crop_name = st.selectbox("Select Crop", list(crop_mapping.keys()))
    soil_name = st.selectbox("Soil Type", list(soil_mapping.keys()))
    
    st.divider()
    st.subheader("🧪 Soil & Environment")
    temp = st.slider("Temperature (°C)", -10, 50, 25)
    ph = st.slider("Soil pH", 4.0, 9.0, 6.5)
    hum = st.slider("Humidity (%)", 0, 100, 60)
    
    st.divider()
    st.subheader("🧬 Nutrients")
    n_val = st.number_input("Nitrogen (N)", 0.0, 200.0, 50.0)
    p_val = st.number_input("Phosphorus (P)", 0.0, 200.0, 40.0)
    k_val = st.number_input("Potassium (K)", 0.0, 200.0, 30.0)
    sq_val = st.slider("Soil Quality Index", 0.0, 100.0, 80.0)

# Main Dashboard Area
col1, col2 = st.columns([1, 1])

if st.button("🚀 Analyze & Generate Prediction"):
    payload = {
        "Crop_Type": crop_mapping[crop_name],
        "Soil_Type": soil_mapping[soil_name],
        "Soil_pH": ph,
        "Temperature": temp,
        "Humidity": float(hum),
        "Wind_Speed": 12.0,
        "N": n_val, 
        "P": p_val, 
        "K": k_val,
        "Soil_Quality": sq_val,
        "month": date.month,
        "year": date.year
    }
    
    try:
        response = requests.post("http://127.0.0.1:8000/predict", json=payload)
        result = response.json()
        
        with col1:
            st.metric(label="Predicted Yield", value=f"{result['predicted_yield']} MT/Ha")
            
            if result['low_yield_alert']:
                st.error(f"**Status:** {result['recommendation']}")
            else:
                st.success(f"**Status:** {result['recommendation']}")

        with col2:
            if "feature_importance" in result:
                st.subheader("📊 Factor Impact Analysis")
                fi_df = pd.DataFrame({
                    "Factor": list(result['feature_importance'].keys()),
                    "Influence": list(result['feature_importance'].values())
                }).sort_values(by="Influence", ascending=True)
                
                # Filter out month/year for a cleaner chart if desired
                st.bar_chart(fi_df.set_index("Factor"))

    except Exception as e:
        st.error(f"Connection Error: {e}")

else:
    st.info("👈 Adjust field conditions in the sidebar and click the button to see the prediction.")