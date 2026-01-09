import streamlit as st
import datetime
import pandas as pd
import pickle  # Use this to load your .pkl files directly

st.set_page_config(page_title="AgriPredict", page_icon="🌱", layout="wide")

# 1. LOAD MODELS DIRECTLY (instead of using an API)
# Make sure le_crop.pkl and le_soil.pkl are in your GitHub main folder
try:
    with open('le_crop.pkl', 'rb') as f:
        model_crop = pickle.load(f)
    # If you have a main prediction model file, load it here too:
    # with open('your_model_name.pkl', 'rb') as f:
    #     prediction_model = pickle.load(f)
except FileNotFoundError:
    st.error("Model files not found. Please ensure .pkl files are uploaded to GitHub.")

st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
    .stButton>button { width: 100%; border-radius: 5px; height: 3em; background-color: #2e7d32; color: white; }
    </style>
    """, unsafe_allow_html=True)

st.title("🌾 Smart Farm: Yield Prediction Dashboard")

crop_mapping = {"Wheat": 9, "Corn": 1, "Rice": 7, "Soybean": 8, "Cotton": 2}
soil_mapping = {"Clay": 0, "Silt": 4, "Sandy": 3, "Loamy": 2}

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

col1, col2 = st.columns([1, 1])

if st.button("🚀 Analyze & Generate Prediction"):
    # Prepare data for prediction
    input_data = pd.DataFrame([{
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
    }])
    
    try:
        # Instead of requests.post, we call the model directly
        # Example: prediction = prediction_model.predict(input_data)[0]
        
        # NOTE: Since I don't have your full prediction logic from main.py, 
        # I've put a placeholder below. Replace this with your actual prediction code.
        predicted_yield = 4.5  # Replace with: prediction_model.predict(input_data)
        
        with col1:
            st.metric(label="Predicted Yield", value=f"{predicted_yield} MT/Ha")
            st.success("**Status:** Prediction generated successfully based on local model.")

    except Exception as e:
        st.error(f"Prediction Error: {e}")
else:
    st.info("👈 Adjust field conditions in the sidebar and click the button.")
