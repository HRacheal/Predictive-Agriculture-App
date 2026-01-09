import streamlit as st
import datetime
import pandas as pd
import joblib
import urllib.request
import os
import zipfile

st.set_page_config(page_title="AgriPredict", page_icon="🌱", layout="wide")

# --- MODEL DOWNLOADING AND LOADING ---
MODEL_URL = "https://github.com/user-attachments/files/24524928/crop_yield_model.zip"
ZIP_PATH = "crop_yield_model.zip"
MODEL_PATH = "crop_yield_model.pkl"

@st.cache_resource # This ensures it only downloads once
def load_large_model():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("Downloading and extracting model... please wait."):
            urllib.request.urlretrieve(MODEL_URL, ZIP_PATH)
            with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
                zip_ref.extractall(".")
    return joblib.load(MODEL_PATH)

try:
    model = load_large_model()
    st.sidebar.success("✅ Model Ready")
except Exception as e:
    st.sidebar.error(f"❌ Model Error: {e}")

# --- STYLING ---
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
    # Prepare the data exactly as the model expects it
    input_df = pd.DataFrame([{
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
        prediction = model.predict(input_df)[0]
        with col1:
            st.metric(label="Predicted Yield", value=f"{prediction:.2f} MT/Ha")
            if prediction < 3.0: # Example logic for recommendation
                st.warning("**Recommendation:** Consider increasing Nitrogen (N) levels.")
            else:
                st.success("**Recommendation:** Conditions are optimal for this crop.")
    except Exception as e:
        st.error(f"Prediction Error: {e}")
