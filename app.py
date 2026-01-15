import streamlit as st
import datetime
import pandas as pd
import joblib
import os

st.set_page_config(page_title="AgriPredict",  layout="wide")

# --- MODEL LOADING ---
# This is the "Brain" you just uploaded with Git LFS
MODEL_PATH = "crop_yield_model.pkl"

@st.cache_resource 
def load_active_model():
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)
    else:
        st.error(f"Model file {MODEL_PATH} not found in the cloud storage.")
        return None

model = load_active_model()

if model:
    st.sidebar.success(" Model Ready")

# --- UI LOGIC ---
st.title("🌾 Smart Farm: Yield Prediction Dashboard")

crop_mapping = {"Wheat": 9, "Corn": 1, "Rice": 7, "Soybean": 8, "Cotton": 2}
soil_mapping = {"Clay": 0, "Silt": 4, "Sandy": 3, "Loamy": 2}

with st.sidebar:
    st.header("Field Conditions")
    date = st.date_input("Observation Date", datetime.date.today())
    crop_name = st.selectbox("Select Crop", list(crop_mapping.keys()))
    soil_name = st.selectbox("Soil Type", list(soil_mapping.keys()))
    temp = st.slider("Temperature (°C)", -10, 50, 25)
    ph = st.slider("Soil pH", 4.0, 9.0, 6.5)
    hum = st.slider("Humidity (%)", 0, 100, 60)
    n_val = st.number_input("Nitrogen (N)", 0.0, 200.0, 50.0)
    p_val = st.number_input("Phosphorus (P)", 0.0, 200.0, 40.0)
    k_val = st.number_input("Potassium (K)", 0.0, 200.0, 30.0)
    sq_val = st.slider("Soil Quality Index", 0.0, 100.0, 80.0)

# --- THE PREDICTION BUTTON ---
if st.button("Analyze & Generate Prediction"):
    if model is None:
        st.error("Model not loaded correctly.")
    else:
        # Prepare data for the model
        input_df = pd.DataFrame([{
            "Crop_Type": crop_mapping[crop_name],
            "Soil_Type": soil_mapping[soil_name],
            "Soil_pH": ph,
            "Temperature": temp,
            "Humidity": float(hum),
            "Wind_Speed": 12.0, 
            "N": n_val, "P": p_val, "K": k_val,
            "Soil_Quality": sq_val,
            "month": date.month, "year": date.year
        }])
        
        try:
            prediction = model.predict(input_df)[0]
            st.divider()
            st.metric(label="Estimated Yield", value=f"{prediction:.2f} MT/Ha")
            st.success("Prediction generated successfully on your mobile device!")
        except Exception as e:
            st.error(f"Prediction Error: {e}")
