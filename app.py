import streamlit as st
import pandas as pd
import joblib
import urllib.request
import os
import zipfile

# 1. Setup paths
MODEL_URL = "https://github.com/user-attachments/files/24524928/crop_yield_model.zip"
ZIP_PATH = "crop_yield_model.zip"
MODEL_PATH = "crop_yield_model.pkl"

@st.cache_resource
def load_large_model():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("Downloading model... this may take a minute."):
            # We use a custom opener to prevent timeout errors (Error 60)
            opener = urllib.request.build_opener()
            opener.addheaders = [('User-agent', 'Mozilla/5.0')]
            urllib.request.install_opener(opener)
            urllib.request.urlretrieve(MODEL_URL, ZIP_PATH)
            
            with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
                zip_ref.extractall(".")
    return joblib.load(MODEL_PATH)

# 2. Try to load the model at startup
model = None
try:
    model = load_large_model()
    st.sidebar.success("✅ Model Ready")
except Exception as e:
    st.sidebar.error(f"❌ Model Loading Failed: {e}")

# ... (rest of your UI code here) ...

# 3. Fix the prediction button logic
if st.button("🚀 Analyze & Generate Prediction"):
    if model is not None:
        try:
            # Ensure your input_df matches what the model expects
            # input_df = pd.DataFrame([payload]) 
            prediction = model.predict(input_df)[0]
            st.metric(label="Predicted Yield", value=f"{prediction:.2f} MT/Ha")
        except Exception as e:
            st.error(f"Prediction Error: {e}")
    else:
        st.error("The model is not loaded. Please wait for the download to finish or check sidebar errors.")
