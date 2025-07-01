import streamlit as st
import pandas as pd
import joblib
from utils import predict_salary, recommend_strategy
from src.preprocess.preprocessing import load_and_prepare_data
from config import DATA_PATH, ML_MODEL_PATH, OPT_MODEL_PATH

st.title("📊 AI-Powered Hiring Optimization Tool")

ml_model = joblib.load(ML_MODEL_PATH)
opt_model = joblib.load(OPT_MODEL_PATH)

role = st.selectbox("Select Role", ["Software quality assurance tester"])
exp = st.selectbox("Experience Level", ["Entry", "Mid", "Senior"])
location = st.text_input("Enter City and State (e.g., Austin, TX)")
budget = st.number_input("Hiring Budget ($)", min_value=10000, step=1000)

if st.button("Predict & Recommend"):
    df = load_and_prepare_data(DATA_PATH)
    prediction = predict_salary(ml_model, df, role, exp, location)
    strategy = recommend_strategy(opt_model, budget)

    st.success(f"Predicted Cost: ${prediction:,.2f}")
    st.info(f"Recommended Hiring Strategy: {strategy}")
