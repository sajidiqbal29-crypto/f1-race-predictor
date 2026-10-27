
import streamlit as st
import pandas as pd
import joblib

st.title("🏎‍🟀 Formula 1 Race Winner Predictor")
st.write("Predicting race winners using Machine Learning & Historical Data.")

# Load saved model
model = joblib.load('f1_model.pkl')

st.success("Model loaded successfully! Ready for predictions.")
