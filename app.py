import streamlit as st
import joblib
import pandas as pd

# Page Title
st.title("🏎‍🟀 Formula 1 Race Winner Predictor")
st.write("Predicting race winners using Machine Learning & Historical Data.")

# Load the trained model
@st.cache_resource
def load_model():
    model = joblib.load('f1_model.pkl')
    return model

model = load_model()
st.success("Model loaded successfully! Ready for predictions.")

st.divider()

# User Inputs for Prediction
st.subheader("🏁 Enter Race Details:")

grid = st.slider("Starting Grid Position", min_value=1, max_value=20, value=1)
points = st.number_input("Points", min_value=0.0, max_value=50.0, value=25.0)

# Prediction Button
if st.button("Predict Winner"):
    # Input data frame matching model features
    input_data = pd.DataFrame([[grid, points]], columns=['grid', 'points'])
    
    prediction = model.predict(input_data)
    probability = model.predict_proba(input_data)
    
    if prediction[0] == 1:
        st.balloons()
        st.success(f"🎉 *Prediction: WINNER!* (Win Probability: {probability[0][1]*100:.2f}%)")
    else:
        st.warning(f"❌ *Prediction: Not a Winner.* (Win Probability: {probability[0][1]*100:.2f}%)")

