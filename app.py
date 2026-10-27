import streamlit as st
import joblib

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
st.subheader("🏁 Enter Race & Driver Details:")

grid = st.slider("Starting Grid Position", min_value=1, max_value=20, value=1)
driverId = st.number_input("Driver ID", min_value=1, max_value=100, value=1)
constructorId = st.number_input("Constructor ID", min_value=1, max_value=50, value=1)
driver_points = st.number_input("Driver Points", min_value=0.0, max_value=500.0, value=25.0)
driver_standing_pos = st.slider("Driver Standing Position", min_value=1, max_value=20, value=1)
constructor_points = st.number_input("Constructor Points", min_value=0.0, max_value=1000.0, value=50.0)
constructor_st = st.slider("Constructor Standing Position", min_value=1, max_value=20, value=1)

# Prediction Button
if st.button("Predict Winner"):
    # Pass values directly as a list
    input_data = [[
        grid, driverId, constructorId, driver_points, 
        driver_standing_pos, constructor_points, constructor_st
    ]]
    
    prediction = model.predict(input_data)
    probability = model.predict_proba(input_data)
    
    if prediction[0] == 1:
        st.balloons()
        st.success(f"🎉 *Prediction: WINNER!* (Win Probability: {probability[0][1]*100:.2f}%)")
    else:
        st.warning(f"❌ *Prediction: Not a Winner.* (Win Probability: {probability[0][1]*100:.2f}%)")
