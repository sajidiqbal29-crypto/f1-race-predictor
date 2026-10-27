# Prediction Button
if st.button("Predict Winner"):
    # Pass values as a list directly to avoid feature name mismatch errors
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
