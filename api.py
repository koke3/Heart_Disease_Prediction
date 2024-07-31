import streamlit as st
import pandas as pd
from pycaret.classification import load_model, predict_model

# Load the model
model = load_model('api')

# App title
st.title("Heart Disease Prediction")

# Display an image
st.image("https://github.com/Prem07a/Heart-Disease/raw/main/data/image/hd.png", caption="Heart Disease Prediction", use_column_width=True)

# User input
st.sidebar.header("User Input Features")
age = st.sidebar.number_input("Age", min_value=1, max_value=120, value=25)
sex = st.sidebar.selectbox("Sex", options=[0, 1], format_func=lambda x: "Female" if x == 0 else "Male")
chest_pain_type = st.sidebar.selectbox("Chest Pain Type", options=[0, 1, 2, 3])
resting_blood_pressure = st.sidebar.number_input("Resting Blood Pressure", min_value=50, max_value=200, value=120)
serum_cholesterol = st.sidebar.number_input("Serum Cholesterol in mg/dl", min_value=100, max_value=400, value=200)
fasting_blood_sugar = st.sidebar.selectbox("Fasting Blood Sugar > 120 mg/dl", options=[0, 1])
resting_ecg = st.sidebar.selectbox("Resting Electrocardiographic Results", options=[0, 1, 2])
max_heart_rate = st.sidebar.number_input("Maximum Heart Rate Achieved", min_value=60, max_value=220, value=150)
exercise_angina = st.sidebar.selectbox("Exercise Induced Angina", options=[0, 1])
oldpeak = st.sidebar.number_input("ST Depression Induced by Exercise Relative to Rest", min_value=0.0, max_value=10.0, value=1.0)
slope = st.sidebar.selectbox("Slope of the Peak Exercise ST Segment", options=[0, 1, 2])
major_vessels = st.sidebar.selectbox("Number of Major Vessels Colored by Fluoroscopy", options=[0, 1, 2, 3])
thal = st.sidebar.selectbox("Thal", options=[0, 1, 2], format_func=lambda x: "Normal" if x == 0 else "Fixed Defect" if x == 1 else "Reversible Defect")

# Predict button
if st.sidebar.button("Predict"):
    # Prepare data for model input
    data = {
        'cp': chest_pain_type,
        'trestbps': resting_blood_pressure,
        'chol': serum_cholesterol,
        'fbs': fasting_blood_sugar,
        'restecg': resting_ecg,
        'thalach': max_heart_rate,
        'exang': exercise_angina,
        'ca': major_vessels,
        'age': age,
        'sex': sex,
        'oldpeak': oldpeak,
        'slope': slope,
        'thal': thal
    }

    df = pd.DataFrame([data])
    
    # Make prediction
    predictions = predict_model(model, data=df)
    
    # Check for 'prediction_label' column
    if 'prediction_label' in predictions.columns and 'prediction_score' in predictions.columns:
        predicted_grade = predictions["prediction_label"].iloc[0]
        prediction_score = predictions["prediction_score"].iloc[0]
        
        # Map results to human-readable format
        grade_map = {
            0: 'No disease',
            1: 'Disease',
        }
        grade = grade_map.get(predicted_grade, "Unknown")
        
        # Display results
        st.success(f"Predicted Health Status: {grade}")
        st.info(f"Prediction Confidence Score: {prediction_score:.2f}")
        
        # Provide user-friendly analysis
        analysis = (
            f"Based on the information provided, the model predicts that the patient {'does not have' if predicted_grade == 0 else 'has'} heart disease. "
            f"The confidence score of this prediction is {prediction_score:.2f}. "
            "Here's a breakdown of the key factors that influenced this prediction:\n\n"
            f"1. **Age**: {age} years old.\n"
            f"2. **Sex**: {'Female' if sex == 0 else 'Male'}.\n"
            f"3. **Chest Pain Type**: Type {chest_pain_type}.\n"
            f"4. **Resting Blood Pressure**: {resting_blood_pressure} mm Hg.\n"
            f"5. **Serum Cholesterol**: {serum_cholesterol} mg/dl.\n"
            f"6. **Fasting Blood Sugar**: {'> 120 mg/dl' if fasting_blood_sugar == 1 else '<= 120 mg/dl'}.\n"
            f"7. **Resting ECG Results**: {resting_ecg}.\n"
            f"8. **Maximum Heart Rate Achieved**: {max_heart_rate} bpm.\n"
            f"9. **Exercise Induced Angina**: {'Yes' if exercise_angina == 1 else 'No'}.\n"
            f"10. **ST Depression Induced by Exercise**: {oldpeak}.\n"
            f"11. **Slope of the Peak Exercise ST Segment**: {slope}.\n"
            f"12. **Number of Major Vessels Colored by Fluoroscopy**: {major_vessels}.\n"
            f"13. **Thalassemia**: {'Normal' if thal == 0 else 'Fixed Defect' if thal == 1 else 'Reversible Defect'}."
        )
        
        st.subheader("Detailed Analysis")
        st.write(analysis)
    else:
        st.error("The prediction did not return the expected columns.")

# To run the app, use the following command in the terminal:
# streamlit run app_streamlit.py
