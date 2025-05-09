import streamlit as st
import pandas as pd
import joblib

# Load the trained random forest model
model = joblib.load('model_rf.pkl')

# Function to apply one-hot encoding manually (assuming df_encoded was used during training)
def one_hot_encode(df_encoded, column, prefix, categories):
    # Create one-hot encoding with all possible categories
    for category in categories:
        df_encoded[f"{prefix}_{category}"] = (df_encoded[column] == category).astype(int)
    df_encoded = df_encoded.drop(columns=[column])  # Drop the original column
    return df_encoded

# Function to preprocess the user input to match the training data structure
def preprocess_input(user_input):
    # Categorical columns and their possible values (same as used in training)
    categorical_columns = {
        'Smoking': ['Yes', 'No'],
        'AlcoholDrinking': ['Yes', 'No'],
        'Stroke': ['Yes', 'No'],
        'DiffWalking': ['Yes', 'No'],
        'Sex': ['Male', 'Female'],
        'AgeCategory': ["18-24", "25-29", "30-34", "35-39", "40-44", "45-49", "50-54", 
                        "55-59", "60-64", "65-69", "70-74", "75-79", "80 or older"],
        'Race': ["White", "Black", "Asian", "American Indian/Alaskan Native", "Other", "Hispanic"],
        'Diabetic': ["Yes", "No", "No, borderline diabetes", "Yes (during pregnancy)"],
        'PhysicalActivity': ['Yes', 'No'],
        'GenHealth': ["Excellent", "Very good", "Good", "Fair", "Poor"],
        'Asthma': ['Yes', 'No'],
        'KidneyDisease': ['Yes', 'No'],
        'SkinCancer': ['Yes', 'No']
    }

    # Apply one-hot encoding for each categorical column
    for column, categories in categorical_columns.items():
        if column in user_input.columns:
            user_input = one_hot_encode(user_input, column, column, categories)
    
    # Ensure that all columns expected by the model are present (add missing columns as zeros)
    model_columns = [col for col in model.feature_names_in_]  # Columns expected by the model
    
    for column in model_columns:
        if column not in user_input.columns:
            user_input[column] = 0
    
    # Reorder the columns to match the model's expected order
    user_input = user_input[model_columns]
    
    return user_input

st.title("Heart Disease Risk Predictor")

# User inputs
bmi = st.number_input("BMI", min_value=10.0, max_value=60.0)
smoking = st.selectbox("Do you smoke?", ["Yes", "No"])
alcohol = st.selectbox("Do you drink alcohol?", ["Yes", "No"])
stroke = st.selectbox("Have you ever had a stroke?", ["Yes", "No"])
physical_health = st.slider("Physical Health (days of poor health in past 30 days)", 0, 30)
mental_health = st.slider("Mental Health (days of poor mental health)", 0, 30)
diff_walking = st.selectbox("Difficulty Walking?", ["Yes", "No"])
sex = st.selectbox("Gender", ["Male", "Female"])
age_category = st.selectbox("Age Category", [
    "18-24", "25-29", "30-34", "35-39", "40-44", "45-49", "50-54",
    "55-59", "60-64", "65-69", "70-74", "75-79", "80 or older"
])
race = st.selectbox("Race", ["White", "Black", "Asian", "American Indian/Alaskan Native", "Other", "Hispanic"])
diabetic = st.selectbox("Are you diabetic?", ["Yes", "No", "No, borderline diabetes", "Yes (during pregnancy)"])
physical_activity = st.selectbox("Engaged in physical activity in the past 30 days?", ["Yes", "No"])
gen_health = st.selectbox("General Health", ["Excellent", "Very good", "Good", "Fair", "Poor"])
sleep_time = st.slider("Average Sleep Time (hours)", 0, 24)
asthma = st.selectbox("Do you have asthma?", ["Yes", "No"])
kidney_disease = st.selectbox("Kidney Disease?", ["Yes", "No"])
skin_cancer = st.selectbox("Skin Cancer?", ["Yes", "No"])

# Collect inputs
user_input = pd.DataFrame({
    'BMI': [bmi],
    'Smoking': [smoking],
    'AlcoholDrinking': [alcohol],
    'Stroke': [stroke],
    'PhysicalHealth': [physical_health],
    'MentalHealth': [mental_health],
    'DiffWalking': [diff_walking],
    'Sex': [sex],
    'AgeCategory': [age_category],
    'Race': [race],
    'Diabetic': [diabetic],
    'PhysicalActivity': [physical_activity],
    'GenHealth': [gen_health],
    'SleepTime': [sleep_time],
    'Asthma': [asthma],
    'KidneyDisease': [kidney_disease],
    'SkinCancer': [skin_cancer]
})

# Preprocess the user input (one-hot encoding and missing feature handling)
user_input = preprocess_input(user_input)

# Model prediction
if st.button("Predict"):
    # Make prediction with the preprocessed input
    prediction = model.predict(user_input)[0]
    
    if prediction == 1:
        st.error("⚠️ High Risk of Heart Disease")
    else:
        st.success("✅ Low Risk of Heart Disease")
