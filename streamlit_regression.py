import streamlit as st
import numpy as pd
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pickle

# Load the trained model
model = tf.keras.models.load_model(
    "/Users/jasper/Desktop/RNN-classification/regression_model.h5"
)

# Load the encoders and the scaler
with open(
    "/Users/jasper/Desktop/RNN-classification/label_encoder_gender.pkl", "rb"
) as file:
    label_encoder_gender = pickle.load(file)


with open(
    "/Users/jasper/Desktop/RNN-classification/one_hot_encoder_Geography.pkl", "rb"
) as file:
    one_hot_encoder_Geography = pickle.load(file)

with open("/Users/jasper/Desktop/RNN-classification/scaler.pkl", "rb") as file:
    scaler = pickle.load(file)

## Streamlit app
st.title("Estimated Salary Prediction")

# User Input
geography = st.selectbox("Geography", one_hot_encoder_Geography.categories_[0])
gender = st.selectbox("Gender", label_encoder_gender.classes_)
age = st.slider("Age", 18, 92)
balance = st.number_input("Balance")
credit_score = st.number_input("Credit Score", 0, 10)
exited = st.number_input("Exited")
tenure = st.slider("Tenure", 0, 10)
num_of_products = st.slider("Number of Products", 1, 4)
has_credit_card = st.selectbox("Has Credit Card", [0, 1])
is_active_member = st.selectbox("Is Active Member", [0, 1])

# Prepare the input data
input_data = pd.DataFrame(
    {
        "CreditScore": [credit_score],
        # "Geography": [geography],
        "Gender": [label_encoder_gender.transform([gender])[0]],
        "Age": [age],
        "Tenure": [tenure],
        "Balance": [balance],
        "NumOfProducts": [num_of_products],
        "HasCrCard": [has_credit_card],
        "IsActiveMember": [is_active_member],
        "Exited": [exited],
    }
)

# One-hot-Encode Geography
geo_encoded = one_hot_encoder_Geography.transform([[geography]]).toarray()
geo_encoded = pd.DataFrame(
    geo_encoded, columns=one_hot_encoder_Geography.get_feature_names_out(["Geography"])
)

# Concatenate the input data and the encoded geography
input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded], axis=1)

# Scale the input data
input_data_scaled = scaler.transform(input_data)

# Make the prediction
prediction = model.predict(input_data_scaled)
predicted_salary = prediction[0][0]

# Display the prediction
st.write(f"Predicted Salary: ${predicted_salary:.2f}")
