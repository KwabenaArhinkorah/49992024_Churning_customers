import streamlit as st
import pandas as pd
from keras.models import load_model
import pickle

# Load the trained model
best_model = load_model("keras_model.h5")

# Load the scaler object
with open("my_scaler.pkl", "rb") as scaler_file:
    scaler_objects = pickle.load(scaler_file)

# Check if the scaler object is a tuple
if isinstance(scaler_objects, tuple):
    # If it's a tuple, take the first element as the scaler
    scaler = scaler_objects[0]
else:
    # Otherwise, assume it's the scaler object
    scaler = scaler_objects

# Define features used during training
features = ['TotalCharges', 'MonthlyCharges', 'Contract_Month-to-month', 'PaymentMethod_Electronic check',
            'OnlineSecurity_No', 'gender', 'PaperlessBilling', 'Partner', 'OnlineBackup', 'TechSupport_No',
            'Dependents', 'MultipleLines_No', 'DeviceProtection_No', 'InternetService_Fiber optic',
            'StreamingMovies_No', 'StreamingTV_No', 'tenure', 'SeniorCitizen', 'PhoneService']

# Create a Streamlit app
st.set_page_config(
    page_title="Churn Prediction App",
    layout="centered",
    initial_sidebar_state="expanded"
)

st.title("Customer Churn Prediction App")

# Collect user input for a new customer
total_charge = st.number_input("What is this customer's total charge for this year?")
monthly_charge = st.number_input("What is this customer's monthly charge?")
contract = st.select_slider("What type of contract does this customer have? Enter 0 for monthly, 1 for yearly.",
                            [0, 1])
payment_m = st.select_slider(
    "How does this customer usually pay? Enter 0 for bank transfer, 1 for credit card, 2 for electronic check, 3 for mailed check",
    [0, 1, 2, 3])
Osecurity = st.select_slider(
    "Do they have online Security? Enter 0 for no,1 for if they don't have phone service, and 2 for yes", [0, 1, 2])
gender = st.select_slider("What is your gender? Enter 0 for female, 1 for male", [0, 1])
paperlessBilling = st.select_slider("Are they billed paperlessly? Enter 0 for no, 1 for Yes.", [0, 1])
partner = st.select_slider("Do they have partners? 0: No, 1: Yes", [0, 1])
onlineBack = st.select_slider(
    "Do they have online backups? Enter 0 for No and 1 if they dont have phones, 2 for Yes ", [0, 1, 2])
techS = st.select_slider(
    "Do you offer them tech support? Enter 0 for No, 1 if they don't have internet service, and 2 for yes.",
    [0, 1, 2])
dependent = st.select_slider("Do they have dependents? Enter 0 for no, 1 for yes", [0, 1])
multipleLines = st.select_slider(
    "Do they have multiple phone lines? Enter 0 for no, 1 if they don't have phone service, and 2 for yes.",
    [0, 1, 2])
devicePro = st.select_slider(
    "Do they have device protection? Enter 0 for no, 1 if they don't have internet service, 2 for yes ",
    [0, 1, 2])
internetService = st.select_slider("Do they have internet service? O for no, 1 for yes.", [0, 1])
streamMov = st.select_slider("Do they stream movies? 0 for no, 1 if no internet service 2 for yes.", [0, 1, 2])
streamTV = st.select_slider(
    "Do they stream television? 0 for no,1 if no internet service and 2 for yes", [0, 1, 2])
tenure = st.number_input("How long have they stayed with you? (in years)")
senior = st.slider("How old are they?", 0, 100)
phoneService = st.select_slider("Do they have phone service? 0 for no, 1 for yes.", [0, 1])

# Create a dictionary for the user input
user_input = {
    'TotalCharges': total_charge,
    'MonthlyCharges': monthly_charge,
    'Contract': contract,
    'PaymentMethod': payment_m,
    'OnlineSecurity': Osecurity,
    'gender': gender,
    'PaperlessBilling': paperlessBilling,
    'Partner': partner,
    'OnlineBackup': onlineBack,
    'TechSupport': techS,
    'Dependents': dependent,
    'MultipleLines': multipleLines,
    'DeviceProtection': devicePro,
    'InternetService': internetService,
    'StreamingMovies': streamMov,
    'StreamingTV': streamTV,
    'tenure': tenure,
    'SeniorCitizen': senior,
    'PhoneService': phoneService
}

button = st.button("Predict")

if button:
    # Convert the user input dictionary to a DataFrame
    use = pd.DataFrame([user_input], columns=features)
    # Use the scaler_objects to scale the user input
    scaled_user_input = scaler.transform(use)
    # Predict using the loaded model
    prediction = best_model.predict(scaled_user_input)[0][0]

    if prediction <= 0.5:
        st.write("This customer will not churn")
    else:
        st.write("This customer will churn.")
