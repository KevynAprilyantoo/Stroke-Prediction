import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from streamlit_option_menu import option_menu

etc_models = joblib.load('etc_model.joblib')

rfc_models = joblib.load('rfc_model.joblib')

xgb_models = joblib.load('xgb_model.joblib')

gbc_models = joblib.load('gbc_model.joblib')

option_gender = ['Male', 'Female']
option_married = ['Yes', 'No']
option_work = ['Govt Job', 'Never worked', 'Private', 'Self employed', 'children']
option_residence = ['Urban', 'Rural']
option_smoking = ['Unknown', 'formerly smoked', 'never smoked','smokes']

def user_input_features():
    st.subheader('Please enter the following inputs:')

    gender = st.selectbox("Gender: ", options= option_gender)
    age = st.number_input("Age",min_value=0, max_value=110, value=0)
    hypertension = st.number_input("Hypertension (0 = No / 1 = Yes)",min_value=0, max_value=1, value=0)
    hearth = st.number_input("Hearth Disease (0 = No / 1 = Yes)",min_value=0, max_value=1, value=0)
    married = st.selectbox("Ever Married ",options= option_married)
    work = st.selectbox("Work Type", options= option_work)
    residence = st.selectbox("Residence Type", options= option_residence)
    glucose = st.number_input("Average Glucose Level ", min_value=0, max_value=1000, value=0)
    BMI = st.number_input("Body Mass Index", min_value=0, max_value=100, value=0)
    smoking = st.selectbox("Smoking Status", options= option_smoking)

    preprocessInput = preprocess_input(gender, age, hypertension, hearth, married, work, residence, glucose, BMI, smoking)
    return preprocessInput

def preprocess_input(Gender, Age, Hypertension, Hearth, Married, Work, Residence, Glucose, BMIndex, smoking):
    # Encode categorical variables
    if Gender == 'Male':
        Gender_encoded = 1
    elif Gender == "Female":
        Gender_encoded = 0

    # age = {'Infant' : 0, "Child" : 1, "Teenagers" : 3, "Young Adult" : 4, "Adult" : 5, "Middle Aged": 6, "Senior":7}
    if Age >= 0 and Age <= 2:
        Agegroup = 0
    elif Age > 2 and Age <= 12:
        Agegroup = 1
    elif Age > 12 and Age <= 18:
        Agegroup = 2
    elif Age > 18 and Age <= 35:
        Agegroup = 3
    elif Age > 35 and Age <= 50:
        Agegroup = 4
    elif Age > 50 and Age <= 65:
        Agegroup = 5
    elif Age > 65 and Age <= 100:
        Agegroup = 6

    if Married == 'Yes':
        Married = 1
    elif Married == 'No':
        Married = 0

    if Work == 'Govt Job':
        work = 0
    elif Work == 'Never worked':
        work = 1   
    elif Work == 'Private':
        work = 2
    elif Work == 'Self employed':
        work = 3
    elif Work == 'children':
        work = 4
    
    if Residence == 'Urban':
        residence_type = 1
    elif Residence == 'Rural':
        residence_type = 0

    if smoking == 'Unknown':
        smoking = 0
    elif smoking == 'formerly smoked':
        smoking = 1
    elif smoking == 'never smoked':
        smoking = 2
    elif smoking == 'smokes':
        smoking = 3

    return [Gender_encoded, Age, Hypertension, Hearth, Married, work, residence_type, Glucose, BMIndex, smoking, Agegroup]


def main():
    st.title('Stroke Detection')
    
    model_name = st.selectbox('Select Model', ('Extra Trees', 'Random Forest', 'XGB', 'Gradien Boosting'))
   
    input_df = np.array(user_input_features()).reshape(1, -1)
    

    st.subheader('User Input Summary')
    st.write(input_df)

    if st.button('Predict'):
        if model_name == 'Extra Trees':
            model = etc_models
        elif model_name == 'Random Forest':
            model = rfc_models
        elif model_name == 'XGB':
            model = xgb_models
        else:
            model = gbc_models

        prediction = model.predict(input_df)
        prediction_proba = np.max(model.predict_proba(input_df))

        st.subheader('Prediction')
        st.success('Stroke' if str(prediction) == 1 else 'Not Stroke')

        st.subheader('Prediction Probability')
        st.success(f'Accuracy: {prediction_proba * 100:.2f}%')

        if 'predictions' not in st.session_state:
            st.session_state['predictions'] = []
        if 'inputs' not in st.session_state:
            st.session_state['inputs'] = []
        if 'true_labels' not in st.session_state:
            st.session_state['true_labels'] = []
        
        st.session_state['predictions'].append(prediction[0])
        st.session_state['inputs'].append(input_df)

a, b, c = st.columns([0.2, 0.6, 0.2])
with b:
    st.image("images.jpeg", use_column_width=True)

st.subheader("""A stroke prediction app utilizes various demographic and health data such as gender, age, hypertension, heart conditions, marital status, occupation, residence, glucose levels, BMI, and smoking status to assess an individual's likelihood of experiencing a stroke. 
             These factors are analyzed using statistical models or machine learning algorithms to provide personalized risk 
             assessments. Users receive recommendations for lifestyle adjustments or medical interventions based on their assessed
              risk level, empowering proactive measures to reduce stroke risk and improve cardiovascular health.""")

if __name__ == '__main__':
    main()