import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pandas as pd
import pickle

## load the trained model
model = tf.keras.models.load_model('salary_regression_model.keras')

# Load the encoder and scaler

with open('label_encoder_gender.pkl','rb') as file:
    label_encoder_gender = pickle.load(file)

with open('onehot_encoder_geo.pkl','rb') as file:
    onehot_encoder_geo = pickle.load(file)

with open('scaler.pkl','rb') as file:
    scaler = pickle.load(file)


st.title('SALARY REGRESSION MODEL')

st.subheader('Geography')
geography = st.selectbox('Geography',onehot_encoder_geo.categories_[0])
st.subheader('Gender')
gender = st.selectbox('Gender',label_encoder_gender.classes_)
st.subheader('Age')
age = st.slider('Age',18,93)
st.subheader('Balance')
balance = st.number_input('Balance')
st.subheader('Credit Score')
credit_score = st.number_input('Credit Score')
st.subheader('Tenure')
tenure = st.slider('Tenure',0,10)
st.subheader('Number of product ')
num_of_products = st.slider("Number of Products",1,4)
st.subheader('Has Credit Card')
has_cr_card = st.selectbox('Has Credit Card',[0,1])
st.subheader('Is Active Member')
is_active_member = st.selectbox('Is Active Member',[0,1])

## prepare the input data
input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender' : [label_encoder_gender.transform([gender])[0]],
    'Age': [age],
    'Tenure':[tenure],
    'Balance':[balance],
    'NumOfProducts':[num_of_products],
    'HasCrCard':[has_cr_card],
    'IsActiveMember':[is_active_member],
})

geo_encoded = onehot_encoder_geo.transform([[geography]]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded,columns=onehot_encoder_geo.get_feature_names_out(['Geography']))

input_data = pd.concat([input_data.reset_index(drop=True),geo_encoded_df],axis=1)

input_data_scaled = scaler.transform(input_data)

prediction_salary =model.predict(input_data_scaled)

prediction_salary = prediction[0][0]
prediction_salary=  SALARY()
print("SALARY")








