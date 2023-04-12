import streamlit as st
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model, load_model

# Load model and feature names
with open('pipeline.pkl', 'rb') as file_1:
    pipeline= joblib.load(file_1)

model_func_tune= load_model('model.h5')

# Set up app title and header image
st.set_page_config(page_title='Churn Rate Prediction')
st.title('Churn Rate Prediction')
st.image('https://i.imgur.com/rToAL0W.png', use_column_width=True)

# Collect user input using sliders
st.subheader('Enter Customers Information:')
membership_category = st.selectbox('membership_category', tuple(['Silver Membership', 'Gold Membership', 'No Membership',
                                                                 'Basic Membership', 'Platinum Membership', 'Premium Membership']))
joined_through_referral = st.selectbox('joined_through_referral', tuple(['Yes', 'No']))
feedback = st.selectbox('feedback', tuple(['Poor Product Quality', 'Poor Website', 'No reason specified',
       'Quality Customer Care', 'Poor Customer Service',
       'Reasonable Price', 'User Friendly Website', 'Too many ads',
       'Products always in Stock']))
avg_transaction_value = st.slider ('avg_transaction_value $', 0, 100000)
avg_frequency_login_days = st.slider ('avg_frequency_login_days ', 0, 50)
points_in_wallet = st.slider ('points_in_wallet $', 0, 1500)


# Generate new DataFrame based on user input
new_data = pd.DataFrame({
    'membership_category': [membership_category],
    'joined_through_referral': [joined_through_referral],
    'feedback': [feedback],
    'avg_transaction_value': [avg_transaction_value],
    'avg_frequency_login_days': [avg_frequency_login_days],
    'points_in_wallet': [points_in_wallet]
})
df= pipeline.transform(new_data)
# Make prediction and display result
if st.button('Predict'):
    y_pred_func_tuned = model_func_tune.predict(df)
    y_pred_func_tuned = np.where(y_pred_func_tuned >=0.5, 1, 0)  
    if y_pred_func_tuned == 1:
        message = "From the Customers information, it seems that the customers is churn."
        color = 'red'
    else:
        message = "From the Customers information, it seems that the customers is not churn."
        color = 'green'
    st.subheader('Prediction:')
    st.write(message, unsafe_allow_html=True, )
    st.markdown(f'<h1 style="color:{color};text-align:center">{y_pred_func_tuned}</h1>', unsafe_allow_html=True)
