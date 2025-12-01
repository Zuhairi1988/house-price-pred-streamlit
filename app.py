import streamlit as st
import joblib
import numpy as np

st.title("House Price Prediction App")

st.divider()

st.write("This app uses Machine Learning for Predicting House Price")

st.divider()

bedrooms = st.number_input("Number of bedrooms", min_value = 0, value = 0)
bathrooms = st.number_input("Number of bathrooms", min_value = 0, value = 0)
grade = st.number_input("Grade house(1-10)", min_value = 0, value = 0)
livingarea = st.number_input("Living area(sqf)", min_value=0, value=2000)
areaofthehouse = st.number_input("Area of house(sqf)", min_value=300, value=300)
livingarearenov = st.number_input("Living area renovation", min_value=0, value=2000)
numberviews = st.number_input("Number of views(view times)", min_value=0, value=0)
Areabasement = st.number_input("Area basement(sqf)", min_value=0, value=0)

st.divider()

model = joblib.load("model.pkl")

x = [[bedrooms,bathrooms,grade,livingarea,areaofthehouse,livingarearenov,numberviews,Areabasement]]

predictbutton = st.button("Predict")

if predictbutton:
    x_array = np.array(x)

    prediction = model.predict(x_array)
    

    formatted_price = f"RP{prediction[0]:,.2f}"
    st.write(f"Price prediction is {formatted_price}")


else:
    st.write("Click Here")
    

   



