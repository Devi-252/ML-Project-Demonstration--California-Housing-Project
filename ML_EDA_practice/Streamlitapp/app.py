import streamlit as st
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib as jb
from joblib import dump, load

st.set_page_config(layout="wide")

scaler = jb.load("scaler.pkl")

st.title("Restaurant rating Prediction App")

st.caption("this app helps you to predict a restaurants review class")

st.divider()

averagecost = st.number_input(
    "Please enter the estimate average cost for two",
    min_value=50,
    max_value=999999,
    value=1000,
    step=200
)

# ✅ fixed: use selectbox instead of checkbox with options
tablebooking = st.selectbox("Restaurant has table booking?", ["Yes", "No"])

onlinedelivery = st.selectbox("Restaurant has online booking?", ["Yes", "No"])

pricerange = st.selectbox(
    "what is the price range (1 Cheapest, 4 Most Expensive)", [1, 2, 3, 4]
)

predictbutton = st.button("Predict the review")

st.divider()

model = jb.load("mlmodel.pkl")

# ✅ fixed: logic comparing selectbox strings
bookingstatus = 1 if tablebooking == "Yes" else 0
deliverystatus = 1 if onlinedelivery == "Yes" else 0

values =[[averagecost, bookingstatus, deliverystatus, pricerange]]
my_X_values=np.array(values)

X= scaler.transform(my_X_values)
if predictbutton:
    st.snow()

    # ✅ fixed: store prediction and display
    prediction = model.predict(X)

    # Above 2 below 2.5  Poor
    # Above 2.5 below 3.5 Average
    # Above 3.5 below 4.0 Good
    # Above 4 below 4.5 Very Good'
    # Above 4.5 Excellent

    if prediction <2.5:
        st.write("Poor")
    elif prediction <3.5:
        st.write("Average")
    elif prediction <4.0:
        st.write("Good")
    elif prediction <4.5:
        st.write("Very Good")
    else:
        st.write("Excellent") 