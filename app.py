import streamlit as st
import numpy as np
import joblib
import pandas as pd

# ---------------- LOAD MODELS ----------------
svm_model = joblib.load("svm_crime_model.pkl")
rf_model = joblib.load("rf_crime_model.pkl")
scaler = joblib.load("scaler.pkl")

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Crime Rate Prediction System", layout="wide")

st.title("🚔 Crime Rate Prediction System")
st.write("Predict crime-prone areas and forecast crime rate using Machine Learning")

# ---------------- SIDEBAR INPUTS ----------------
st.sidebar.header("🔢 Input Features")

city_code = st.sidebar.number_input("City Code", min_value=0, step=1)
crime_desc = st.sidebar.number_input("Crime Description Code", min_value=0, step=1)
victim_age = st.sidebar.number_input("Victim Age", min_value=0, max_value=100)
police_deployed = st.sidebar.number_input("Police Deployed", min_value=0, step=1)
year = st.sidebar.number_input("Year", min_value=2000, max_value=2030, step=1)

# ---------------- INPUT PREPARATION ----------------
svm_input = np.array([[city_code, crime_desc, victim_age, police_deployed, year]])
svm_input_scaled = scaler.transform(svm_input)

reg_input = np.array([[city_code, year, police_deployed]])

# ---------------- BUTTONS ----------------
col1, col2 = st.columns(2)

with col1:
    if st.button("🔍 Predict Crime-Prone Area"):
        result = svm_model.predict(svm_input_scaled)
        if result[0] == 1:
            st.error("⚠️ High Crime-Prone Area")
        else:
            st.success("✅ Low Crime Area")

with col2:
    if st.button("📈 Forecast Crime Rate"):
        crime_rate = rf_model.predict(reg_input)
        st.info(f"Estimated Crime Count: {int(crime_rate[0])}")

# ---------------- CODE REFERENCES ----------------
st.markdown("---")
st.subheader("📌 Code Reference Tables (For Easy Selection)")

# City mapping
city_mapping = {
    0: "Delhi", 1: "Mumbai", 2: "Chennai", 3: "Kolkata",
    4: "Bengaluru", 5: "Hyderabad", 6: "Pune", 7: "Ahmedabad",
    8: "Jaipur", 9: "Chandigarh", 10: "Bhopal",
    11: "Indore", 12: "Lucknow"
}

# Crime mapping
crime_mapping = {
    0: "Theft", 1: "Robbery", 2: "Assault", 3: "Murder",
    4: "Kidnapping", 5: "Cyber Crime", 6: "Fraud",
    7: "Domestic Violence", 8: "Drug Abuse",
    9: "Sexual Harassment", 10: "Burglary"
}

col3, col4 = st.columns(2)

with col3:
    st.markdown("### 🏙 City Codes")
    city_df = pd.DataFrame(
        list(city_mapping.items()),
        columns=["City Code", "City Name"]
    )
    st.dataframe(city_df, use_container_width=True)

with col4:
    st.markdown("### 🚨 Crime Description Codes")
    crime_df = pd.DataFrame(
        list(crime_mapping.items()),
        columns=["Crime Code", "Crime Type"]
    )
    st.dataframe(crime_df, use_container_width=True)

# ---------------- POLICE DEPLOYMENT EXPLANATION ----------------
st.markdown("---")
st.subheader("🚓 Police Deployment Scale (Meaning of Values)")

police_mapping = {
    "0": "No / Very Low Police Presence",
    "1 – 3": "Low Police Presence",
    "4 – 6": "Moderate Police Presence",
    "7 – 9": "High Police Presence",
    "10+": "Very High / Intensive Policing"
}

police_df = pd.DataFrame(
    list(police_mapping.items()),
    columns=["Police Deployed Value", "Meaning"]
)

st.dataframe(police_df, use_container_width=True)

# ---------------- FOOTER ----------------
st.markdown("---")
st.caption("📊 Machine Learning Project | SVM Classification + Random Forest Regression")
