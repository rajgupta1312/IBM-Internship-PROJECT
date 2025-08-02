import streamlit as st
import pandas as pd
import requests

# 1️⃣ IBM Cloud API Configuration
API_KEY = "ouBEPwQ5JaYSWAIPBVJS8afIH99Z3aqqHmI7MCc2D2my"  # Replace with your IBM Cloud API key
DEPLOYMENT_URL = "https://eu-gb.ml.cloud.ibm.com/ml/v4/deployments/362ab47c-e03c-4056-bc87-6151d5466ebe/predictions?version=2021-05-01"

# Function to get IBM access token
def get_ibm_token():
    token_response = requests.post(
        'https://iam.cloud.ibm.com/identity/token',
        data={"apikey": API_KEY, "grant_type": 'urn:ibm:params:oauth:grant-type:apikey'}
    )
    return token_response.json()["access_token"]

# Function to call Watsonx ML Model for predictions
def get_predictions(df):
    token = get_ibm_token()

    # Required model input fields
    input_fields = [
        "STATE_NAME",
        "DISTRICT_NAME",
        "NO_OF_ROAD_WORK_SANCTIONED",
        "LENGTH_OF_ROAD_WORK_SANCTIONED",
        "NO_OF_BRIDGES_SANCTIONED",
        "COST_OF_WORKS_SANCTIONED"
    ]

    # Prepare payload
    payload_scoring = {
        "input_data": [{
            "fields": input_fields,
            "values": df[input_fields].values.tolist()
        }]
    }

    # Call IBM Watsonx deployment API
    response = requests.post(
        DEPLOYMENT_URL,
        json=payload_scoring,
        headers={'Authorization': 'Bearer ' + token}
    )

    result = response.json()

    # Extract predictions & confidence (probabilities)
    predictions = [row[0] for row in result['predictions'][0]['values']]
    confidences = [max(row[1]) if isinstance(row[1], list) else None for row in result['predictions'][0]['values']]

    # Append results to DataFrame
    df['Predicted_PMGSY_Scheme'] = predictions
    df['Confidence_Score'] = confidences
    return df

# ------------------- Streamlit UI -------------------
st.set_page_config(page_title="PMGSY Rural Project Classifier", layout="wide")
st.title("🏗 PMGSY Rural Project Classification Dashboard")

st.markdown("""
**Features:**
1. Enter a **single project manually** to get an instant prediction with confidence.
2. Upload a **CSV dataset** for **batch prediction**.
3. **Download predictions** as CSV for reporting.
---
""")

# Sidebar for mode selection
mode = st.sidebar.radio("Choose Mode", ["Manual Prediction", "Batch CSV Prediction"])

# ------------------- Manual Input Mode -------------------
if mode == "Manual Prediction":
    st.subheader("🔹 Manual Project Classification")

    col1, col2 = st.columns(2)

    with col1:
        state = st.text_input("State Name", "Chhattisgarh")
        district = st.text_input("District Name", "Raipur")
        roads = st.number_input("Number of Road Works Sanctioned", min_value=0, value=10)
        bridges = st.number_input("Number of Bridges Sanctioned", min_value=0, value=1)

    with col2:
        length = st.number_input("Length of Road Works (km)", min_value=0.0, value=50.0, format="%.3f")
        cost = st.number_input("Cost of Works Sanctioned (Cr)", min_value=0.0, value=100.0, format="%.5f")

    if st.button("Predict Scheme"):
        input_df = pd.DataFrame([[
            state, district, roads, length, bridges, cost
        ]], columns=[
            "STATE_NAME", "DISTRICT_NAME", "NO_OF_ROAD_WORK_SANCTIONED",
            "LENGTH_OF_ROAD_WORK_SANCTIONED", "NO_OF_BRIDGES_SANCTIONED",
            "COST_OF_WORKS_SANCTIONED"
        ])

        with st.spinner("Predicting..."):
            result_df = get_predictions(input_df)

        st.success("Prediction Completed!")
        st.write("### Result:")
        st.dataframe(result_df)

        scheme = result_df['Predicted_PMGSY_Scheme'].iloc[0]
        conf = result_df['Confidence_Score'].iloc[0]
        st.markdown(f"**Predicted Scheme:** {scheme}  \n**Confidence:** {conf:.2f}")

# ------------------- Batch CSV Upload Mode -------------------
else:
    st.subheader("🔹 Batch Prediction from CSV")

    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write("### Uploaded Dataset Preview")
        st.dataframe(df.head())

        if st.button("Predict Schemes for All Projects"):
            with st.spinner("Predicting all rows..."):
                result_df = get_predictions(df)
            
            st.success("Batch Prediction Completed!")
            st.write("### Results")
            st.dataframe(result_df)

            # Download button
            csv = result_df.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Download Predictions as CSV", data=csv, file_name="PMGSY_Predictions.csv", mime='text/csv')
