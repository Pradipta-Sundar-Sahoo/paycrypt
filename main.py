import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import datetime
import joblib
from fastapi import FastAPI
from threading import Thread
import uvicorn

# ==========================
# 🚀 FASTAPI BACKEND
# ==========================
app = FastAPI(title="Toll Plaza Management API")

# Load Data and Models
df = pd.read_csv("preprocessed_tollplaza_data.csv")
df["initiated_time"] = pd.to_datetime(df["initiated_time"])
df["time_interval"] = pd.to_datetime(df["time_interval"])

traffic_model = joblib.load("traffic_prediction_model.pkl")
lane_optimizer = joblib.load("lane_optimizer.pkl") if "lane_optimizer.pkl" in df else None

# API: Traffic Prediction
@app.get("/api/traffic")
def get_traffic_prediction(plaza: str, hour: int):
    input_features = pd.DataFrame({"hour": [hour]})  # Adjust based on trained model
    prediction = traffic_model.predict(input_features)[0]
    return {"plaza": plaza, "hour": hour, "predicted_transaction_count": int(prediction)}

# API: Lane Recommendations
@app.get("/api/lane")
def get_lane_recommendations(plaza: str, hour: int):
    if lane_optimizer:
        recommendations = lane_optimizer.get_lane_recommendations(plaza, hour)
        return recommendations
    return {"error": "Lane optimizer model not found"}

# Run FastAPI in a separate thread
def run_fastapi():
    uvicorn.run(app, host="127.0.0.1", port=8000)

Thread(target=run_fastapi, daemon=True).start()

# ==========================
# 🎨 STREAMLIT FRONTEND
# ==========================
st.set_page_config(page_title="Toll Plaza Dashboard", layout="wide")

# Sidebar Inputs
st.sidebar.title("Toll Plaza Traffic Analytics")
plaza_name = st.sidebar.text_input("Enter Plaza Name:", "Electronic City")
hour = st.sidebar.slider("Select Hour of the Day:", 0, 23, 8)

# Fetch Traffic Prediction
st.header("🚦 Traffic Prediction")
if st.button("Get Prediction"):
    response = requests.get("http://127.0.0.1:8000/api/traffic", params={"plaza": plaza_name, "hour": hour})
    if response.status_code == 200:
        prediction = response.json()
        st.metric(label="Predicted Transactions", value=prediction["predicted_transaction_count"])
    else:
        st.error("Failed to fetch prediction!")

# Fetch Lane Recommendations
st.header("🛣️ Lane Optimization")
if st.button("Get Lane Recommendations"):
    response = requests.get("http://127.0.0.1:8000/api/lane", params={"plaza": plaza_name, "hour": hour})
    if response.status_code == 200:
        lane_data = response.json()
        df_lanes = pd.DataFrame(lane_data["recommended_lanes"])
        st.write(df_lanes)
        fig = px.bar(df_lanes, x="lane", y="expected_volume", color="role", title="Lane Allocation")
        st.plotly_chart(fig)
    else:
        st.error("Failed to fetch lane data!")

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("🚀 **Toll Plaza AI Dashboard** | FastAPI + Streamlit")
