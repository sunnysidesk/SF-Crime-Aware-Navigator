import streamlit as st
import os, joblib
import pandas as pd
import numpy as np
from datetime import datetime
from utils import (
    create_ors_client, geocode_address, get_route_coords, assess_route,
    iterative_reroute_min_risk, plot_route_on_map
)

from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score

import openrouteservice
import requests
import folium
from streamlit_folium import st_folium
from shapely.geometry import Polygon, mapping
from shapely.ops import unary_union
from dotenv import load_dotenv


import streamlit as st
import threading
from kafka import KafkaConsumer
import json
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter


# Initialize geocoder with a unique user_agent
geolocator = Nominatim(user_agent="sf-crime-app")
reverse = RateLimiter(geolocator.reverse, min_delay_seconds=1, max_retries=2, error_wait_seconds=2.0)

def get_location_name(lat, lon):
    try:
        location = reverse((lat, lon), exactly_one=True, language="en")
        if location is None:
            return "Unknown location"
        # Use neighborhood or road if available, fallback to address
        if "neighbourhood" in location.raw["address"]:
            return location.raw["address"]["neighbourhood"]
        if "suburb" in location.raw["address"]:
            return location.raw["address"]["suburb"]
        if "road" in location.raw["address"]:
            return location.raw["address"]["road"]
        return ", ".join(location.address.split(",")[:2])  # Show street and neighborhood only
    except Exception:
        return "Unknown location"

# --- Streamlit app setup ---

# # This is safe as a global!
# live_events = []


EVENTS_FILE = "live_events.json"

def load_events():
    if not os.path.exists(EVENTS_FILE):
        return []
    with open(EVENTS_FILE, "r") as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            return []

def save_event(event):
    events = load_events()
    events.append(event)
    # Keep only latest 100
    events = events[-100:]
    with open(EVENTS_FILE, "w") as f:
        json.dump(events, f)



if "live_events" not in st.session_state:
    st.session_state["live_events"] = []

def kafka_listener():
    print("Kafka listener thread starting!")
    consumer = KafkaConsumer(
        'crime-events',
        bootstrap_servers='localhost:9092',
        auto_offset_reset='earliest',
        group_id='crime-risk-app',
        value_deserializer=lambda m: json.loads(m.decode('utf-8'))
    )
    print("Kafka consumer created, waiting for events...")
    for msg in consumer:
        print(f"RECEIVED: {msg.value}")
        save_event(msg.value)

if "kafka_started" not in st.session_state:
    threading.Thread(target=kafka_listener, daemon=True).start()
    st.session_state["kafka_started"] = True
    


events = load_events()
st.sidebar.header("🚨 Live Crime Incidents")
st.write("Current number of incidents:", len(events))
if events:
    for event in reversed(events[-10:]):
        t = event.get("datetime", "")
        desc = event.get("description", "")
        cat = event.get("category", "")
        lat = event.get("latitude", "")
        lon = event.get("longitude", "")
        st.sidebar.write(
            f"**[{t}]**: {cat} - {desc}\n\n"
            f"📍 ({lat:.5f}, {lon:.5f})"
        )
else:
    st.sidebar.write("No incidents yet. Waiting for Kafka events...")

if st.sidebar.button("🔄 Refresh incidents"):
    st.rerun()


# --- Load secrets and models ---
load_dotenv("py.env")
api_key = os.getenv("ORS_API_KEY")
# st.write(f"API KEY: {api_key}")  # Remove for production

# Load models ONCE at startup
clf = joblib.load("models/risk_model.joblib")
ohe = joblib.load("models/encoder.joblib")
day_labels = ohe.get_feature_names_out(['day_of_week_encoded'])
ors_client = create_ors_client(api_key)

st.title("San Francisco Crime Route Risk Analyzer")

# --- UI: Get user input ---
start_address = st.text_input("Enter your starting address", "")
end_address = st.text_input("Enter your destination address", "")
hour = st.number_input("Hour of travel (0-23)", min_value=0, max_value=23, value=12)
minute = st.number_input("Minute of travel (0-59)", min_value=0, max_value=59, value=0)
day_str = st.selectbox("Day of the week", [
    "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"
])

# --- Main logic only runs if button is clicked ---
if 'route_result' not in st.session_state:
    st.session_state['route_result'] = None

if st.button("Check Route Safety"):
    # 1. Only call these ONCE:
    start = geocode_address(start_address, api_key)
    end = geocode_address(end_address, api_key)
    # st.write(f"Start: {start}, End: {end}")  # (optional for debugging)

    if not (start and end):
        st.error("Could not geocode one or both addresses. Please check or use a more specific address.")
        st.stop()

    st.success(f"Valid addresses found: {start_address} -> {end_address}")

    coords = get_route_coords(start, end, ors_client)
    if coords is None:
        st.error("Could not fetch a walking route between these addresses.")
        st.stop()

    try:
        result = iterative_reroute_min_risk(
            coords, start, end, hour, minute, day_str,
            clf, ohe, day_labels, ors_client
        )
        st.session_state['route_result'] = (result, start, end)
    except Exception as e:
        st.error(f"Error during rerouting/risk calculation: {e}")
        st.stop()

if st.session_state['route_result']:
    result, start, end = st.session_state['route_result']
    st.success(f"Route risk score: {result['avg_risk']:.2f}")

## Show rerouting status and risk level
    if result["was_rerouted"]:
        if result["avg_risk"] <= 0.5:
            st.info(f"⚠️ The route was automatically rerouted to avoid high-risk areas! Buffer used: {result['buffer_used']}")
        else:
            st.warning("🚨 The route was rerouted, but even the safest alternate path is still high risk. Please be careful or travel at another time.")
    else:
        if result["avg_risk"] <= 0.5:
            st.success("👍 The original path is within safe risk limits (no rerouting needed).")
        else:
            st.warning("🚨 The original path is high risk, and no safer alternate route could be found. Please be careful or travel at another time.")

# --- Plot the route on a map ---
    st.subheader("Route Map")
    if result is None or "coords" not in result or result["coords"] is None:
            st.error("Rerouting failed or returned no valid route. Try another address or time.")
            st.stop()
        
    folium_map = plot_route_on_map(
        result["coords"], start, end, result["avg_risk"], result["risk_per_point"], rerouted=result["was_rerouted"]
    )
    st_folium(folium_map, width=700)

# --- END OF SCRIPT ---
