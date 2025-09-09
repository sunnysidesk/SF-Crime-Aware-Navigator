import openrouteservice
# from google.colab import userdata
import requests
import folium
from shapely.geometry import Polygon, mapping
from shapely.ops import unary_union

from kafka import KafkaProducer
import json
import time

producer = KafkaProducer(
    bootstrap_servers='localhost:9092',  # change if cloud
    value_serializer=lambda v: json.dumps(v).encode('utf-8')
)

sample_event = {
    "incident_id": "abc123",
    "datetime": "2025-05-26 22:40",
    "category": "Assault",
    "description": "Assault reported near Civic Center",
    "latitude": 37.7815,
    "longitude": -122.4167
}

while True:
    producer.send('crime-events', sample_event)
    print("Sent:", sample_event)
    time.sleep(10)  # send every 10 seconds