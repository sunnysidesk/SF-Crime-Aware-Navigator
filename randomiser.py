from kafka import KafkaProducer
from faker import Faker
import random
import json
import time
from datetime import datetime

# --- Configuration ---
KAFKA_BROKER = 'localhost:9092'  # Update if using Confluent Cloud or another broker
TOPIC = 'crime-events'

# --- Setup ---
fake = Faker()
producer = KafkaProducer(
    bootstrap_servers=KAFKA_BROKER,
    value_serializer=lambda v: json.dumps(v).encode('utf-8')
)

# SF bounding box (approx)
LAT_MIN, LAT_MAX = 37.70, 37.81
LON_MIN, LON_MAX = -122.52, -122.36

CATEGORIES = [
    "Assault", "Robbery", "Burglary", "Larceny Theft", "Arson",
    "Weapons Offense", "Homicide", "Sex Offense", "Rape", "Vandalism"
]

DESCRIPTIONS = [
    "Assault reported", "Robbery in progress", "Burglary at residence",
    "Theft from vehicle", "Vandalism spotted", "Weapon found",
    "Homicide investigation", "Sexual assault", "Arson at warehouse"
]

while True:
    event = {
        "incident_id": fake.uuid4(),
        "datetime": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "category": random.choice(CATEGORIES),
        "description": random.choice(DESCRIPTIONS),
        "latitude": round(random.uniform(LAT_MIN, LAT_MAX), 6),
        "longitude": round(random.uniform(LON_MIN, LON_MAX), 6)
    }
    producer.send(TOPIC, event)
    print("Sent:", event)
    time.sleep(random.uniform(1, 5))  # 1–5 seconds between events