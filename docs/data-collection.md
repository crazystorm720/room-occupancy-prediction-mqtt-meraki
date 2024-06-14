# Data Collection Guide

## Overview
This document provides a step-by-step guide to set up the data collection pipeline for the Room Occupancy Prediction project. We use MQTT to collect real-time sensor data from Meraki devices, including chassis fan speed, temperature, and humidity, and store it in a TimescaleDB database.

## Prerequisites
Before you start, ensure you have the following installed on your system:
- Docker
- Python 3.7 or higher
- `pip` (Python package installer)

## Setting Up MQTT Broker

### Step 1: Install Mosquitto MQTT Broker
We will use the Mosquitto MQTT broker for this project. You can run it using Docker.

```bash
docker run -d --name mosquitto -p 1883:1883 eclipse-mosquitto
```

### Step 2: Configure MQTT Broker
By default, Mosquitto should be running with basic configuration. For more advanced settings, you can create a `mosquitto.conf` file and mount it to the container.

```bash
docker run -d --name mosquitto -p 1883:1883 -v /path/to/mosquitto.conf:/mosquitto/config/mosquitto.conf eclipse-mosquitto
```

## Setting Up TimescaleDB

### Step 1: Install TimescaleDB
Run the following command to start a TimescaleDB instance using Docker:

```bash
docker run -d --name timescaledb -p 5432:5432 -e POSTGRES_PASSWORD=mysecretpassword timescale/timescaledb:latest-pg12
```

### Step 2: Configure TimescaleDB
Once the container is running, you can connect to the TimescaleDB instance using a PostgreSQL client like `psql`:

```bash
docker exec -it timescaledb psql -U postgres
```

Create a new database and enable TimescaleDB extension:

```sql
CREATE DATABASE room_occupancy;
\c room_occupancy;
CREATE EXTENSION IF NOT EXISTS timescaledb;
```

## Collecting Sensor Data

### Step 1: Install Python Packages
Install the necessary Python packages for MQTT and PostgreSQL integration:

```bash
pip install paho-mqtt psycopg2-binary
```

### Step 2: Create Data Collection Script
Create a Python script `data_collection.py` to connect to the MQTT broker, process the messages, and store them in TimescaleDB.

```python
import paho.mqtt.client as mqtt
import psycopg2
import json

# MQTT settings
MQTT_BROKER = "localhost"
MQTT_PORT = 1883
MQTT_TOPIC = "meraki/sensors"

# TimescaleDB settings
DB_HOST = "localhost"
DB_NAME = "room_occupancy"
DB_USER = "postgres"
DB_PASS = "mysecretpassword"

# Connect to TimescaleDB
conn = psycopg2.connect(host=DB_HOST, database=DB_NAME, user=DB_USER, password=DB_PASS)
cur = conn.cursor()

# Create table if not exists
cur.execute("""
CREATE TABLE IF NOT EXISTS sensor_data (
    time TIMESTAMPTZ PRIMARY KEY,
    temperature DOUBLE PRECISION,
    humidity DOUBLE PRECISION,
    chassis_fan_speed DOUBLE PRECISION
);
SELECT create_hypertable('sensor_data', 'time', if_not_exists => TRUE);
""")
conn.commit()

# MQTT callback functions
def on_connect(client, userdata, flags, rc):
    print("Connected with result code " + str(rc))
    client.subscribe(MQTT_TOPIC)

def on_message(client, userdata, msg):
    data = json.loads(msg.payload)
    timestamp = data['timestamp']
    temperature = data['temperature']
    humidity = data['humidity']
    chassis_fan_speed = data['chassis_fan_speed']

    cur.execute("""
    INSERT INTO sensor_data (time, temperature, humidity, chassis_fan_speed)
    VALUES (%s, %s, %s, %s)
    ON CONFLICT (time) DO NOTHING;
    """, (timestamp, temperature, humidity, chassis_fan_speed))
    conn.commit()

# Set up MQTT client
client = mqtt.Client()
client.on_connect = on_connect
client.on_message = on_message

client.connect(MQTT_BROKER, MQTT_PORT, 60)
client.loop_forever()
```

### Step 3: Run Data Collection Script
Run the `data_collection.py` script to start collecting data:

```bash
python data_collection.py
```

## Testing Data Collection

To ensure that the data collection setup is working correctly, you can publish test messages to the MQTT broker using the following Python script:

```python
import paho.mqtt.publish as publish
import json
from datetime import datetime

MQTT_BROKER = "localhost"
MQTT_PORT = 1883
MQTT_TOPIC = "meraki/sensors"

test_message = {
    "timestamp": datetime.utcnow().isoformat(),
    "temperature": 22.5,
    "humidity": 45.0,
    "chassis_fan_speed": 1200
}

publish.single(MQTT_TOPIC, json.dumps(test_message), hostname=MQTT_BROKER, port=MQTT_PORT)
```

Run this script to send a test message:

```bash
python publish_test_message.py
```

Check the TimescaleDB database to verify that the data has been inserted correctly.

## Conclusion

You have now set up the data collection pipeline for the Room Occupancy Prediction project. The MQTT broker collects real-time sensor data from Meraki devices, and the data is stored in TimescaleDB for further processing and analysis. This setup allows for efficient data ingestion and storage, enabling you to build and train predictive models using the collected data.

For any issues or further customization, refer to the respective documentation for Mosquitto MQTT broker and TimescaleDB.
