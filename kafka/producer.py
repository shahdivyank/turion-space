from kafka import KafkaProducer
import pandas as pd
import json
import time

# Initialize producer
producer = KafkaProducer(
    bootstrap_servers='localhost:9092',
    value_serializer=lambda v: json.dumps(v).encode('utf-8')
)

# Read CSV
df = pd.read_csv('./data/1000_thermal_data.csv')

for _, row in df.iterrows():
    # Parse timestamp and sensor readings
    packet_time = row['packet_time']
    sensors = json.loads(row['items'])  # Convert JSON string to dict

    for sensor_name, sensor_value in sensors.items():
        topic = sensor_name.lower()  # e.g., SENSOR1 -> sensor1

        # Build message payload
        message = {
            'packet_time': packet_time,
            'sensor': sensor_name,
            'value': sensor_value
        }

        print(f"Sending to topic '{topic}': {message}")
        producer.send(topic, value=message)

    time.sleep(1)  # Simulate real-time delay

producer.flush()
producer.close()
