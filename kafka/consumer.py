from kafka import KafkaConsumer
import json

# Define the sensor topics you want to consume from
topics = [f'sensor{i}' for i in range(1,21)]

consumer = KafkaConsumer(
    *topics,
    bootstrap_servers='localhost:9092',
    auto_offset_reset='earliest',
    value_deserializer=lambda m: json.loads(m.decode('utf-8'))
)

print(f"Listening for messages from topics: {topics}")
for message in consumer:
    print(f"[{message.topic}] Received: {message.value}")
