import influxdb_client
import pandas as pd
from influxdb_client.client.write_api import SYNCHRONOUS
from environs import env
import json


def env_setup():

    env.read_env()

    bucket = env("INFLUXDB_BUCKET")
    org = env("INFLUXDB_ORGANIZATION")
    token = env("INFLUXDB_TOKEN")
    

    return token, org, bucket


def parse_data():
    df = pd.read_csv("../data/1000_thermal_data.csv")
    result = df.head()
    print(result)
    return df


def initialize():
    df = parse_data()

    formatted_df = df[["packet_time", "items"]].copy()
    formatted_df["items"] = formatted_df["items"].apply(json.loads)


    setup = env_setup()
    token = setup[0]
    org = setup[1]
    bucket = setup[2]
    url="http://localhost:8086"

    client = influxdb_client.InfluxDBClient(
        url=url,
        token=token,
        org=org
    )

    write_api = client.write_api(write_options=SYNCHRONOUS)

    try:
        for _, row in formatted_df.iterrows():
            timestamp = row["packet_time"]
            sensors = row["items"]

            for key, value in sensors.items():
                
                if isinstance(value, (int, float)):
                    point = (
                        influxdb_client.Point("thermal_readings")
                        .tag("sensor", key)
                        .field("temperature", value)
                        .time(timestamp)
                    )
                    write_api.write(bucket=bucket, org=org, record=point)
        
        print("✅ SUCCESS")
    except Exception as e:
        print("❌ ERROR:", e)


print("Ha")
initialize()