import influxdb_client
from influxdb_client.client.write_api import SYNCHRONOUS
from environs import env
env.read_env()

bucket = env("INFLUXDB_BUCKET")
org = env("INFLUXDB_ORGANIZATION")
token = env("INFLUXDB_TOKEN")
url="http://influxdb2:8086"

print(bucket, org, token, url)

client = influxdb_client.InfluxDBClient(
   url=url,
   token=token,
   org=org
)

write_api = client.write_api(write_options=SYNCHRONOUS)


try:
    p = influxdb_client.Point("my_measurement").tag("location", "Prague").field("temperature", 25.3)
    write_api.write(bucket=bucket, org=org, record=p)
    print("✅ SUCCESS")
except Exception as e:
    print("❌ ERROR:", e)
