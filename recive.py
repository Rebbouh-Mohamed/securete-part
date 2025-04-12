import paho.mqtt.client as mqtt
import requests

def send_system_stat(indoor_temp):
    url = "http://localhost:8000/temp/add-stats/"
    

    data = {
        "indoor_temp": indoor_temp,
        "humidity": 45.0,
        "server_load": 0.25,
        "external_temp": 18.0,
        "ac_level": 1.0,
        "fans_active": 1.0,
        "hour": 12.0  # Assume noon
    }

    requests.post(url, json=data)
    
def on_connect(client, userdata, flags, rc):
    print("✅ Connected with result code", rc)
    client.subscribe("esp32/temperature")

def on_message(client, userdata, msg):
    print(f"📡 {msg.topic}: {msg.payload.decode()}")
    indoor_temp=msg.payload.decode()
    send_system_stat(indoor_temp)
client = mqtt.Client()
client.on_connect = on_connect
client.on_message = on_message

# Your PC’s IP address (since it's the broker too)
client.connect("localhost", 1883, 60)

print("🚀 Listening for ESP32 messages...")
client.loop_forever()


#192.168.209.162