from helper import detect_faces,monitor_unauthorized_person
import paho.mqtt.client as mqtt
import cv2
from datetime import datetime
import requests


allow=False

def send_rfid(rfid):
    url = 'http://localhost:8000/rfid/create/'

    data = {
        'id_rfid': str(rfid),  # ✅ match the field name in the serializer
        'name': str(rfid),
        'last_access': datetime.now().isoformat(),  # optional if model handles it
        'is_active': True  # optional if default is set
    }

    requests.post(url, json=data)

def on_connect(client, userdata, flags, rc):
    print("✅ Connected with result code", rc)
    client.subscribe("esp32/rfid/motion")
    client.subscribe("esp32/rfid/uid")


def on_message(client, userdata, msg):
    
    print(f"📡 {msg.topic}: {msg.payload.decode()}")
    if msg.payload.decode()=='Motion Detected!':
        print("camera")
        # Configuration
        # Update with your YOLO model path
        API_key = "AIzaSyBs-aw-lP7CrxAcmK33UX1v3tfCylMAdyk"  # Replace with actual API key
        Model_Path="/home/rebbouh/Desktop/yolo/yolov8n-face.pt"
        # Step 1: Detect faces
        result = detect_faces(Model_Path, API_key)
        
        # Step 2: If unauthorized person detected, start monitoring
        if result == "unauthorized":
            # Get a fresh frame to start with
            cap = cv2.VideoCapture(0,cv2.CAP_V4L2)
            ret, frame = cap.read()
            cap.release()
            
            if ret:
                # Start monitoring the unauthorized person
                monitor_unauthorized_person(frame, API_key)
            else:
                print("❌ Failed to get initial frame for monitoring")
        elif result == "authorized":
            print("✅ Authorized person detected - system relaxed")
        else:
            print("⚠️ No definitive detection made")
        allow=False
    else:
        # print(msg.payload.decode())
        send_rfid(msg.payload.decode())



client = mqtt.Client()
client.on_connect = on_connect
client.on_message = on_message

# Your PC’s IP address (since it's the broker too)
client.connect("localhost", 1883, 60)

print("🚀 Listening for ESP32 messages...")
client.loop_forever()







#/home/rebbouh/Desktop/yolo/yolov8n-face.pt"