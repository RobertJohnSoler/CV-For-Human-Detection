import cv2
import requests
import numpy as np
from ultralytics import YOLO

# Load YOLOv8 model
model = YOLO("yolov8n.pt")  # Using pre-trained YOLOv8 Nano model

# Image URL
image_url = "https://scontent-ord5-2.xx.fbcdn.net/v/t39.30808-6/461325477_10225612205002842_7091785603339753515_n.jpg?stp=cp6_dst-jpg_tt6&_nc_cat=106&ccb=1-7&_nc_sid=6ee11a&_nc_ohc=foCyihytD_sQ7kNvgGzqv-9&_nc_oc=AdjxLcpahE2wv3EGrKwoXyxHG_QpGeAvVUOvlgfADOg13wiv4uJE6FLT2yHG7YBt5aeafTYjz17XUp385IYBs0wK&_nc_zt=23&_nc_ht=scontent-ord5-2.xx&_nc_gid=AL3JiMwT6EHpTkx4DLmagk_&oh=00_AYFwC3qQ70HOpr2gne-lvUJqO4uPypCDlyr-7du91VsFBQ&oe=67D2420A"

# Download the image
response = requests.get(image_url, stream=True)
if response.status_code == 200:
    # Convert image bytes to NumPy array
    img_array = np.asarray(bytearray(response.content), dtype=np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    # Save image locally
    image_path = "profile.jpg"
    cv2.imwrite(image_path, img)
    print(f"Image saved as {image_path}")

    # Run YOLOv8 detection
    results = model(image_path)

    # Get detected class indexes
    print("cls is", results[0].boxes.cls)
    print("confidence is", results[0].boxes.conf)