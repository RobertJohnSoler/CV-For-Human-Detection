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

    for i, box in enumerate(results[0].boxes.xyxy):
        x_min, y_min, x_max, y_max = map(int, box)  # Convert to integer values
        cls = int(results[0].boxes.cls[i])  # Class index
        conf = results[0].boxes.conf[i].item()
        label = f"{model.names[cls]} {conf:.2f}"  # Class name
        
        if "person" in label:
            # Draw rectangle around detected object
            cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            
            # Put class label
            cv2.putText(img, label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Show the image with detections
    width = int(img.shape[1] * 0.5)
    height = int(img.shape[0] * 0.5)
    cv2.namedWindow("YOLOv8 Object Detection", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("YOLOv8 Object Detection", width, height)  # Adjust size as needed
    cv2.imshow("YOLOv8 Object Detection", img)
    cv2.waitKey(0)  # Wait until a key is pressed
    cv2.destroyAllWindows()  # Close the window
    
    detected_objects = results[0].boxes.cls  # List of detected class indices
    class_names = model.names  # Map class indices to names

    # Check if "person" (class index 0) is detected
    person_detected = any(class_names[int(cls)] == "person" for cls in detected_objects)

    if person_detected:
        print("✅ Person detected in the image!")
    else:
        print("❌ No person detected in the profile picture.")
else:
    print("❌ Failed to download the image.")