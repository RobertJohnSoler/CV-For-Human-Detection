import cv2
import requests
import numpy as np
from ultralytics import YOLO
import os
import shutil
from deepface import DeepFace

def analyze_img(image_url: str):
    
    model = YOLO("yolov8n.pt")  # Using pre-trained YOLOv8 Nano model
    objects = model.names
    response = requests.get(image_url, stream=True)

    if response.status_code == 200:
        
        shutil.rmtree("detected_people")
        os.mkdir("detected_people")
        
        # Convert image bytes to NumPy array
        img_array = np.asarray(bytearray(response.content), dtype=np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

        # Save image locally
        image_path = "profile.jpg"
        cv2.imwrite(image_path, img)
        print(f"Image saved as {image_path}")

        # Run YOLOv8 detection
        results = model.predict(image_path)
        
        for i, box in enumerate(results[0].boxes):
            detected_coords = box.xyxy
            x_min = int(detected_coords[0][0])
            y_min = int(detected_coords[0][1])
            x_max = int(detected_coords[0][2])
            y_max = int(detected_coords[0][3])
            detected_object = objects[int(box.cls)]
            
            if detected_object == "person":
                print("person detected!")
                person_img = img[y_min:y_max, x_min:x_max]
                filename = f"detected_people/cropped_{i}.jpg"
                cv2.imwrite(filename, person_img)
                analysis = DeepFace.analyze(img_path=filename, actions=['gender', 'age', 'race'], enforce_detection=True)
                print(f"Results for {filename}", analysis)
                
    else:
        print("Failed to download image.")