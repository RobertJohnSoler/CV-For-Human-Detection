import cv2
import requests
import numpy as np
from ultralytics import YOLO
import os
import shutil
from deepface import DeepFace
from z_person import person
import json

def analyze_img(image_url: str):
    
    detected_people = []
    
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
                analysis = DeepFace.analyze(img_path=filename, actions=['gender', 'age', 'race'], enforce_detection=True)[0]
                detected_person = person(filename, analysis)
                detected_people.append(detected_person)
                
    else:
        print("Failed to download image.")
    
    return detected_people
 
 
        
url = ""
detections = analyze_img(url)
for d in detections:
    print(json.dumps(d.getPersonInfo(), indent=4))