import cv2
import numpy as np
import requests
import os
from deepface import DeepFace

# URL of the image
image_url = "https://scontent-ord5-2.xx.fbcdn.net/v/t39.30808-6/461325477_10225612205002842_7091785603339753515_n.jpg?stp=cp6_dst-jpg_tt6&_nc_cat=106&ccb=1-7&_nc_sid=6ee11a&_nc_ohc=foCyihytD_sQ7kNvgGzqv-9&_nc_oc=AdjxLcpahE2wv3EGrKwoXyxHG_QpGeAvVUOvlgfADOg13wiv4uJE6FLT2yHG7YBt5aeafTYjz17XUp385IYBs0wK&_nc_zt=23&_nc_ht=scontent-ord5-2.xx&_nc_gid=AL3JiMwT6EHpTkx4DLmagk_&oh=00_AYFwC3qQ70HOpr2gne-lvUJqO4uPypCDlyr-7du91VsFBQ&oe=67D2420A"

# Download the image
response = requests.get(image_url, stream=True)
if response.status_code == 200:
    img_array = np.asarray(bytearray(response.content), dtype=np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    # Save image locally
    image_path = "downloaded_image.jpg"
    cv2.imwrite(image_path, img)
    print(f"Image saved as {image_path}")

    # Run face detection
    try:
        analysis = DeepFace.analyze(img_path=image_path, actions=['gender', 'age'], enforce_detection=True)
        print("Face detected! This is likely a person.")
        print(analysis)
        for face in analysis:
            region = face["region"]
            x, y, w, h = region["x"], region["y"], region["w"], region["h"]
            
            # Draw a rectangle around the face
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Put text (Predicted Gender)
            gender = face["dominant_gender"]
            confidence = face["face_confidence"]
            text = f"{gender} ({confidence*100:.1f}%)"
            cv2.putText(img, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Show the image with detections
        width = int(img.shape[1] * 0.5)
        height = int(img.shape[0] * 0.5)
        cv2.namedWindow("Deepface Detection", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Deepface Detection", width, height)  # Adjust size as needed
        cv2.imshow("Deepface Detection", img)
        cv2.waitKey(0)  # Wait until a key is pressed
        cv2.destroyAllWindows()  # Close the window
    except:
        print("No face detected. This may not be a person.")

else:
    print("Failed to download the image.")