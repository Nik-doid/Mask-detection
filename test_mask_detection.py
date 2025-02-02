import torch
import cv2
import numpy as np
import sys
from pathlib import Path
import os

# Ensure YOLOv7 directory is correctly added to the path
YOLO_PATH = Path("C:/ML/MaskDetection/yolov7").resolve()
sys.path.append(str(YOLO_PATH))
os.chdir(str(YOLO_PATH))  # Change working directory to YOLOv7 root

from models.experimental import attempt_load  # Import YOLOv7 model
from utils.general import non_max_suppression, scale_coords
from utils.datasets import letterbox

# Load the trained YOLOv7 model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")  # Debug statement

model = attempt_load(YOLO_PATH / "runs/train/exp4/weights/best.pt", map_location=device)
print("Model loaded successfully!")  # Debug statement
model.to(device).eval()

# Open webcam (0 = default camera)
cap = cv2.VideoCapture(0)
cap.set(3, 640)  # Width
cap.set(4, 480)  # Height

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame from webcam.")  # Debug statement
        break  # Stop if webcam is not accessible

    # Convert frame to RGB and preprocess
    img = letterbox(frame, 416, stride=32, auto=True)[0]  # Resize to 416x416
    img = img[:, :, ::-1].transpose(2, 0, 1)  # Convert BGR to RGB and rearrange shape
    img = np.ascontiguousarray(img)

    # Debug: Save the preprocessed image
    cv2.imwrite("debug_preprocessed.jpg", cv2.cvtColor(img.transpose(1, 2, 0), cv2.COLOR_RGB2BGR))

    # Convert image to tensor
    img = torch.from_numpy(img).to(device)
    img = img.float() / 255.0  # Normalize
    img = img.unsqueeze(0)  # Add batch dimension

    # Perform inference
    with torch.no_grad():
        pred = model(img)[0]
        print("Raw predictions:", pred)  # Debug raw output
        pred = non_max_suppression(pred, 0.1, 0.45)  # Lower confidence threshold

    # Process detections
    detections_found = False
    for det in pred:
        if len(det):
            detections_found = True
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], frame.shape).round()
            for *xyxy, conf, cls in det:
                label = f"Mask" if int(cls) == 0 else "No Mask"
                print(f"Detection: {label}, Confidence: {conf:.2f}, Class ID: {int(cls)}")  # Debug detections
                color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
                cv2.rectangle(frame, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), color, 2)
                cv2.putText(frame, f"{label}: {conf:.2f}", (int(xyxy[0]), int(xyxy[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # Display "No detections found" on the camera panel
    if not detections_found:
        text = "No detections found"
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
        text_x = (frame.shape[1] - text_size[0]) // 2  # Center the text horizontally
        text_y = (frame.shape[0] + text_size[1]) // 2  # Center the text vertically
        cv2.putText(frame, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Show result
    cv2.imshow("Mask Detection", frame)

    # Press 'Q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
