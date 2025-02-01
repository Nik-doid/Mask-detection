import torch
import cv2
import numpy as np
import sys
from pathlib import Path

# Ensure YOLOv7 directory is correctly added to the path
YOLO_PATH = Path("C:/ML/MaskDetection/yolov7").resolve()
sys.path.append(str(YOLO_PATH))

from yolov7.models.experimental import attempt_load  # Import YOLOv7 model
from yolov7.utils.general import non_max_suppression, scale_coords
from yolov7.utils.datasets import letterbox

# Load the trained YOLOv7 model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = attempt_load(YOLO_PATH / "runs/train/exp3/weights/best.pt", map_location=device)
model.to(device).eval()

# Open webcam (0 = default camera)
cap = cv2.VideoCapture(0)
cap.set(3, 640)  # Width
cap.set(4, 480)  # Height

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break  # Stop if webcam is not accessible

    # Convert frame to RGB and preprocess
    img = letterbox(frame, 640, stride=32, auto=True)[0]
    img = img[:, :, ::-1].transpose(2, 0, 1)  # Convert BGR to RGB and rearrange shape
    img = np.ascontiguousarray(img)

    # Convert image to tensor
    img = torch.from_numpy(img).to(device)
    img = img.float() / 255.0  # Normalize
    img = img.unsqueeze(0)  # Add batch dimension

    # Perform inference
    with torch.no_grad():
        pred = model(img)[0]
        pred = non_max_suppression(pred, 0.25, 0.45)  # Confidence and IoU thresholds

    # Process detections
    for det in pred:
        if len(det):
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], frame.shape).round()
            for *xyxy, conf, cls in det:
                label = f"Mask" if int(cls) == 0 else "No Mask"
                color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
                cv2.rectangle(frame, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), color, 2)
                cv2.putText(frame, f"{label}: {conf:.2f}", (int(xyxy[0]), int(xyxy[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # Show result
    cv2.imshow("Mask Detection", frame)

    # Press 'Q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
