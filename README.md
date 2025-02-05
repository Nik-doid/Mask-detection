Overview
This project implements a real-time mask detection system using the YOLOv7 (You Only Look Once) object detection model. The system detects whether a person is wearing a mask or not in real-time using a webcam feed. It is designed to help enforce mask-wearing compliance in public spaces, workplaces, or other environments where mask usage is required.

The model is trained on a custom dataset with two classes:

Mask: Person wearing a mask.

No Mask: Person not wearing a mask.

Features
Real-Time Detection: Detects masks in real-time using a webcam feed.

High Accuracy: Utilizes the YOLOv7 model, known for its speed and accuracy in object detection.

Custom Training: The model is trained on a custom dataset to ensure it performs well in specific environments.

Cross-Platform: Works on both CPU and GPU (CUDA-enabled devices).

Easy to Use: Simple setup and execution with detailed instructions.

How It Works
Input: The system captures video frames from a webcam or a pre-recorded video.

Preprocessing: Each frame is resized to 416x416 pixels and normalized to match the model's input requirements.

Inference: The YOLOv7 model processes the frame and predicts bounding boxes for detected objects (masks or no masks).

Postprocessing: Non-maximum suppression (NMS) is applied to filter out overlapping boxes and low-confidence detections.

Output: The system displays the video feed with bounding boxes and labels indicating whether a mask is detected or not.


https://drive.google.com/drive/folders/1dyRqkoqZZ47uZIMw6FRr7ugITvKr06nV?usp=sharing

You can use this link to download it and use it as required.
