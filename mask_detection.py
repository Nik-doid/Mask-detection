# Import the RoboFlow library
from roboflow import Roboflow

# Initialize RoboFlow with your API key
rf = Roboflow(api_key="fsx2oiHNBRNHgi61tkp9")

# Load the project and model
project = rf.workspace("joseph-nelson").project("mask-wearing")
version = project.version(19)

# Download the dataset in YOLOv7 format
dataset = version.download("yolov7")

# Print the dataset path
print("Dataset downloaded to:", dataset.location)