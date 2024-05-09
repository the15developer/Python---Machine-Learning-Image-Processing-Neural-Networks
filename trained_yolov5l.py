import os
import torch
from yolov5.train import run

# Load the pre-trained YOLOv5l model
model = torch.hub.load('ultralytics/yolov5', 'yolov5l')

# Modify the model to have the correct number of output classes
model.model[-1].nc = len(['bicycle', 'bus', 'car', 'motorbike', 'person'])

# Set the training parameters
data = 'C:/Users/Danny/Documents/spyder/Teknofest/Dataset1/data.yaml'
epochs = 30
batch_size = 32

# Run the training
run(
    data=data,
    model=model,  # Use the pre-loaded model
    epochs=epochs,
    batch_size=batch_size,
    project='C:/Users/Danny/Documents/spyder/Teknofest/PROJEMIZ',
    name='run1'
)
