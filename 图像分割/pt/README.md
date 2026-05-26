---
tags:
- ultralytics
- yolov8
- image-segmentation
- computer-vision
- pytorch
- defect-detection
library_name: ultralytics
license: agpl-3.0
---

# YOLOv8 Nano Segmentation: Cracks & Drywall Joints

This is a fine-tuned YOLOv8 Nano segmentation model (`yolov8n-seg`) designed to detect and mask structural cracks and drywall joints/taping areas. 

It was trained to provide a lightweight, fast baseline for construction quality assurance, automated structural inspection, and defect detection.

## Model Details
* **Base Model:** YOLOv8 Nano Segmentation (`yolov8n-seg.pt`)
* **Task:** Instance Segmentation
* **Epochs:** 50
* **Classes:**
  * `0`: Joint / Tape / Drywall Seam
  * `1`: Crack / Wall Crack

## Dataset
The model was trained on a custom merged dataset compiled from Roboflow. 
* **Total Training Images:** ~6,187 images
* **Composition:** Mixed dataset containing varied examples of wall/surface cracks and drywall taping joints.

## Performance Metrics
Based on the validation set after 50 epochs of training:
* **mAP50 (Mask):** ~0.90+
* **mAP50-95 (Mask):** ~0.60+
* Box detection metrics mirror the segmentation performance closely, indicating strong localization before masking.

*(Training curves and loss metrics can be viewed in the `results.jpg` image attached to this repository).*

## How to Use

You can load and run this model directly using the `ultralytics` library.

### Installation
```bash
pip install ultralytics opencv-python
```

Inference Code
Python
```bash
from ultralytics import YOLO
import cv2

# Load the model (ensure you download the .pt file from this repo)
model = YOLO("yolov8n-seg-cracks-joints.pt") # rename if your file is named differently

# Run inference on an image
image_path = "your_test_image.jpg"
results = model(image_path)

# Extract and display results
result = results[0]
if result.masks is not None:
    for i, mask in enumerate(result.masks.data):
        class_id = int(result.boxes.cls[i].item())
        class_name = "Crack" if class_id == 1 else "Joint/Tape"
        print(f"Detected: {class_name}")

# Save the visualized prediction
result.save(filename="prediction_output.jpg")