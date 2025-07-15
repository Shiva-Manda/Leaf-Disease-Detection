Plant Disease Detection using YOLOv8

This project leverages **YOLOv8 (You Only Look Once)** for real-time object detection of plant leaf diseases. The model is trained on a curated and augmented version of the **PlantDoc dataset** to identify various diseases in tomato and other plant species.
- **Classes**: 8-class tomato-specific subset or full 30-class PlantDoc
- **Format**: YOLO format (image + label `.txt`)
- **Augmentation Techniques**:
  - Rotation
  - Flipping
  - Gaussian Noise
  - Biquadratic Blur
## 🧠 Model
- **Base Model**: `YOLOv8n.pt` (lightweight version)
- **Framework**: [Ultralytics YOLOv8]
- **Optimizer**: AdamW
- **Image Size**: 640×640
- **Epochs**: 150
- **Learning Rate**: 0.000833
## 🚀 Training
from ultralytics import YOLO
model = YOLO('yolov8n.pt')
model.train(
    data='tomato8.yaml',      
    epochs=150,
    imgsz=640,
    batch=16,
    optimizer='AdamW',
    lr0=0.000833,
    name='plant_disease_model'
)

## Reference 
[https://www.sciencedirect.com/science/article/pii/S2214662825000052?ref=pdf_download&fr=RR-2&rr=907086e4ed557f7c]
