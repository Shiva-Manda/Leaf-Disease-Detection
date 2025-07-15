from ultralytics import YOLO
# Load base model
model = YOLO('yolov8n.pt')  
# Train with AdamW and custom learning rate
model.train(
    data='data_adamw.yaml',
    epochs=150,
    imgsz=640,
    batch=16,
    optimizer='AdamW',
    lr0=0.000833, 
    name='train_adamw_originalval'
)
