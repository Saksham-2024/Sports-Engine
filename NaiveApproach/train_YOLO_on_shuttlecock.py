from ultralytics import YOLO
import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

if __name__ == "__main__":
    model = YOLO('yolov8s.pt')
    results = model.train(
        data='Shuttlecock/data.yaml',
        epochs=50,
        imgsz=416,
        batch=8,
        lr0=0.005,
        device=device,
        optimizer='Adam',
        patience=10,
        project='runs/train',
        name='shuttlecock_yolov8',
        exist_ok=True,
        amp=False
    )
    metrics = model.val()
    model.export(format='onnx')
