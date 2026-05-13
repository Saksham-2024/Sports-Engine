from ultralytics import YOLO
import torch
import yaml

# Load Config
with open('configs.yaml', 'r') as f:
    config = yaml.safe_load(f)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

yolo_cfg = config['hyperparameters']['yolo']
SEED = config['hyperparameters']['general']['seed']

if __name__ == "__main__":
    model = YOLO('yolov8s.pt')
    results = model.train(
        data='Shuttlecock/data.yaml',
        epochs=yolo_cfg['epochs'],
        imgsz=yolo_cfg['imgsz'],
        batch=yolo_cfg['batch'],
        lr0=yolo_cfg['lr0'],
        device=device,
        optimizer=yolo_cfg['optimizer'],
        patience=yolo_cfg['patience'],
        project='runs/train',
        name='shuttlecock_yolov8',
        exist_ok=True,
        amp=yolo_cfg['amp'],
        seed=SEED,
    )
    metrics = model.val()
    model.export(format='onnx')
