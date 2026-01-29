import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO
import os

model = YOLO('runs/train/shuttlecock_yolov8/weights/best.pt')
video_dir = 'videos'
videos = [video_dir + '/' + vid for vid in os.listdir(video_dir) if vid.endswith(('.mp4', '.avi'))]
output_dir = 'keyframes_through_shuttle'
os.makedirs(output_dir, exist_ok=True)
keyframe_data = []
