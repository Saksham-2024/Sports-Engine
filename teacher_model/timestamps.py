import pandas as pd
import json
import os
import cv2

dataset_dir = '../dataset'
json_dir = os.path.join(dataset_dir, 'json')
video_dir = os.path.join(dataset_dir, 'videos')
output_file = '1-timestamps.csv'

items = [d for d in os.listdir(json_dir) if d.endswith('.json')]
items.sort()

timestamps = []

for i, file in enumerate(items):
    video_file = file.replace('.json', '.mp4')
    video_path = os.path.join(video_dir, video_file)
    fps = cv2.VideoCapture(video_path).get(cv2.CAP_PROP_FPS)
    with open(os.path.join(json_dir, file), 'r') as f:
        data = json.load(f)
        for point_no, rally_info in data.items():
            rally = rally_info["PointInfo"]["Rally"]
            for stroke in rally:
                stroke_num = stroke.get("StrokeNum")
                stroke_type = stroke.get("StrokeType")
                stroke_begin = stroke.get("StrokeBegin") * fps if stroke.get("StrokeBegin") is not None else None
                stroke_end = stroke.get("StrokeEnd") * fps if stroke.get("StrokeEnd") is not None else None
                stroke_type = stroke_type.replace("-Bh", "") if stroke_type else None
                playing_side = stroke.get("Player")
                if playing_side == "T1P1":
                    playing_side = 1
                else:
                    playing_side = 2
                camera = "normal" if stroke.get("Camera") == "normal" else "focus"

                
                if None in (stroke_num, stroke_type, stroke_begin, stroke_end) or camera != "normal":
                    continue
                
                timestamps.append({
                    "match_no": i + 1,
                    "point_no": point_no,
                    "stroke_num": stroke_num,
                    "stroke_type": stroke_type,
                    "stroke_begin": stroke_begin,
                    "stroke_end": stroke_end,
                    "playing_side": playing_side,
                    "camera": camera    
                })

df = pd.DataFrame(timestamps)
df.to_csv(output_file, index=False)