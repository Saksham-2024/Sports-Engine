import pandas as pd
import numpy as np
from ultralytics import YOLO
import cv2
import pickle
import os

existing_dataset_path = '../teacher_model/1-timestamps.csv'
homography_cache_file = '../teacher_model/homography_cache.pkl'
video_dir = '../dataset/videos'
output_csv = 'formatted_dataset.csv'
completed_strokes = set()

df = pd.read_csv(existing_dataset_path)
grouped = df.groupby(['match_no'])
model = YOLO('yolov8m.pt')

if os.path.exists(output_csv):
    try:
        existing_csv = pd.read_csv(output_csv, usecols=['match_no', 'point_no', 'stroke_num'])
        completed_subset = existing_csv.drop_duplicates()
        completed_strokes = sorted(set(tuple(x) for x in completed_subset.to_numpy()))
        last_rally = completed_strokes[-1]
        print(f"Resuming... Found {len(completed_strokes)} completely processed strokes in CSV.")
    except Exception as e:
        print(f"Starting fresh. Could not read output CSV: {e}")

def court_to_pixel(cx, cy, H):
    point = np.array([[[cx, cy]]], dtype='float32')
    transformed = cv2.perspectiveTransform(point, np.linalg.inv(H))
    pixel_x, pixel_y = transformed[0][0]
    return int(pixel_x), int(pixel_y)

def pixel_to_court(x, y, H):
    point = np.array([[[x, y]]], dtype='float32')
    transformed = cv2.perspectiveTransform(point, H)
    return transformed[0][0]

def is_on_court(court_x, court_y, court_width=5.18, court_length=13.4):
        return 0 <= court_x <= court_width and 0 <= court_y <= court_length

with open(homography_cache_file, 'rb') as f:
    matrices = pickle.load(f)
    print("Homography Matrices Loaded")

for match, group in grouped:
    match_no = match[0]
    video_path = video_dir + f'/match{match_no}.mp4'
    H = matrices[video_path]
    cap = cv2.VideoCapture(video_path)
    print(f"Going through match {match_no}. Total items: {len(group)}")

    for i in range(len(group)):
        start_frame = int(group.iloc[i]['stroke_begin'])
        end_frame = int(group.iloc[i]['stroke_end'])
        playing_side = group.iloc[i]['playing_side']
        stroke_num = group.iloc[i]['stroke_num']
        point_no = group.iloc[i]['point_no']
        
        current_stroke_id = (match_no, point_no, stroke_num)  
        if current_stroke_id <= last_rally:
            print(f"Skipping Match {match_no}, Point {point_no}, Stroke {stroke_num} - Already complete.")
            continue    
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        print(f"Processing item {i + 1}. {((i / len(group)) * 100):.2f}% completed")

        data = []
        for frame in range(start_frame, end_frame + 1):
            row = {}
            ret, img = cap.read()
            
            for col in df.columns:
                row[col] = group.iloc[i][col]
            row['frame'] = frame

            if not ret:
                row['p1_cx'] = row['p1_cy'] = row['p1x1'] = row['p1y1'] = row['p1x2'] = row['p1y2'] = np.nan
                row['p2_cx'] = row['p2_cy'] = row['p2x1'] = row['p2y1'] = row['p2x2'] = row['p2y2'] = np.nan
                data.append(row)
                continue
        
            results = model.predict(img, verbose=False)
            players_pos = []
            
            for result in results:
                if result.boxes is not None and len(result.boxes) > 0:
                    for box in result.boxes:
                        if int(box.cls) == 0: # 0 is person class
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                            feet_center_x = int((x1 + x2) / 2)
                            feet_center_y = int(y2)           
                            court_coords = pixel_to_court(feet_center_x, feet_center_y, H)

                            if is_on_court(court_coords[0], court_coords[1]):
                                players_pos.append([court_coords[0], court_coords[1], x1, y1, x2, y2])

            if len(players_pos) == 1:
                if playing_side == 1 and players_pos[0][1] >= 6.7:
                    p1_feet_x = players_pos[0][0]
                    p1_feet_y = 6.7 - (players_pos[0][1] - 6.7)
                    p1_cx, p1_cy = court_to_pixel(p1_feet_x, p1_feet_y, H)
                    
                    p2x1, p2y1, p2x2, p2y2 = players_pos[0][2:6]
                    p1y2 = p1_cy
                    p1x2 = p2x2
                    p1x1 = p2x1
                    p1y1 = p1y2 - (p2y2 - p2y1)
                    
                    players_pos.append([p1_feet_x, p1_feet_y, p1x1, p1y1, p1x2, p1y2])
                else:
                    players_pos.append([np.nan] * 6)

            if len(players_pos) < 2:
                row['p1_cx'] = row['p1_cy'] = row['p1x1'] = row['p1y1'] = row['p1x2'] = row['p1y2'] = np.nan
                row['p2_cx'] = row['p2_cy'] = row['p2x1'] = row['p2y1'] = row['p2x2'] = row['p2y2'] = np.nan
            else:
                players_pos.sort(key=lambda p: p[1])
                row['p1_cx'], row['p1_cy'], row['p1x1'], row['p1y1'], row['p1x2'], row['p1y2'] = players_pos[0]
                row['p2_cx'], row['p2_cy'], row['p2x1'], row['p2y1'], row['p2x2'], row['p2y2'] = players_pos[1]

            data.append(row)
        
        stroke_df = pd.DataFrame(data)
        def has_large_gap(series, max_gap=6):
            is_nan = series.isna()
            consecutive_nans = is_nan.groupby((~is_nan).cumsum()).sum()
            return consecutive_nans.max() > max_gap

        p1_bad = has_large_gap(stroke_df['p1_cx'], max_gap=6)
        p2_bad = has_large_gap(stroke_df['p2_cx'], max_gap=6)
        
        if p1_bad or p2_bad:
            print(f"Dropped Stroke {i+1} (Match {match_no}): Exceeded 6 consecutive missing frames.")
            continue 
        write_header = not os.path.exists(output_csv)
        stroke_df.to_csv(output_csv, mode='a', header=write_header, index=False)
        completed_strokes.add(current_stroke_id)
    
print(f"Tracking complete. Data saved to {output_csv}")