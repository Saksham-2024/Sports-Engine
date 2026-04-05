import numpy as np
import pandas as pd
import cv2
import mediapipe as mp
import os
from ultralytics import YOLO
import pickle

dataset_dir = '../dataset'
video_dir = os.path.join(dataset_dir, 'videos')

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
model = YOLO('yolov8m.pt')

df = pd.read_csv('1-timestamps.csv')
grouped = df.groupby(['match_no'])
output_csv = '2-pose_dataset(raw).csv'
# avg shot duration - 30 frames

# Targeting Config_____________________________________________________________________________

TARGETED_MATCHES = {19, 20, 21}           # ← set your 2-3 match numbers here
TARGET_STROKES   = {'Drive', 'Net-Kill', 'Clear', 'Dropshot'}   # ← stroke types to extract from those matches

# _____________________________________________________________________________________________

completed_strokes = set()
if os.path.exists(output_csv):
    try:
        existing_csv = pd.read_csv(output_csv, usecols=['match_no', 'point_no', 'stroke_num'])
        completed_subset = existing_csv.drop_duplicates()
        completed_strokes = set(tuple(x) for x in completed_subset.to_numpy())
        print(f"Resuming... Found {len(completed_strokes)} completely processed strokes in CSV.")
    except Exception as e:
        print(f"Starting fresh. Could not read output CSV: {e}")


data = []

def pixel_to_court(x, y, H):
    point = np.array([[[x, y]]], dtype='float32')
    transformed = cv2.perspectiveTransform(point, H)
    return transformed[0][0]

def is_on_court(court_x, court_y, court_width=5.18, court_length=13.4):
        return 0 <= court_x <= court_width and 0 <= court_y <= court_length

def resize(img, target_size=640):
    h, w = img.shape[:2]
    scale = target_size / max(h, w)
    new_w, new_h = int(w * scale), int(h * scale)
    resized_img = cv2.resize(img, (new_w, new_h))
    pad_top = (target_size - new_h) // 2
    pad_bottom = target_size - new_h - pad_top
    pad_left = (target_size - new_w) // 2
    pad_right = target_size - new_w - pad_left
    resized_img = cv2.copyMakeBorder(resized_img, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    return resized_img

def homography(time, video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening video file: {video_path}")
        return False, np.zeros((3, 3), dtype=np.float64)

    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, time)
        ret, frame = cap.read()
        if not ret:
            print(f"Error reading frame {time} from video {video_path} for homography setup")
            return False, np.zeros((3, 3), dtype=np.float64)

        screen_w, screen_h = 1280, 720
        h, w = frame.shape[:2]
        scale = min(screen_w / w, screen_h / h)
        display_w, display_h = int(w * scale), int(h * scale)
        display_frame = cv2.resize(frame, (display_w, display_h))
        
        points = []

        # Click the 4 corners of the court        
        # Order: top-left, top-right, bottom-right, bottom-left

        def click_event(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                points.append((x, y))
                cv2.circle(display_frame, (x, y), 5, (0, 0, 255), -1)
                cv2.imshow('Mark 4 Court Corners', display_frame)

        cv2.imshow('Mark 4 Court Corners', display_frame)
        cv2.setMouseCallback('Mark 4 Court Corners', click_event)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        print("Selected corner points (camera view):", points)

        if len(points) != 4:
            print("mark only 4 points in the order - top-left, top-right, bottom-right, bottom-left")
            return False, np.zeros((3, 3), dtype=np.float64)

        # Define the real-world coordinates (in meters)
        real_world_court = np.array([
            [0, 0],        # top-left
            [5.18, 0],     # top-right
            [5.18, 13.4],  # bottom-right
            [0, 13.4]      # bottom-left
        ], dtype=np.float32)

        # Compute homography matrix
        image_points = np.array(points, dtype=np.float64)
        H, _ = cv2.findHomography(image_points, real_world_court)
        print("\nHomography matrix (H):\n", H)
        return True, H

    except Exception as e:
        print(f"Error during homography setup for match: {e}")
        return False, np.zeros((3, 3), dtype=np.float64)

homography_cache_file = 'homography_cache.pkl'
matrices = {}

if os.path.exists(homography_cache_file):
    with open(homography_cache_file, 'rb') as f:
        matrices = pickle.load(f)
        print(f"Loaded {len(matrices)} homography matrices from cache.")

cache_updated = False

for match, group in grouped:
    video_path = video_dir + f'/match{match[0]}.mp4'
    if video_path not in matrices:
        print(f"\n New match detected: {match[0]}. Opening camera view for calibration...")
        
        i = 0
        while i < len(group):
            if group.iloc[i]['camera'] != 'normal':
                i += 1
            else:
                start_frame = int(group.iloc[i]['stroke_begin'])
                ret, H = homography(start_frame, video_path)
                
                if ret:
                    matrices[video_path] = H
                    cache_updated = True
                    break
                else:
                    print(f'Error calculating homography for match {match[0]}, moving to next normal frame.')
                    i += 1

if cache_updated:
    with open(homography_cache_file, 'wb') as f:
        pickle.dump(matrices, f)
    print("\n✅ Homography cache updated and saved with new matches.")
else:
    print("\n✅ All matches in the CSV are already calibrated. Proceeding to extraction...")


for match, group in grouped:
    match_no = match[0]
    video_path = video_dir + f'/match{match[0]}.mp4'
    if video_path not in matrices:
        continue

    # ── Targeted match: print what will be skipped, then filter group ─────────
    is_targeted = match_no in TARGETED_MATCHES
    if is_targeted:
        full_count     = len(group)
        filtered_group = group[group['stroke_type'].isin(TARGET_STROKES)]
        skip_count     = full_count - len(filtered_group)
        print(f"\nMatch {match_no} is TARGETED — extracting only {TARGET_STROKES}.")
        print(f"  {len(filtered_group)} qualifying strokes | {skip_count} strokes skipped.")
        group = filtered_group
        if len(group) == 0:
            print(f"  No qualifying strokes found in match {match_no}, skipping entirely.")
            continue
    
    H = matrices[video_path]
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening video file: {video_path}")
        continue
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    for i in range(len(group)):
        stroke_begin = int(group.iloc[i]['stroke_begin'])
        stroke_end = int(group.iloc[i]['stroke_end'])
        stroke_type = group.iloc[i]['stroke_type']
        playing_side = group.iloc[i]['playing_side']
        stroke_num = group.iloc[i]['stroke_num']
        point_no = group.iloc[i]['point_no']

        current_stroke_id = (match[0], point_no, stroke_num)
        if current_stroke_id in completed_strokes:
            print(f"Skipping Match {match[0]}, Point {point_no}, Stroke {stroke_num} - Already complete.")
            continue
        
        no_frames = stroke_end - stroke_begin
        if no_frames <= 30:
            diff = 30 - no_frames
            stroke_begin = max(0, stroke_begin - diff // 2)
            stroke_end = stroke_end + diff // 2
        elif no_frames > 30:
            stroke_begin = stroke_begin + (no_frames - 30) // 2
            stroke_end = stroke_end - (no_frames - 30) // 2
        
        # to cover edge cases where adjusting stroke begin and stroke end causes off by one errors
        while stroke_end - stroke_begin < 30:
            stroke_end += 1
        
        while stroke_end - stroke_begin > 30:
            stroke_end -= 1

        # Extract frames around the stroke
        missing_frames = 0
        stroke_data = []
        last_valid_box = None
        start_frame = int(stroke_begin)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        current_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)
        while current_frame < start_frame:
            ret, _ = cap.read()
            if not ret:
                break
            current_frame += 1
        frame_no = current_frame

        def handle_missing_frame(frame_no):
            flattened_landmarks = {
                "match_no": match[0],
                "point_no": group.iloc[i]['point_no'],
                "stroke_num": group.iloc[i]['stroke_num'],
                "frame_no": frame_no,
                "stroke_type": stroke_type,
                "playing_side": playing_side,
                "court_x": np.nan,
                "court_y": np.nan
            }
            for j in range(11,33):
                flattened_landmarks[f'x{j}'] = np.nan
                flattened_landmarks[f'y{j}'] = np.nan
                flattened_landmarks[f'z{j}'] = np.nan
                flattened_landmarks[f'v{j}'] = np.nan
            stroke_data.append(flattened_landmarks)

        while frame_no < stroke_end:    
            ret, img = cap.read()
            if not ret:
                handle_missing_frame(frame_no)
                missing_frames += 1
                frame_no += 1
                continue

            results = model.predict(img, verbose = False)
            players_pos = []
            for result in results:
                if result.boxes is not None and len(result.boxes) > 0:
                    for box in result.boxes:
                        if int(box.cls[0]) == 0:  # Player
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                            feet_center_x = int((x1 + x2) / 2)
                            feet_center_y = int(y2)

                            court_coords = pixel_to_court(feet_center_x, feet_center_y, H)
                            court_x, court_y = court_coords[0], court_coords[1]
                            if is_on_court(court_x, court_y):
                                players_pos.append((court_x, court_y, x1, y1, x2, y2))
            
            x_start, y_start, x_end, y_end = None, None, None, None
            if len(players_pos) < 2:
                if (len(players_pos) == 1 and ((playing_side == 1 and players_pos[0][1] >= 6.7) or (playing_side == 2 and players_pos[0][1] <= 6.7))) or len(players_pos) == 0:
                    missing_frames += 1
                    handle_missing_frame(frame_no)
                    frame_no += 1
                    continue
                    
                else:
                    x_start, y_start, x_end, y_end = players_pos[0][2], players_pos[0][3], players_pos[0][4], players_pos[0][5]
                    selected_court_x, selected_court_y = players_pos[0][0], players_pos[0][1]

            else:    
                players_pos.sort(key=lambda p: p[1])
                if group.iloc[i]['playing_side'] == 1:
                    x_start, y_start, x_end, y_end = players_pos[0][2], players_pos[0][3], players_pos[0][4], players_pos[0][5]
                    selected_court_x, selected_court_y = players_pos[0][0], players_pos[0][1]   
                else:
                    x_start, y_start, x_end, y_end = players_pos[1][2], players_pos[1][3], players_pos[1][4], players_pos[1][5]   
                    selected_court_x, selected_court_y = players_pos[1][0], players_pos[1][1]  
            
            if y_end > y_start and x_end > x_start:
                last_valid_box = (x_start, y_start, x_end, y_end)
                img = img[int(y_start):int(y_end), int(x_start):int(x_end)]
            else:
                if last_valid_box is not None:
                    x_start, y_start, x_end, y_end = last_valid_box
                    img = img[int(y_start):int(y_end), int(x_start):int(x_end)]
                else:
                    img = np.zeros((640, 640, 3), dtype=np.uint8)            
            
            resized_img = resize(img)
            landmarks = pose.process(cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB)).pose_landmarks

            if landmarks is not None:
                flattened_landmarks = {
                    "match_no": match[0],
                    "point_no": group.iloc[i]['point_no'],
                    "stroke_num": group.iloc[i]['stroke_num'],
                    "frame_no": frame_no,
                    "stroke_type": stroke_type,
                    "playing_side": playing_side,
                    "court_x": selected_court_x,
                    "court_y": (6.7 - selected_court_y) if playing_side == 1 else (selected_court_y - 6.7)
                }
                
                invisible_landmarks = 0
                for k, lm in enumerate(landmarks.landmark):
                    if k <= 10:
                        continue
                    if lm.visibility < 0.05:
                        invisible_landmarks += 1
                        flattened_landmarks[f'x{k}'] = np.nan
                        flattened_landmarks[f'y{k}'] = np.nan
                        flattened_landmarks[f'z{k}'] = np.nan
                        flattened_landmarks[f'v{k}'] = lm.visibility
                    else:
                        flattened_landmarks[f'x{k}'] = lm.x
                        flattened_landmarks[f'y{k}'] = lm.y
                        flattened_landmarks[f'z{k}'] = lm.z
                        flattened_landmarks[f'v{k}'] = lm.visibility

                if invisible_landmarks >= 6:
                    missing_frames += 1

                stroke_data.append(flattened_landmarks)

            else:
                missing_frames += 1
                handle_missing_frame(frame_no)
            
            print(f'processed frame_no {frame_no}')
            frame_no += 1
            
        if len(stroke_data) == 30 and missing_frames <= 6:
            stroke_df = pd.DataFrame(stroke_data)
            write_header = not os.path.exists(output_csv)
            stroke_df.to_csv(output_csv, mode='a', header=write_header, index=False)
            
            completed_strokes.add(current_stroke_id)
            print('successful')
        else:
            print('failed')

        print(f'processing of match_no {match[0]}, point_no {point_no}, stroke_num {stroke_num}')
