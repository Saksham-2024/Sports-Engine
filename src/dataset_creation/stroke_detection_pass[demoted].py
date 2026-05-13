import os
import cv2
import json
import torch
import joblib #type: ignore
import pickle
import numpy as np
import pandas as pd
import mediapipe as mp
from ultralytics import YOLO
from collections import deque
from teacher_model.lstm import BadmintonBiLSTM
from teacher_model.prep_dataset import angle_xy, angle_xz, angle_yz, vector, angle, dist, pixel_to_court, is_on_court, resize
from format_existing_dataset import court_to_pixel

MODEL_PATH = '../teacher_model/badminton_lstm_best.pth'
EXISTING_DATA_PATH = 'formatted_dataset.csv'
OUTPUT_CSV = 'expanded_dataset.csv'
SCALER_PATH = '../teacher_model/scaler.pkl' 
UNLABELED_VIDEO_DIR = os.path.join(os.curdir, 'unlabeled_videos')
RALLY_PATH = os.path.join(os.curdir, 'outputs/rallies')
HIT_FRAMES_PATH = os.path.join(os.curdir, 'outputs/joints')
CONFIDENCE_THRESHOLD = 0.85
epsilon = 1e-8
completed_strokes = set()
yolo = YOLO('yolov8m.pt')

homography_cache_file = 'homography_cache.pkl'
if os.path.exists(homography_cache_file):
    with open(homography_cache_file, 'rb') as f:
        matrices = pickle.load(f)
        print("Loaded homography matrices from cache! No clicking needed.")
else:
    print("Failed to load Homography matrices.") 

last_match_no = -1
resume_frame = 0
last_point_no = 0
last_stroke_num = 0

if os.path.exists(OUTPUT_CSV):
    try:
        existing_output_csv = pd.read_csv(OUTPUT_CSV)
        if not existing_output_csv.empty:
            last_row = existing_output_csv.iloc[-1]
            last_match_no = int(last_row['match_no'])
            resume_frame = int(last_row['stroke_end'])
            last_point_no = int(last_row['point_no'].replace('Point ', ''))
            last_stroke_num = int(last_row['stroke_num'])
            last_hitter = int(last_row['playing_side'])
            print(f"Resuming Match {last_match_no} from Frame {resume_frame}...")
    except Exception as e:
        print(f"Could not read output csv. Starting Fresh. error: {e}")

video_files = [f for f in os.listdir(UNLABELED_VIDEO_DIR) if f.endswith('.mp4')]
print('Files ready for 2nd pass')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
scaler = joblib.load(SCALER_PATH)

STROKE_NAMES = ['Block', 'Clear', 'Drive', 'Dropshot', 'Net-Lift', 'Net-Shot', 'Serve', 'Smash']

model = BadmintonBiLSTM(input_size=108, hidden_size=128, num_layers=2, num_classes=len(STROKE_NAMES), dropout=0.30)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, model_complexity=1, min_detection_confidence=0.5)

def predict(window):
    window_df = pd.DataFrame(window)
    window_df = window_df.interpolate(method='linear', limit_direction='both')
    window_df = window_df.fillna(0)
    interpolated_window = window_df.to_numpy()
    window_scaled = scaler.transform(interpolated_window)
    X_tensor = torch.tensor(window_scaled, dtype=torch.float32).unsqueeze(0).to(device)
    
    result = model(X_tensor)

    probabilities = torch.nn.functional.softmax(result, dim=1)
    confidence, predicted_class = torch.max(probabilities, dim=1)

    return confidence.item(), predicted_class.item()

def extract_features(landmarks, court_x, court_y, prev_coords, prev_angles):
    # FIXED: Handle missing landmarks explicitly to avoid crash
    if landmarks is None:
        return np.full(106, np.nan), prev_coords, prev_angles

    features = []
    row = {}
    invisible_landmarks = 0 # FIXED: Initialized variable
    
    for i, lm in enumerate(landmarks.landmark):
        if i <= 10:
            continue
        if lm.visibility < 0.05:
            invisible_landmarks += 1
            row[f'x{i}'] = np.nan
            row[f'y{i}'] = np.nan
            row[f'z{i}'] = np.nan
            row[f'v{i}'] = lm.visibility
        else:
            row[f'x{i}'] = lm.x
            row[f'y{i}'] = lm.y
            row[f'z{i}'] = lm.z
            row[f'v{i}'] = lm.visibility

    joints = {
        'shoulder': [11, 12], 'elbow': [13, 14], 'wrist': [15, 16],
        'hip': [23, 24], 'knee': [25, 26], 'ankle': [27, 28]
    }
    coords = {}
    vectors = {}
    angles = {}
    angular_velocities = {}
    velocities = {}
    dists = {}
    
    for joint_name, indices in joints.items():
        coords[f'left_{joint_name}_x'] = row[f'x{indices[0]}']
        coords[f'left_{joint_name}_y'] = row[f'y{indices[0]}']
        coords[f'left_{joint_name}_z'] = row[f'z{indices[0]}']
        coords[f'left_{joint_name}_v'] = row[f'v{indices[0]}']

        coords[f'right_{joint_name}_x'] = row[f'x{indices[1]}']
        coords[f'right_{joint_name}_y'] = row[f'y{indices[1]}']
        coords[f'right_{joint_name}_z'] = row[f'z{indices[1]}']
        coords[f'right_{joint_name}_v'] = row[f'v{indices[1]}']
    
    mid_hip_x = (coords['left_hip_x'] + coords['right_hip_x']) / 2
    mid_hip_y = (coords['left_hip_y'] + coords['right_hip_y']) / 2
    mid_hip_z = (coords['left_hip_z'] + coords['right_hip_z']) / 2
    
    mid_shoulder_x = (coords['left_shoulder_x'] + coords['right_shoulder_x']) / 2
    mid_shoulder_y = (coords['left_shoulder_y'] + coords['right_shoulder_y']) / 2
    mid_shoulder_z = (coords['left_shoulder_z'] + coords['right_shoulder_z']) / 2
    
    torso_length = dist(mid_hip_x, mid_hip_y, mid_hip_z, mid_shoulder_x, mid_shoulder_y, mid_shoulder_z)
    torso_length = max(torso_length, 1e-5) 

    # --- APPLY NORMALIZATION TO ALL COORDS ---
    for joint in joints:
        for lr in ['left', 'right']:
            coords[f'{lr}_{joint}_x'] = (coords[f'{lr}_{joint}_x'] - mid_hip_x) / torso_length
            coords[f'{lr}_{joint}_y'] = (coords[f'{lr}_{joint}_y'] - mid_hip_y) / torso_length
            coords[f'{lr}_{joint}_z'] = (coords[f'{lr}_{joint}_z'] - mid_hip_z) / torso_length

    # ... (Your exact vector and angle logic remains unchanged here) ...
    vectors['left_hip_shoulder'] = vector(coords['left_hip_x'], coords['left_hip_y'], coords['left_hip_z'], coords['left_shoulder_x'], coords['left_shoulder_y'], coords['left_shoulder_z'])
    vectors['right_hip_shoulder'] = vector(coords['right_hip_x'], coords['right_hip_y'], coords['right_hip_z'], coords['right_shoulder_x'], coords['right_shoulder_y'], coords['right_shoulder_z'])
    vectors['left_shoulder_elbow'] = vector(coords['left_shoulder_x'], coords['left_shoulder_y'], coords['left_shoulder_z'], coords['left_elbow_x'], coords['left_elbow_y'], coords['left_elbow_z'])
    vectors['right_shoulder_elbow'] = vector(coords['right_shoulder_x'], coords['right_shoulder_y'], coords['right_shoulder_z'], coords['right_elbow_x'], coords['right_elbow_y'], coords['right_elbow_z'])
    vectors['left_shoulder_hip'] = vectors['left_hip_shoulder'] * -1
    vectors['right_shoulder_hip'] = vectors['right_hip_shoulder'] * -1
    vectors['left_hip_knee'] = vector(coords['left_hip_x'], coords['left_hip_y'], coords['left_hip_z'], coords['left_knee_x'], coords['left_knee_y'], coords['left_knee_z'])
    vectors['right_hip_knee'] = vector(coords['right_hip_x'], coords['right_hip_y'], coords['right_hip_z'], coords['right_knee_x'], coords['right_knee_y'], coords['right_knee_z'])
    vectors['left_elbow_wrist'] = vector(coords['left_elbow_x'], coords['left_elbow_y'], coords['left_elbow_z'], coords['left_wrist_x'], coords['left_wrist_y'], coords['left_wrist_z'])
    vectors['right_elbow_wrist'] = vector(coords['right_elbow_x'], coords['right_elbow_y'], coords['right_elbow_z'], coords['right_wrist_x'], coords['right_wrist_y'], coords['right_wrist_z'])
    vectors['left_knee_ankle'] = vector(coords['left_knee_x'], coords['left_knee_y'], coords['left_knee_z'], coords['left_ankle_x'], coords['left_ankle_y'], coords['left_ankle_z'])
    vectors['right_knee_ankle'] = vector(coords['right_knee_x'], coords['right_knee_y'], coords['right_knee_z'], coords['right_ankle_x'], coords['right_ankle_y'], coords['right_ankle_z'])
    vectors['left_shoulder_wrist'] = vector(coords['left_shoulder_x'], coords['left_shoulder_y'], coords['left_shoulder_z'], coords['left_wrist_x'], coords['left_wrist_y'], coords['left_wrist_z'])
    vectors['right_shoulder_wrist'] = vector(coords['right_shoulder_x'], coords['right_shoulder_y'], coords['right_shoulder_z'], coords['right_wrist_x'], coords['right_wrist_y'], coords['right_wrist_z'])
    vectors['shoulders'] = vector(coords['left_shoulder_x'], coords['left_shoulder_y'], coords['left_shoulder_z'], coords['right_shoulder_x'], coords['right_shoulder_y'], coords['right_shoulder_z'])
    vectors['hips'] = vector(coords['left_hip_x'], coords['left_hip_y'], coords['left_hip_z'], coords['right_hip_x'], coords['right_hip_y'], coords['right_hip_z'])
    
    angles['left_hip_shoulder_elbow_xy_angle'] = angle_xy(vectors['left_hip_shoulder'], vectors['left_shoulder_elbow']) 
    angles['left_hip_shoulder_elbow_yz_angle'] = angle_yz(vectors['left_hip_shoulder'], vectors['left_shoulder_elbow']) 
    angles['left_hip_shoulder_elbow_xz_angle'] = angle_xz(vectors['left_hip_shoulder'], vectors['left_shoulder_elbow'])

    angles['right_hip_shoulder_elbow_xy_angle'] = angle_xy(vectors['right_hip_shoulder'], vectors['right_shoulder_elbow']) 
    angles['right_hip_shoulder_elbow_yz_angle'] = angle_yz(vectors['right_hip_shoulder'], vectors['right_shoulder_elbow']) 
    angles['right_hip_shoulder_elbow_xz_angle'] = angle_xz(vectors['right_hip_shoulder'], vectors['right_shoulder_elbow'])  

    angles['left_shoulder_hip_knee_xy_angle'] = angle_xy(vectors['left_shoulder_hip'], vectors['left_hip_knee'])
    angles['left_shoulder_hip_knee_yz_angle'] = angle_yz(vectors['left_shoulder_hip'], vectors['left_hip_knee'])
    angles['left_shoulder_hip_knee_xz_angle'] = angle_xz(vectors['left_shoulder_hip'], vectors['left_hip_knee'])

    angles['right_shoulder_hip_knee_xy_angle'] = angle_xy(vectors['right_shoulder_hip'], vectors['right_hip_knee'])
    angles['right_shoulder_hip_knee_yz_angle'] = angle_yz(vectors['right_shoulder_hip'], vectors['right_hip_knee'])
    angles['right_shoulder_hip_knee_xz_angle'] = angle_xz(vectors['right_shoulder_hip'], vectors['right_hip_knee'])

    angles['left_shoulder_elbow_wrist_angle'] = angle(vectors['left_shoulder_elbow'], vectors['left_elbow_wrist'])
    angles['right_shoulder_elbow_wrist_angle'] = angle(vectors['right_shoulder_elbow'], vectors['right_elbow_wrist'])

    angles['left_hip_knee_ankle_angle'] = angle(vectors['left_hip_knee'], vectors['left_knee_ankle'])
    angles['right_hip_knee_ankle_angle'] = angle(vectors['right_hip_knee'], vectors['right_knee_ankle'])

    angles['torso_rotation_angle'] = angle(vectors['shoulders'], vectors['hips'])

    angles['left_hip_shoulder_wrist_angle'] = angle(vectors['left_hip_shoulder'], vectors['left_shoulder_wrist'])
    angles['right_hip_shoulder_wrist_angle'] = angle(vectors['right_hip_shoulder'], vectors['right_shoulder_wrist'])

    if prev_angles is None or prev_coords is None:
        for deg in angles:
            angular_velocities[f'{deg}_vel'] = np.nan
        velocities['left_wrist_velocity'] = velocities['right_wrist_velocity'] = np.nan
        velocities['left_elbow_velocity'] = velocities['right_elbow_velocity'] = np.nan
        velocities['left_hip_velocity'] = velocities['right_hip_velocity'] = np.nan
        velocities['left_shoulder_velocity'] = velocities['right_shoulder_velocity'] = np.nan
        velocities['left_rel_wrist_velocity'] = velocities['right_rel_wrist_velocity'] = np.nan
        velocities['left_rel_elbow_velocity'] = velocities['right_rel_elbow_velocity'] = np.nan
    else:
        for deg in angles:
            angular_velocities[f'{deg}_vel'] = (angles[deg] - prev_angles[deg]) * 25
    
        velocities['left_wrist_velocity'] = dist(prev_coords['left_wrist_x'], prev_coords['left_wrist_y'], prev_coords['left_wrist_z'], coords['left_wrist_x'], coords['left_wrist_y'], coords['left_wrist_z']) * 25
        velocities['right_wrist_velocity'] = dist(prev_coords['right_wrist_x'], prev_coords['right_wrist_y'], prev_coords['right_wrist_z'], coords['right_wrist_x'], coords['right_wrist_y'], coords['right_wrist_z']) * 25
        velocities['left_elbow_velocity'] = dist(prev_coords['left_elbow_x'], prev_coords['left_elbow_y'], prev_coords['left_elbow_z'], coords['left_elbow_x'], coords['left_elbow_y'], coords['left_elbow_z']) * 25
        velocities['right_elbow_velocity'] = dist(prev_coords['right_elbow_x'], prev_coords['right_elbow_y'], prev_coords['right_elbow_z'], coords['right_elbow_x'], coords['right_elbow_y'], coords['right_elbow_z']) * 25
        velocities['left_hip_velocity'] = dist(prev_coords['left_hip_x'], prev_coords['left_hip_y'], prev_coords['left_hip_z'], coords['left_hip_x'], coords['left_hip_y'], coords['left_hip_z']) * 25
        velocities['right_hip_velocity'] = dist(prev_coords['right_hip_x'], prev_coords['right_hip_y'], prev_coords['right_hip_z'], coords['right_hip_x'], coords['right_hip_y'], coords['right_hip_z']) * 25
        velocities['left_shoulder_velocity'] = dist(prev_coords['left_shoulder_x'], prev_coords['left_shoulder_y'], prev_coords['left_shoulder_z'], coords['left_shoulder_x'], coords['left_shoulder_y'], coords['left_shoulder_z']) * 25
        velocities['right_shoulder_velocity'] = dist(prev_coords['right_shoulder_x'], prev_coords['right_shoulder_y'], prev_coords['right_shoulder_z'], coords['right_shoulder_x'], coords['right_shoulder_y'], coords['right_shoulder_z']) * 25
        velocities['left_rel_wrist_velocity'] = velocities['left_wrist_velocity'] - velocities['left_elbow_velocity']
        velocities['right_rel_wrist_velocity'] = velocities['right_wrist_velocity'] - velocities['right_elbow_velocity']
        velocities['left_rel_elbow_velocity'] = velocities['left_elbow_velocity'] - velocities['left_shoulder_velocity']
        velocities['right_rel_elbow_velocity'] = velocities['right_elbow_velocity'] - velocities['right_shoulder_velocity']
    
    dists['left_hip_wrist_dist'] = dist(coords['left_hip_x'], coords['left_hip_y'], coords['left_hip_z'], coords['left_wrist_x'], coords['left_wrist_y'], coords['left_wrist_z'])
    dists['right_hip_wrist_dist'] = dist(coords['right_hip_x'], coords['right_hip_y'], coords['right_hip_z'], coords['right_wrist_x'], coords['right_wrist_y'], coords['right_wrist_z'])
    dists['wrist_to_wrist_dist'] = dist(coords['left_wrist_x'], coords['left_wrist_y'], coords['left_wrist_z'], coords['right_wrist_x'], coords['right_wrist_y'], coords['right_wrist_z'])
    dists['ankle_to_ankle_dist'] = dist(coords['left_ankle_x'], coords['left_ankle_y'], coords['left_ankle_z'], coords['right_ankle_x'], coords['right_ankle_y'], coords['right_ankle_z'])
    dists['left_shoulder_wrist_dist'] = dist(coords['left_shoulder_x'], coords['left_shoulder_y'], coords['left_shoulder_z'], coords['left_wrist_x'], coords['left_wrist_y'], coords['left_wrist_z'])
    dists['right_shoulder_wrist_dist'] = dist(coords['right_shoulder_x'], coords['right_shoulder_y'], coords['right_shoulder_z'], coords['right_wrist_x'], coords['right_wrist_y'], coords['right_wrist_z'])
    dists['left_wrist_y_shoulder_y_dist'] = coords['left_wrist_y'] - coords['left_shoulder_y']
    dists['right_wrist_y_shoulder_y_dist'] = coords['right_wrist_y'] - coords['right_shoulder_y']

    def vectorize(feature_dict):
        for k, v in feature_dict.items():
            features.append(v)
    
    features.append(court_x)
    features.append(court_y)
    vectorize(coords)
    vectorize(angles)
    vectorize(angular_velocities)
    vectorize(velocities)
    vectorize(dists)

    prev_coords = coords.copy()
    prev_angles = angles.copy()
    
    return np.array(features), prev_coords, prev_angles

with torch.no_grad():
    for video in video_files:
        video_path = os.path.join(UNLABELED_VIDEO_DIR, video)
        match_no = int(video.replace('.mp4', ''))

        point_no = 0
        stroke_num = 1
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error processing match{video}.")
            continue
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        if match_no < last_match_no:
            print(f"Skipping Match {match_no}. Already processed.")
            continue
        elif match_no == last_match_no:
            print(f"Fast-forwarding Match {match_no} to frame {resume_frame}...")
            cap.set(cv2.CAP_PROP_POS_FRAMES, resume_frame)
            current_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            while current_frame < resume_frame:
                ret, _ = cap.read()
                if not ret:
                    break
                current_frame += 1
            point_no = last_point_no
            stroke_num = last_stroke_num + 1 
        else:
            point_no = 0
            stroke_num = 1
            last_hitter = 0

        H = matrices[video_path]
        rolling_window_p1 = deque(maxlen=30)
        rolling_window_p2 = deque(maxlen=30)
        
        # 0: open, 1: expect P1 to hit, 2: expect P2 to hit
        missing_frame = 0

        p1_prev_coords, p1_prev_angles = None, None
        p2_prev_coords, p2_prev_angles = None, None

        with open(os.path.join(RALLY_PATH, video.split('.')[0] + '.json'), 'r') as f:
            metadata = json.load(f)
        
        rallies = metadata['rally']
        
        for i, rally in enumerate(rallies):
            rows = {}
            start_frame = int(rally[0] - 15)
            end_frame = int(rally[1] + 14)
            rally_id = rally[2]
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            current_pos = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            joints_file_path = os.path.join(HIT_FRAMES_PATH, f"{video.split('.')[0]},rally_{rally_id}.json")
            with open(joints_file_path, 'r') as f:
                joints_data = json.load(f)
                
            hit_frames = joints_data['hit frames']
            hit_frame_id = 0

            while current_pos < start_frame:
                ret, _ = cap.read()
                if not ret: break
                current_pos += 1
        
            current_frame = start_frame
            while cap.isOpened() and current_frame <= end_frame:
                ret, frame = cap.read()
                
                if not ret:
                    break
                    
                players_pos = []
                results = yolo.predict(frame, verbose=False)
                
                for result in results:
                    if result.boxes is not None and len(result.boxes) > 0:
                        for box in result.boxes:
                            if int(box.cls) == 0:
                                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                                feet_center_x = int((x1 + x2) / 2)
                                feet_center_y = int(y2)
                                court_coords = pixel_to_court(feet_center_x, feet_center_y, H)
                                court_x, court_y = court_coords[0], court_coords[1]
                                if is_on_court(court_x, court_y):
                                    players_pos.append((court_x, court_y, x1, y1, x2, y2))

                if len(players_pos) == 1:
                    # Interpolate missing player logic
                    if last_hitter == 1 and players_pos[0][1] >= 6.7:
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
                
                row = {}
                if len(players_pos) == 0:
                    row['p1_cx'] = row['p1_cy'] = row['p1x1'] = row['p1y1'] = row['p1x2'] = row['p1y2'] = np.nan
                    row['p2_cx'] = row['p2_cy'] = row['p2x1'] = row['p2y1'] = row['p2x2'] = row['p2y2'] = np.nan
                    missing_frame += 1
                else:
                    players_pos.sort(key=lambda p: p[1])
                    row['p1_cx'], row['p1_cy'], row['p1x1'], row['p1y1'], row['p1x2'], row['p1y2'] = players_pos[0]
                    row['p2_cx'], row['p2_cy'], row['p2x1'], row['p2y1'], row['p2x2'], row['p2y2'] = players_pos[1]
                    missing_frame = 0
                
                rows[current_frame] = row
                if current_frame >= hit_frames[hit_frame_id] - 15 and current_frame <= hit_frames[hit_frame_id] + 14:
                    print(f"Frame {current_frame} is within hit frame {hit_frames[hit_frame_id]} for rally {rally_id}.")
                    
                    # --- Extract Player 1 for stroke detection---
                    x_start, y_start, x_end, y_end = row['p1x1'], row['p1y1'], row['p1x2'], row['p1y2']
                    if not np.isnan(x_start) and y_end > y_start and x_end > x_start:
                        p1_crop = frame[int(y_start):int(y_end), int(x_start):int(x_end)].copy()
                        resized_img = resize(p1_crop)
                        landmarks = pose.process(cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB)).pose_landmarks
                    else:
                        landmarks = None
                        
                    p1_feats, p1_prev_coords, p1_prev_angles = extract_features(landmarks,row['p1_cx'], row['p1_cy'], p1_prev_coords, p1_prev_angles)
                    rolling_window_p1.append(p1_feats) # Append just the array, not the tuple!
                    
                    # --- Extract Player 2 for stroke detection---
                    x_start, y_start, x_end, y_end = row['p2x1'], row['p2y1'], row['p2x2'], row['p2y2']
                    if not np.isnan(x_start) and y_end > y_start and x_end > x_start:
                        p2_crop = frame[int(y_start):int(y_end), int(x_start):int(x_end)].copy()
                        resized_img = resize(p2_crop)
                        landmarks = pose.process(cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB)).pose_landmarks
                    else:
                        landmarks = None
                        
                    p2_feats, p2_prev_coords, p2_prev_angles = extract_features(landmarks,row['p2_cx'], row['p2_cy'], p2_prev_coords, p2_prev_angles)
                    rolling_window_p2.append(p2_feats)
                    
                    if current_frame == hit_frames[hit_frame_id] + 14:
                        
                        if len(rolling_window_p1) == 30 and missing_frame <= 6:
                            conf1, cls1 = predict(rolling_window_p1)
                            conf2, cls2 = predict(rolling_window_p2)

                            winner = 1 if (conf1 >= CONFIDENCE_THRESHOLD and conf1 > conf2) else (2 if conf2 >= CONFIDENCE_THRESHOLD else 0)

                            if winner > 0:
                                stroke_type = STROKE_NAMES[cls1] if winner == 1 else STROKE_NAMES[cls2]
                                
                                if winner == last_hitter:
                                    print(f"⚠️ Back-to-Back hit by P{winner}. Resetting Rally!")
                                    point_no += 1
                                    stroke_num = 1
                                elif stroke_type == 'Serve':
                                    point_no += 1
                                    stroke_num = 1

                                # Log the successful stroke
                                for i in range(30):
                                    f_idx = hit_frames[hit_frame_id] - 15 + i
                                    stroke_data = {
                                        'match_no': match_no, 'point_no': point_no, 'stroke_num': stroke_num,
                                        'frame': f_idx, 'stroke_type': stroke_type, 'playing_side': winner,
                                        'p1_cx': rows[f_idx]['p1_cx'], 'p1_cy': rows[f_idx]['p1_cy'], 'p1x1': rows[f_idx]['p1x1'], 'p1y1': rows[f_idx]['p1y1'], 'p1x2': rows[f_idx]['p1x2'], 'p1y2': rows[f_idx]['p1y2'],
                                        'p2_cx': rows[f_idx]['p2_cx'], 'p2_cy': rows[f_idx]['p2_cy'], 'p2x1': rows[f_idx]['p2x1'], 'p2y1': rows[f_idx]['p2y1'], 'p2x2': rows[f_idx]['p2x2'], 'p2y2': rows[f_idx]['p2y2']
                                    }
                                    pd.DataFrame([stroke_data]).to_csv(OUTPUT_CSV, mode='a', header=not os.path.exists(OUTPUT_CSV), index=False)
                                
                                print(f"P{winner} Hit: {stroke_type} at Frame {hit_frames[hit_frame_id]} (Point {point_no}, Stroke {stroke_num})")
                                stroke_num += 1
                                last_hitter = winner
                            else:
                                # Confidences were too low, log as 'none' to prevent timeline holes
                                pass_to_none_logger = True
                        else:
                            # Missing too many frames or deque incomplete, log as 'none'
                            pass_to_none_logger = True
                            
                        # 🟢 THE FALLBACK LOGGER (Prevents the Silent Drop hole)
                        if 'pass_to_none_logger' in locals() and pass_to_none_logger:
                            for i in range(30):
                                f_idx = hit_frames[hit_frame_id] - 15 + i
                                stroke_data = {
                                    'match_no': match_no, 'point_no': point_no, 'stroke_num': -1,
                                    'frame': f_idx, 'stroke_type': 'none', 'playing_side': 0,
                                    'p1_cx': rows[f_idx]['p1_cx'], 'p1_cy': rows[f_idx]['p1_cy'], 'p1x1': rows[f_idx]['p1x1'], 'p1y1': rows[f_idx]['p1y1'], 'p1x2': rows[f_idx]['p1x2'], 'p1y2': rows[f_idx]['p1y2'],
                                    'p2_cx': rows[f_idx]['p2_cx'], 'p2_cy': rows[f_idx]['p2_cy'], 'p2x1': rows[f_idx]['p2x1'], 'p2y1': rows[f_idx]['p2y1'], 'p2x2': rows[f_idx]['p2x2'], 'p2y2': rows[f_idx]['p2y2']
                                }
                                pd.DataFrame([stroke_data]).to_csv(OUTPUT_CSV, mode='a', header=not os.path.exists(OUTPUT_CSV), index=False)
                            print(f"Stroke at {hit_frames[hit_frame_id]} failed quality check. Logged 30 frames as 'none'.")
                            del pass_to_none_logger # cleanup for next loop
                            
                        # Move to the next hit frame in the JSON array
                        hit_frame_id = min(hit_frame_id + 1, len(hit_frames) - 1)

                else: 
                    stroke_data = {
                        'match_no': match_no, 'point_no': point_no, 'stroke_num': -1,
                        'frame': current_frame,
                        'stroke_type': 'none', 'playing_side': 0,
                        'p1_cx': rows[current_frame]['p1_cx'], 'p1_cy': rows[current_frame]['p1_cy'], 'p1x1': rows[current_frame]['p1x1'], 'p1y1': rows[current_frame]['p1y1'], 'p1x2': rows[current_frame]['p1x2'], 'p1y2': rows[current_frame]['p1y2'],
                        'p2_cx': rows[current_frame]['p2_cx'], 'p2_cy': rows[current_frame]['p2_cy'], 'p2x1': rows[current_frame]['p2x1'], 'p2y1': rows[current_frame]['p2y1'], 'p2x2': rows[current_frame]['p2x2'], 'p2y2': rows[current_frame]['p2y2']
                    }
                    
                    pd.DataFrame([stroke_data]).to_csv(OUTPUT_CSV, mode='a', header=not os.path.exists(OUTPUT_CSV), index=False)
                    print(f"P0 Hit: nothing at Frame {current_frame} (Point {point_no}, Stroke -1)")
                

                current_frame += 1

        cap.release()
        
        print(f"Match {match_no} completely processed!")

print("Pass 1 Pipeline Complete.")