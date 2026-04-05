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
RALLY_PATH = os.path.join(os.curdir, 'output/rallies')
CONFIDENCE_THRESHOLD = 0.85
COOLDOWN_FRAMES = 5 # Added cooldown to prevent double-logging the same stroke
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
            last_point_no = int(last_row['point_no'])
            last_stroke_num = int(last_row['stroke_num'])
            last_hitter = int(last_row['playing_side'])
            print(f"Resuming Match {last_match_no} from Frame {resume_frame}...")
    except Exception as e:
        print(f"Could not read output csv. Starting Fresh. error: {e}")

video_files = [f for f in os.listdir(UNLABELED_VIDEO_DIR) if f.endswith('.mp4')]
print('Files ready for 1st pass')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
scaler = joblib.load(SCALER_PATH)

STROKE_NAMES = ['Block', 'Clear', 'Drive', 'Dropshot', 'Net-Kill', 'Net-Lift', 'Net-Shot', 'Serve', 'Smash']

model = BadmintonBiLSTM(input_size=106, hidden_size=64, num_layers=2, num_classes=len(STROKE_NAMES), dropout=0.30)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, model_complexity=1, min_detection_confidence=0.5)

def predict(window):
    window_df = pd.DataFrame(window)
    window_df = window_df.interpolate(method='linear', limit_direction='both')
    interpolated_window = window_df.to_numpy()
    window_scaled = scaler.transform(interpolated_window)
    X_tensor = torch.tensor(window_scaled, dtype=torch.float32).unsqueeze(0).to(device)
    
    result = model(X_tensor)

    probabilities = torch.nn.functional.softmax(result, dim=1)
    confidence, predicted_class = torch.max(probabilities, dim=1)

    return confidence.item(), predicted_class.item()

def extract_features(landmarks, prev_coords, prev_angles):
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
            point_no = last_point_no
            stroke_num = last_stroke_num + 1 
        else:
            point_no = 0
            stroke_num = 1
            last_hitter = 0

        H = matrices[video_path]
        rolling_window_p1 = deque(maxlen=30)
        rolling_window_p2 = deque(maxlen=30)
        
        tracking_peak1 = False
        tracking_peak2 = False
        max_conf1 = 0.0
        max_conf2 = 0.0
        best_stroke_id = -1
        peak_frame1 = 0
        peak_frame2 = 0
        
        # 0: open, 1: expect P1 to hit, 2: expect P2 to hit
        cooldown_timer = 0
        missing_frame = 0

        p1_prev_coords, p1_prev_angles = None, None
        p2_prev_coords, p2_prev_angles = None, None

        with open(os.path.join(RALLY_PATH, video.split('.')[0] + '.json'), 'r') as f:
            metadata = json.load(f)
        
        rallies = metadata['rally']

        for i, rally in enumerate(rallies):
            start_frame = int(rally[i][0] - 30)
            end_frame = int(rally[i][1] + 30)
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            current_pos = int(cap.get(cv2.CAP_PROP_POS_FRAMES))

            if current_pos != start_frame:
                # If we landed early, read and discard frames until we catch up
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

                row = {}
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

                if len(players_pos) < 2:
                    row['p1_cx'] = row['p1_cy'] = row['p1x1'] = row['p1y1'] = row['p1x2'] = row['p1y2'] = np.nan
                    row['p2_cx'] = row['p2_cy'] = row['p2x1'] = row['p2y1'] = row['p2x2'] = row['p2y2'] = np.nan
                    missing_frame += 1
                else:
                    players_pos.sort(key=lambda p: p[1])
                    row['p1_cx'], row['p1_cy'], row['p1x1'], row['p1y1'], row['p1x2'], row['p1y2'] = players_pos[0]
                    row['p2_cx'], row['p2_cy'], row['p2x1'], row['p2y1'], row['p2x2'], row['p2y2'] = players_pos[1]
                    missing_frame = 0
                
                # --- Extract Player 1 ---
                x_start, y_start, x_end, y_end = row['p1x1'], row['p1y1'], row['p1x2'], row['p1y2']
                if not np.isnan(x_start) and y_end > y_start and x_end > x_start:
                    p1_crop = frame[int(y_start):int(y_end), int(x_start):int(x_end)].copy()
                    resized_img = resize(p1_crop)
                    landmarks = pose.process(cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB)).pose_landmarks
                else:
                    landmarks = None
                    
                p1_feats, p1_prev_coords, p1_prev_angles = extract_features(landmarks, p1_prev_coords, p1_prev_angles)
                rolling_window_p1.append(p1_feats) # Append just the array, not the tuple!
                
                # --- Extract Player 2 ---
                x_start, y_start, x_end, y_end = row['p2x1'], row['p2y1'], row['p2x2'], row['p2y2']
                if not np.isnan(x_start) and y_end > y_start and x_end > x_start:
                    p2_crop = frame[int(y_start):int(y_end), int(x_start):int(x_end)].copy()
                    resized_img = resize(p2_crop)
                    landmarks = pose.process(cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB)).pose_landmarks
                else:
                    landmarks = None
                    
                p2_feats, p2_prev_coords, p2_prev_angles = extract_features(landmarks, p2_prev_coords, p2_prev_angles)
                rolling_window_p2.append(p2_feats)

                if len(rolling_window_p1) == 30 and missing_frame <= 6:
                    
                    if cooldown_timer > 0:
                        cooldown_timer -= 1
                        continue

                    # 1. ALWAYS predict both players
                    conf1, cls1 = predict(rolling_window_p1)
                    conf2, cls2 = predict(rolling_window_p2)

                    # 2. Update P1's Peak Tracker
                    if conf1 >= CONFIDENCE_THRESHOLD:
                        tracking_peak1 = True
                        if conf1 > max_conf1:
                            max_conf1 = conf1
                            best_cls1 = cls1
                            peak_frame1 = current_frame

                    # 3. Update P2's Peak Tracker
                    if conf2 >= CONFIDENCE_THRESHOLD:
                        tracking_peak2 = True
                        if conf2 > max_conf2:
                            max_conf2 = conf2
                            best_cls2 = cls2
                            peak_frame2 = current_frame

                    # 4. Detect when a swing FINISHES (confidence drops below threshold)
                    p1_finished_swing = tracking_peak1 and conf1 < CONFIDENCE_THRESHOLD
                    p2_finished_swing = tracking_peak2 and conf2 < CONFIDENCE_THRESHOLD

                    # If either player finished a swing, we process the stroke
                    if p1_finished_swing or p2_finished_swing:
                        
                        # Tie-Breaker: If both finish at the exact same time, highest peak wins
                        if p1_finished_swing and p2_finished_swing:
                            winner = 1 if max_conf1 > max_conf2 else 2
                        elif p1_finished_swing:
                            winner = 1
                        else:
                            winner = 2

                        # Extract the winner's data
                        stroke_type = STROKE_NAMES[best_cls1] if winner == 1 else STROKE_NAMES[best_cls2]
                        peak_frame = peak_frame1 if winner == 1 else peak_frame2
                        
                        # --- THE BACK-TO-BACK AUDIT ---
                        if winner == last_hitter:
                            print(f"⚠️ Back-to-Back hit by P{winner} detected. Resetting Rally!")
                            point_no += 1
                            stroke_num = 1
                        elif stroke_type == 'Serve':
                            # A serve always starts a new point
                            point_no += 1
                            stroke_num = 1

                        # --- LOG THE STROKE ---
                        stroke_data = {
                            'match_no': match_no, 'point_no': point_no, 'stroke_num': stroke_num,
                            'stroke_type': stroke_type, 'stroke_begin': peak_frame - 30,
                            'stroke_end': peak_frame, 'playing_side': winner,
                            'p1_cx': row['p1_cx'], 'p1_cy': row['p1_cy'], 'p1x1': row['p1x1'], 'p1y1': row['p1y1'], 'p1x2': row['p1x2'], 'p1y2': row['p1y2'],
                            'p2_cx': row['p2_cx'], 'p2_cy': row['p2_cy'], 'p2x1': row['p2x1'], 'p2y1': row['p2y1'], 'p2x2': row['p2x2'], 'p2y2': row['p2y2']
                        }
                        
                        pd.DataFrame([stroke_data]).to_csv(OUTPUT_CSV, mode='a', header=not os.path.exists(OUTPUT_CSV), index=False)
                        print(f"P{winner} Hit: {stroke_type} at Frame {peak_frame} (Point {point_no}, Stroke {stroke_num})")

                        stroke_num += 1
                        last_hitter = winner
                        cooldown_timer = COOLDOWN_FRAMES
                        
                        tracking_peak1 = False
                        tracking_peak2 = False
                        max_conf1 = 0.0
                        max_conf2 = 0.0    

                elif missing_frame > 6:
                    tracking_peak1 = False
                    tracking_peak2 = False

                current_frame += 1

        cap.release()
        
        print(f"Match {match_no} completely processed!")

print("Pass 1 Pipeline Complete.")