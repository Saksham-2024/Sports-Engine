import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv('3-interpolated_dataset_raw.csv')
output_csv = '4.1-engineered_dataset_smoothed.csv'
epsilon = 1e-8

completed_strokes = set()
if os.path.exists(output_csv):
    try:
        existing_df = pd.read_csv(output_csv, usecols=['match_no', 'playing_side', 'point_no', 'stroke_num'])
        completed_subset = existing_df.drop_duplicates()
        completed_strokes = set(tuple(x) for x in completed_subset.to_numpy())
        print(f"Resuming... Found {len(completed_strokes)} completely processed strokes in CSV.")
    except Exception as e:
        print(f"Starting fresh. Could not read output CSV: {e}")

data = []

def vector(x1, y1, z1, x2, y2, z2):
    return np.array([x2 - x1, y2 - y1, z2 - z1])

def angle_xy(vec1, vec2):
    v1, v2 = np.copy(vec1), np.copy(vec2)
    v1[2], v2[2] = 0, 0
    dot = np.dot(v1, v2)
    mag1, mag2 = np.linalg.norm(v1), np.linalg.norm(v2)
    cos = np.clip(dot / (mag1 * mag2 + epsilon), -1.0, 1.0)
    radians = np.arccos(cos)
    degrees = np.degrees(radians)
    return degrees

def angle_yz(vec1, vec2):
    v1, v2 = np.copy(vec1), np.copy(vec2)
    v1[0], v2[0] = 0, 0
    dot = np.dot(v1, v2)
    mag1, mag2 = np.linalg.norm(v1), np.linalg.norm(v2)
    cos = np.clip(dot / (mag1 * mag2 + epsilon), -1.0, 1.0)
    radians = np.arccos(cos)
    degrees = np.degrees(radians)
    return degrees

def angle_xz(vec1, vec2):
    v1, v2 = np.copy(vec1), np.copy(vec2)
    v1[1], v2[1] = 0, 0
    dot = np.dot(v1, v2)
    mag1, mag2 = np.linalg.norm(v1), np.linalg.norm(v2)
    cos = np.clip(dot / (mag1 * mag2 + epsilon), -1.0, 1.0)
    radians = np.arccos(cos)
    degrees = np.degrees(radians)
    return degrees

def angle(v1, v2):
    dot = np.dot(v1, v2)
    mag1, mag2 = np.linalg.norm(v1), np.linalg.norm(v2)
    cos = np.clip(dot / (mag1 * mag2 + epsilon), -1.0, 1.0)
    radians = np.arccos(cos)
    degrees = np.degrees(radians)
    return degrees

def dist(x1, y1, z1, x2, y2, z2):
    return ((x2 - x1) ** 2 + (y2 - y1) ** 2 + (z2 - z1) ** 2) ** 0.5

handedness_map = {}
grouped = df.groupby(['match_no', 'playing_side'])
def hand():
    for (match, player), group in grouped:
        var_wrist_left = 0
        var_wrist_left += group['x15'].var()
        var_wrist_left += group['y15'].var()
        var_wrist_left += group['z15'].var()    

        var_wrist_right = 0
        var_wrist_right += group['x16'].var()
        var_wrist_right += group['y16'].var()
        var_wrist_right += group['z16'].var()

        handedness_map[(match, player)] = 'right' if var_wrist_right > var_wrist_left else 'left'

hand()
joints = {
    'shoulder': [11, 12],
    'elbow': [13, 14],
    'wrist': [15, 16],
    'hip': [23, 24],
    'knee': [25, 26],
    'ankle': [27, 28]
}

grouped = df.groupby(['match_no', 'playing_side', 'point_no', 'stroke_num'])
for (match, side, pt, strk), group in grouped:
    if (match, side, pt, strk) in completed_strokes:
        continue
    stroke_type = group.iloc[0]['stroke_type']
    if stroke_type == 'FAULT':
        continue

    # 🟢 NEW: THE SMOOTHING FILTER 🟢
    # Apply a centered rolling average to smooth out MediaPipe micro-jitter 
    # before we do ANY velocity or angle math.
    group = group.copy() # Prevent SettingWithCopyWarning
    
    # Isolate only the coordinate columns (x, y, z, v) for the joints we care about
    coord_cols = [col for col in group.columns if col[0] in ['x', 'y', 'z', 'v'] and col[1:].isdigit()]
    
    # Apply a 3-frame rolling mean (Center=True keeps the peak of the swing aligned)
    group[coord_cols] = group[coord_cols].rolling(window=3, min_periods=1, center=True).mean()
    # 🟢 END SMOOTHING 🟢

    prev_coords = None
    prev_angles = None
    stroke_data = []
    for i in range(len(group)):
        row = group.iloc[i]
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
        torso_length = max(torso_length, 1e-5) # Prevent division by zero
        
        # --- APPLY NORMALIZATION TO ALL COORDS ---
        for joint in joints:
            for lr in ['left', 'right']:
                coords[f'{lr}_{joint}_x'] = (coords[f'{lr}_{joint}_x'] - mid_hip_x) / torso_length
                coords[f'{lr}_{joint}_y'] = (coords[f'{lr}_{joint}_y'] - mid_hip_y) / torso_length
                coords[f'{lr}_{joint}_z'] = (coords[f'{lr}_{joint}_z'] - mid_hip_z) / torso_length
        
        vectors['left_hip_shoulder'] = vector(coords['left_hip_x'], coords['left_hip_y'], coords['left_hip_z'], 
                                            coords['left_shoulder_x'], coords['left_shoulder_y'], coords['left_shoulder_z'])
        vectors['right_hip_shoulder'] = vector(coords['right_hip_x'], coords['right_hip_y'], coords['right_hip_z'], 
                                            coords['right_shoulder_x'], coords['right_shoulder_y'], coords['right_shoulder_z'])
        vectors['left_shoulder_elbow'] = vector(coords['left_shoulder_x'], coords['left_shoulder_y'], coords['left_shoulder_z'],
                                            coords['left_elbow_x'], coords['left_elbow_y'], coords['left_elbow_z'])
        vectors['right_shoulder_elbow'] = vector(coords['right_shoulder_x'], coords['right_shoulder_y'], coords['right_shoulder_z'],
                                            coords['right_elbow_x'], coords['right_elbow_y'], coords['right_elbow_z'])
        vectors['left_shoulder_hip'] = vectors['left_hip_shoulder'] * -1
        vectors['right_shoulder_hip'] = vectors['right_hip_shoulder'] * -1
        vectors['left_hip_knee'] = vector(coords['left_hip_x'], coords['left_hip_y'], coords['left_hip_z'], 
                                        coords['left_knee_x'], coords['left_knee_y'], coords['left_knee_z'])
        vectors['right_hip_knee'] = vector(coords['right_hip_x'], coords['right_hip_y'], coords['right_hip_z'], 
                                        coords['right_knee_x'], coords['right_knee_y'], coords['right_knee_z'])
        vectors['left_elbow_wrist'] = vector(coords['left_elbow_x'], coords['left_elbow_y'], coords['left_elbow_z'],
                                             coords['left_wrist_x'], coords['left_wrist_y'], coords['left_wrist_z'])
        vectors['right_elbow_wrist'] = vector(coords['right_elbow_x'], coords['right_elbow_y'], coords['right_elbow_z'],
                                             coords['right_wrist_x'], coords['right_wrist_y'], coords['right_wrist_z'])
        vectors['left_knee_ankle'] = vector(coords['left_knee_x'], coords['left_knee_y'], coords['left_knee_z'],
                                            coords['left_ankle_x'], coords['left_ankle_y'], coords['left_ankle_z'])
        vectors['right_knee_ankle'] = vector(coords['right_knee_x'], coords['right_knee_y'], coords['right_knee_z'],
                                            coords['right_ankle_x'], coords['right_ankle_y'], coords['right_ankle_z'])
        vectors['left_shoulder_wrist'] = vector(coords['left_shoulder_x'], coords['left_shoulder_y'], coords['left_shoulder_z'],
                                                coords['left_wrist_x'], coords['left_wrist_y'], coords['left_wrist_z'])
        vectors['right_shoulder_wrist'] = vector(coords['right_shoulder_x'], coords['right_shoulder_y'], coords['right_shoulder_z'],
                                                coords['right_wrist_x'], coords['right_wrist_y'], coords['right_wrist_z'])
        vectors['shoulders'] = vector(coords['left_shoulder_x'], coords['left_shoulder_y'], coords['left_shoulder_z'],
                                      coords['right_shoulder_x'], coords['right_shoulder_y'], coords['right_shoulder_z'])
        vectors['hips'] = vector(coords['left_hip_x'], coords['left_hip_y'], coords['left_hip_z'],
                                 coords['right_hip_x'], coords['right_hip_y'], coords['right_hip_z'])
        
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

            velocities['left_wrist_velocity'] = np.nan
            velocities['right_wrist_velocity'] = np.nan
            velocities['left_elbow_velocity'] = np.nan
            velocities['right_elbow_velocity'] = np.nan
            velocities['left_hip_velocity'] = np.nan
            velocities['right_hip_velocity'] = np.nan
            velocities['left_shoulder_velocity'] = np.nan
            velocities['right_shoulder_velocity'] = np.nan
            velocities['left_rel_wrist_velocity'] = np.nan
            velocities['right_rel_wrist_velocity'] = np.nan
            velocities['left_rel_elbow_velocity'] = np.nan
            velocities['right_rel_elbow_velocity'] = np.nan

        else:
            for deg in angles:
                angular_velocities[f'{deg}_vel'] = (angles[deg] - prev_angles[deg]) * 25
        
            velocities['left_wrist_velocity'] = dist(prev_coords['left_wrist_x'], prev_coords['left_wrist_y'], prev_coords['left_wrist_z'],
                                            coords['left_wrist_x'], coords['left_wrist_y'], coords['left_wrist_z']) * 25
            velocities['right_wrist_velocity'] = dist(prev_coords['right_wrist_x'], prev_coords['right_wrist_y'], prev_coords['right_wrist_z'],
                                            coords['right_wrist_x'], coords['right_wrist_y'], coords['right_wrist_z']) * 25
            velocities['left_elbow_velocity'] = dist(prev_coords['left_elbow_x'], prev_coords['left_elbow_y'], prev_coords['left_elbow_z'],
                                            coords['left_elbow_x'], coords['left_elbow_y'], coords['left_elbow_z']) * 25
            velocities['right_elbow_velocity'] = dist(prev_coords['right_elbow_x'], prev_coords['right_elbow_y'], prev_coords['right_elbow_z'],
                                            coords['right_elbow_x'], coords['right_elbow_y'], coords['right_elbow_z']) * 25
            velocities['left_hip_velocity'] = dist(prev_coords['left_hip_x'], prev_coords['left_hip_y'], prev_coords['left_hip_z'],
                                        coords['left_hip_x'], coords['left_hip_y'], coords['left_hip_z']) * 25
            velocities['right_hip_velocity'] = dist(prev_coords['right_hip_x'], prev_coords['right_hip_y'], prev_coords['right_hip_z'],
                                        coords['right_hip_x'], coords['right_hip_y'], coords['right_hip_z']) * 25
            velocities['left_shoulder_velocity'] = dist(prev_coords['left_shoulder_x'], prev_coords['left_shoulder_y'], prev_coords['left_shoulder_z'],
                                            coords['left_shoulder_x'], coords['left_shoulder_y'], coords['left_shoulder_z']) * 25
            velocities['right_shoulder_velocity'] = dist(prev_coords['right_shoulder_x'], prev_coords['right_shoulder_y'], prev_coords['right_shoulder_z'],
                                            coords['right_shoulder_x'], coords['right_shoulder_y'], coords['right_shoulder_z']) * 25
            velocities['left_rel_wrist_velocity'] = velocities['left_wrist_velocity'] - velocities['left_elbow_velocity']
            velocities['right_rel_wrist_velocity'] = velocities['right_wrist_velocity'] - velocities['right_elbow_velocity']
            velocities['left_rel_elbow_velocity'] = velocities['left_elbow_velocity'] - velocities['left_shoulder_velocity']
            velocities['right_rel_elbow_velocity'] = velocities['right_elbow_velocity'] - velocities['right_shoulder_velocity']
        
        dists['left_hip_wrist_dist'] = dist(coords['left_hip_x'], coords['left_hip_y'], coords['left_hip_z'],
                                        coords['left_wrist_x'], coords['left_wrist_y'], coords['left_wrist_z'])
        dists['right_hip_wrist_dist'] = dist(coords['right_hip_x'], coords['right_hip_y'], coords['right_hip_z'],
                                        coords['right_wrist_x'], coords['right_wrist_y'], coords['right_wrist_z'])
        dists['wrist_to_wrist_dist'] = dist(coords['left_wrist_x'], coords['left_wrist_y'], coords['left_wrist_z'],
                                       coords['right_wrist_x'], coords['right_wrist_y'], coords['right_wrist_z'])
        dists['ankle_to_ankle_dist'] = dist(coords['left_ankle_x'], coords['left_ankle_y'], coords['left_ankle_z'],
                                        coords['right_ankle_x'], coords['right_ankle_y'], coords['right_ankle_z'])
        dists['left_shoulder_wrist_dist'] = dist(coords['left_shoulder_x'], coords['left_shoulder_y'], coords['left_shoulder_z'],
                                            coords['left_wrist_x'], coords['left_wrist_y'], coords['left_wrist_z'])
        dists['right_shoulder_wrist_dist'] = dist(coords['right_shoulder_x'], coords['right_shoulder_y'], coords['right_shoulder_z'],
                                            coords['right_wrist_x'], coords['right_wrist_y'], coords['right_wrist_z'])
        dists['left_wrist_y_shoulder_y_dist'] = coords['left_wrist_y'] - coords['left_shoulder_y']
        dists['right_wrist_y_shoulder_y_dist'] = coords['right_wrist_y'] - coords['right_shoulder_y']

        prev_coords = coords.copy()
        prev_angles = angles.copy()

        frame_features = {
            'match_no': match, 'playing_side': side, 
            'point_no': pt, 'stroke_num': strk, 'frame_no': row['frame_no'], 'stroke_type': row['stroke_type'],
            'court_x': row['court_x'], 'court_y': row['court_y']
        }
        arm = handedness_map[(match, side)]
        other_arm = 'left' if arm == 'right' else 'right'

        def apply_symmetry(feature_dict):
            for key, value in feature_dict.items():
                if key.startswith(arm + '_'):
                    new_key = key.replace(arm + '_', 'racket_', 1)
                    frame_features[new_key] = value
                elif key.startswith(other_arm + '_'):
                    new_key = key.replace(other_arm + '_', 'balance_', 1)
                    frame_features[new_key] = value
                else:
                    frame_features[key] = value

        apply_symmetry(coords)
        apply_symmetry(angles)
        apply_symmetry(angular_velocities)
        apply_symmetry(velocities)
        apply_symmetry(dists)
        
        stroke_data.append(frame_features)
    
    stroke_df = pd.DataFrame(stroke_data)
    stroke_df['stroke_type'] = stroke_df['stroke_type'].replace('Flick-Serve', 'Serve')

    cols_to_exclude = ['match_no', 'playing_side', 'point_no', 'stroke_num', 'frame_no', 'stroke_type']
    feature_columns = [col for col in stroke_df.columns if col not in cols_to_exclude]
    stroke_df[feature_columns] = stroke_df[feature_columns].interpolate(method='linear', limit_direction='both')

    write_header = not os.path.exists(output_csv)
    stroke_df.to_csv(output_csv, mode='a', header=write_header, index=False)
    
    completed_strokes.add((match, side, pt, strk))
    

print("Math complete. Converting to DataFrame...")
final_df = pd.read_csv(output_csv)
cols_to_exclude = ['match_no', 'playing_side', 'point_no', 'stroke_num', 'frame_no', 'stroke_type']
feature_columns = [col for col in final_df.columns if col not in cols_to_exclude]

X_raw = final_df[feature_columns].to_numpy()

assert len(final_df) % 30 == 0, f"Dataset has {len(final_df)} rows — not divisible by 30. Some strokes are malformed."
num_strokes = len(final_df) // 30
num_features = len(feature_columns)
X_tensor = X_raw.reshape(num_strokes, 30, num_features)

y_strings = final_df.groupby(['match_no', 'playing_side', 'point_no', 'stroke_num'])['stroke_type'].first().values
encoder = LabelEncoder()
y_tensor = encoder.fit_transform(y_strings)

np.savez_compressed('4.2-features_tensor_smoothed.npz', X=X_tensor, y=y_tensor, feature_names=feature_columns, classes=encoder.classes_)
print(f"SUCCESS! Tensor shape: {X_tensor.shape}, y_tensor shape: {y_tensor.shape}")


        