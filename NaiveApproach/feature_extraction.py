import numpy as np
import pandas as pd

df = pd.read_csv('player_positions.csv')
court_dimensions = [[0, 0], [5.18, 0], [5.18, 13.4], [0, 13.4]]
net_dimensions = [[0, 6.7], [5.18, 6.7]]

def euclid(x1, y1, x2, y2):
    # convert to scalars (we only use scalars here)
    try:
        x1f, y1f, x2f, y2f = float(x1), float(y1), float(x2), float(y2)
    except Exception:
        return np.nan
    if np.isnan(x1f) or np.isnan(y1f) or np.isnan(x2f) or np.isnan(y2f):
        return np.nan
    return float(np.hypot(x2f - x1f, y2f - y1f))

features = []
# structure as player1_pos, player2_pos, player1_x, player1_y, player2_x, player2_y,
# stroke_type, prev_stroke_type, player_hitting, 
# euclidean ditance between players, distance from net for player 1, distance from net for player 2
# distance (or displacement) between player 1 and center and player 2 and center
# current velocity of player 1 and player 2 (if previous frame available) as to know what was the 
# vector they were moving in before the stroke

for index, row in df.iterrows():
    p1x, p1y = round(row.get("player1_x", np.nan), 3), round(row.get("player1_y", np.nan), 3)
    p2x, p2y = round(row.get("player2_x", np.nan), 3), round(row.get("player2_y", np.nan), 3)
    
    vel_p1_dx = np.nan
    vel_p1_dy = np.nan
    vel_p2_dx = np.nan
    vel_p2_dy = np.nan

    dist_players = round(euclid(p1x, p1y, p2x, p2y), 3)
    dist_player1_net = round(abs(p1y - 6.7), 3) if np.isfinite(p1y) else np.nan
    dist_player2_net = round(abs(p2y - 6.7), 3) if np.isfinite(p2y) else np.nan

    center1 = (2.59, 3.35)  # center of player 1 side
    center2 = (2.59, 10.05) # center of player 2 side

    # displacements from center (without signs. set according to hitter later)
    disp_p1_center_dx = round(p1x - center1[0], 3)
    disp_p1_center_dy = round(p1y - center1[1], 3)
    disp_p2_center_dx = round(p2x - center2[0], 3)
    disp_p2_center_dy = round(p2y - center2[1], 3)
    dist_p1_center = round(np.hypot(disp_p1_center_dx, disp_p1_center_dy), 3)
    dist_p2_center = round(np.hypot(disp_p2_center_dx, disp_p2_center_dy), 3)
    prev_stroke = ""

    if index > 0 and (row.get('stroke_type') != "Serve"):
        prev = df.iloc[index - 1]
        dt = row.get("hit_time", np.nan) - prev.get("hit_time", np.nan)
        if np.isfinite(dt) and dt > 0:
            # player1: prev -> current
            prev_p1x, prev_p1y = prev.get("player1_x", np.nan), prev.get("player1_y", np.nan)
            if np.isfinite(p1x) and np.isfinite(prev_p1x):
                disp_dx = p1x - prev_p1x
                vel_p1_dx = disp_dx / dt
            
            if np.isfinite(p1y) and np.isfinite(prev_p1y):
                disp_dy = p1y - prev_p1y
                vel_p1_dy = np.nan if np.isnan(disp_dy) else disp_dy / dt

            # player2: prev -> current
            prev_p2x, prev_p2y = prev.get("player2_x", np.nan), prev.get("player2_y", np.nan)
            if np.isfinite(p2x) and np.isfinite(prev_p2x):
                disp_dx2 = p2x - prev_p2x
                vel_p2_dx = disp_dx2 / dt
                
            if np.isfinite(p2y) and np.isfinite(prev_p2y):
                disp_dy2 = p2y - prev_p2y
                vel_p2_dy = disp_dy2 / dt
            
            vel_p1_dx = round(vel_p1_dx, 3)
            vel_p1_dy = round(vel_p1_dy, 3)
            vel_p2_dx = round(vel_p2_dx, 3)
            vel_p2_dy = round(vel_p2_dy, 3)
            prev_stroke = prev.get("stroke_type", "None")
    
    elif row.get('stroke_type') == "Serve":
        vel_p1_dx = 0.0
        vel_p1_dy = 0.0
        vel_p2_dx = 0.0
        vel_p2_dy = 0.0
        prev_stroke = "None"
        

    hitter = row.get("player_hitting", row.get("hitting_player", None))

    if hitter == 2:
        if np.isfinite(vel_p1_dy): vel_p1_dy *= -1
        if np.isfinite(disp_p1_center_dy): disp_p1_center_dy *= -1
        if np.isfinite(disp_p2_center_dy): disp_p2_center_dy *= -1
    else:
        if np.isfinite(vel_p1_dx): vel_p1_dx *= -1
        if np.isfinite(vel_p2_dx): vel_p2_dx *= -1
        if np.isfinite(disp_p2_center_dx): disp_p2_center_dx *= -1
        if np.isfinite(disp_p1_center_dx): disp_p1_center_dx *= -1
    
    prev = df.iloc[index - 1] if index > 0 else None 
    next = df.iloc[index + 1] if index < len(df) - 1 and df.iloc[index + 1]["stroke_type"] != "Serve" else None 
    shuttle_from = ""
    if prev is not None:
        shuttle_from = prev["player1_pos"] if hitter == 2 else prev["player2_pos"]
    else: shuttle_from = "Unknown" 

    shuttle_to = ""
    if next is not None:
        shuttle_to = next.get("player1_pos") if next.get("player_hitting") == 1 else next.get("player2_pos")
    else:
        shuttle_to = "Unknown"

    features.append({
        "match_id": row.get("match"),
        "rally_id": row.get("rally_id"),
        "stroke_num": row.get("stroke_num"),
        "player_hitting": hitter,
        "player1_pos": row.get("player1_pos"),
        "player2_pos": row.get("player2_pos"),
        "player1_x": p1x,
        "player1_y": p1y,
        "player2_x": p2x,
        "player2_y": p2y,
        # distance between players
        "dist_players": dist_players,
        # distance from net
        "dist_player1_net": dist_player1_net,
        "dist_player2_net": dist_player2_net,
        # displacement from center as projections
        "disp_p1_center_dx": disp_p1_center_dx,
        "disp_p1_center_dy": disp_p1_center_dy,
        "disp_p2_center_dx": disp_p2_center_dx,
        "disp_p2_center_dy": disp_p2_center_dy,
        # distance from center
        "dist_p1_center": dist_p1_center,
        "dist_p2_center": dist_p2_center,
        # velocities as projections
        "vel_p1_dx": vel_p1_dx,
        "vel_p1_dy": vel_p1_dy,
        "vel_p2_dx": vel_p2_dx,
        "vel_p2_dy": vel_p2_dy,
        # shot info
        "shuttle_hit_from": shuttle_from,
        "shuttle_hit_to": shuttle_to,
        "stroke_type": row.get("stroke_type"),
        "prev_stroke_type": prev_stroke,
        "player1_pos_mask": 0 if row.get("player1_pos") == "Unknown" else 1,
        "player2_pos_mask": 0 if row.get("player2_pos") == "Unknown" else 1,
        "shuttle_hit_from_mask": 0 if shuttle_from == "Unknown" else 1,
        "shuttle_hit_to_mask": 0 if shuttle_to == "Unknown" else 1,
    })

cols = [
    'player1_x','player1_y',
    'player2_x','player2_y',
    'dist_players','dist_player1_net','dist_player2_net',
    'disp_p1_center_dx','disp_p1_center_dy',
    'disp_p2_center_dx','disp_p2_center_dy',
    'dist_p1_center','dist_p2_center',
    'vel_p1_dx','vel_p1_dy',
    'vel_p2_dx','vel_p2_dy'
]

features_df = pd.DataFrame(features)
for col in cols:
    features_df[col + "_mask"] = features_df[col].apply(lambda x: 0 if np.isnan(x) else 1)

features_df[cols] = features_df[cols].fillna(0)
features_df["prev_stroke_type_mask"] = 0 if features_df["prev_stroke_type"].isnull().all() else 1

features_df.to_csv('features.csv', index=False)