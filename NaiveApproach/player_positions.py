import cv2
import os
import numpy as np
import pandas as pd
try:
    import mediapipe as mp
except Exception:
    mp = None

image_path = "KeyFrames/match_1_point_Point1_stroke_1.jpg"
image = cv2.imread(image_path)
if image is None:
    raise FileNotFoundError(f"Could not read initial image for homography: {image_path}")

points = []

# Click the 4 corners of the court
# Order: top-left, top-right, bottom-right, bottom-left
def click_event(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))
        cv2.circle(image, (x, y), 5, (0, 0, 255), -1)
        cv2.imshow('Mark 4 Court Corners', image)

cv2.imshow('Mark 4 Court Corners', image)
cv2.setMouseCallback('Mark 4 Court Corners', click_event)
cv2.waitKey(0)
cv2.destroyAllWindows()

print("Selected corner points (camera view):", points)

if len(points) != 4:
    raise ValueError("You must click exactly 4 corners (top-left, top-right, bottom-right, bottom-left)")

# Define the real-world coordinates (in meters)
real_world_court = np.array([
    [0, 0],        # top-left
    [5.18, 0],     # top-right
    [5.18, 13.4],  # bottom-right
    [0, 13.4]      # bottom-left
], dtype=np.float32)

# Compute homography matrix
image_points = np.array(points, dtype=np.float32)
H, _ = cv2.findHomography(image_points, real_world_court)
print("\nHomography matrix (H):\n", H)

# Function to convert pixel → real-world coordinates
def pixel_to_court(x, y, H):
    point = np.array([[[x, y]]], dtype='float32')
    transformed = cv2.perspectiveTransform(point, H)
    return transformed[0][0]

from ultralytics import YOLO
model = YOLO("yolov8n.pt")

def detect_players(image, model, H):
    results = model(image, verbose = False)[0].boxes.data.cpu().numpy()
    players = []
    boxes = []
    for box in results:
        x1, y1, x2, y2, conf, cl = box
        if int(cl) == 0:
            cx, cy = (x1 + x2) // 2, y2
            court_x, court_y = pixel_to_court(cx, cy, H)
            if 0 <= court_x <= 5.18 and 0 <= court_y <= 13.4:
                cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
                players.append((cx, cy))
                boxes.append((int(x1), int(y1), int(x2), int(y2), conf, int(cl)))

    players.sort(key=lambda p: p[1])  
    return boxes, players, image

def pos_on_court_along_length(x, y, player):
    if player == 1:
        if 0 <= y <= 6.7 / 3:
            return "Back"
        elif 6.7 / 3 < y <= 2 * 6.7 / 3:
            return "Middle"
        elif 2 * 6.7 / 3 < y <= 6.7:
            return "Front"
    else:
        if 6.7 < y <= 13.4 - 2 * 6.7 / 3:
            return "Front"
        elif 13.4 - 2 * 6.7 / 3 < y <= 13.4 - 6.7 / 3:
            return "Middle"
        elif 13.4 - 6.7 / 3 < y <= 13.4:
            return "Back"

def pos_on_court_along_width(x, y, player):
    if player == 2:
        if 0 <= x <= 5.18 / 3:
            return "Left"
        elif 5.18 / 3 < x <= 2 * 5.18 / 3:
            return "Center"
        else:
            return "Right"
    else:
        if 0 <= x <= 5.18 / 3:
            return "Right"
        elif 5.18 / 3 < x <= 2 * 5.18 / 3:
            return "Center"
        else:
            return "Left"

def pos_on_court(x, y, player):
    width_pos = pos_on_court_along_width(x, y, player)
    length_pos = pos_on_court_along_length(x, y, player)
    if width_pos is None or length_pos is None:
        return width_pos or length_pos or "Unknown"
    else: return width_pos + '-' + length_pos

skeleton_dir = "Skeleton_marked_frames"
os.makedirs(skeleton_dir, exist_ok=True)
records = []
df = pd.read_csv("keyframes_metadata.csv")
mp_pose = mp.solutions.pose if mp is not None else None
pose_model = None 
if mp_pose is not None:
    pose_model = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.3, model_complexity=2)

for i in range(len(df)):
    image_path = df.loc[i, "image_path"]
    image = cv2.imread(image_path)
    if image is None:
        print(f"Could not read image: {image_path}")
        continue

    boxes, players, annotated = detect_players(image, model, H)
    print(f"\nFrame {i} - Detected player positions (court view in meters):", players)
    
    p1 = players[0] if len(players) > 0 else (None, None)
    p2 = players[1] if len(players) > 1 else (None, None)
    x1, y1 = p1
    x2, y2 = p2
    court_x1, court_y1 = pixel_to_court(x1, y1, H) if x1 is not None else (None, None)
    court_x2, court_y2 = pixel_to_court(x2, y2, H) if x2 is not None else (None, None)
    pos1 = pos_on_court(court_x1, court_y1, 1) if court_x1 is not None else "Unknown"
    player_hitting = df.loc[i, "hitting_player"]
    if player_hitting == 1:
        pos2 = pos_on_court_along_width(court_x2, court_y2, 2) if court_x2 is not None else "Unknown"
        if pos2 != "Unknown":
            if pos2 == "Left":
                pos2 = "Right"
            elif pos2 == "Right":
                pos2 = "Left"
            
            length = pos_on_court_along_length(court_x2, court_y2, 2)
            if length != None and length != "Unknown":
                pos2 += '-' + length

    else:
        pos2 = pos_on_court(court_x2, court_y2, 2) if court_x2 is not None else "Unknown"

    basename = os.path.basename(df.loc[i, "image_path"])
    output_path = os.path.join(skeleton_dir, f"annotated_{basename}.jpg")
    prev_stroke_type = ""
    if i == 0:
        prev_stroke_type = "None"
    else:
        prev_row = df.iloc[i - 1]
        curr_row = df.iloc[i]
        # ensure both rows belong to the same rally; else treat as None
        if 'rally_id' in df.columns and prev_row.get('rally_id') == curr_row.get('rally_id'):
            prev_stroke_type = prev_row.get('stroke_type', "None") or "None"
        else:
            prev_stroke_type = "None"

    records.append({
        "match": df.loc[i, "match_no"],
        "rally_id": df.loc[i, "rally_id"],
        "stroke_num": df.loc[i, "stroke_num"],
        "stroke_type": df.loc[i, "stroke_type"],
        "hit_time": df.loc[i, "hit_time"],
        "prev_stroke_type": prev_stroke_type,
        "player_hitting": df.loc[i, "hitting_player"],
        "player1_x": court_x1,
        "player1_y": court_y1,
        "player1_pos": pos1,
        "player2_x": court_x2,
        "player2_y": court_y2,
        "player2_pos": pos2,
        "image_path": output_path
    })
    
    # Draw skeletons to the images which currently has bounding boxes. variable 'annotated' has that image.
    annotated_skeleton = annotated.copy()
    if pose_model is not None and len(boxes) > 0:
        h, w = annotated.shape[:2]
        for (bx1, by1, bx2, by2, conf, cl) in boxes:
            bx1c, by1c = max(0, int(bx1)), max(0, int(by1))
            bx2c, by2c = min(w - 1, int(bx2)), min(h - 1, int(by2))
            if bx2c - bx1c < 10 or by2c - by1c < 10:
                continue
            crop = annotated[by1c:by2c, bx1c:bx2c].copy()
            crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            res = pose_model.process(crop_rgb)
            if not res.pose_landmarks:
                continue
            # draw landmarks mapped back to full image coords
            for lm in res.pose_landmarks.landmark:
                cx = int(lm.x * (bx2c - bx1c)) + bx1c
                cy = int(lm.y * (by2c - by1c)) + by1c
                cv2.circle(annotated_skeleton, (cx, cy), 3, (0, 255, 0), -1)
            # draw connections inside the crop region (in-place)
            mp.solutions.drawing_utils.draw_landmarks(
                annotated_skeleton[by1c:by2c, bx1c:bx2c],
                res.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=2),
                connection_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(color=(0,128,255), thickness=2)
            )
        cv2.imwrite(output_path, annotated_skeleton)
    else:
        print("mediapipe not available or no detected boxes - skipping skeletons.")
        cv2.imwrite(output_path, annotated)

    print(f"Annotated frame saved to: {output_path}")

csv_path = os.path.join("player_positions.csv")
pd.DataFrame(records).to_csv(csv_path, index=False)
