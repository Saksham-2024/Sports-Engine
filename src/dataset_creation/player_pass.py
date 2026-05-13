import os
import sys
sys.path.append(os.path.abspath('..'))
import cv2
import re
import numpy as np
import pickle
import json
from ultralytics import YOLO
import yaml

with open('../../configs/configs.yaml', 'r') as f:
    configs = yaml.safe_load(f)

def pixel_to_court(x, y, H):
    return cv2.perspectiveTransform(np.array([[[x, y]]], dtype='float32'), H)[0][0]

def is_on_court(court_x, court_y, court_width=5.18, court_length=13.4):
    return 0 <= court_x <= court_width and 0 <= court_y <= court_length

CWD           = os.path.dirname(os.path.abspath(__file__))
VIDEO_DIR     = os.path.join(configs['global']['project_root'], configs['global']['video_dir'])
OUTPUT_PATH   = os.path.join(configs['global']['project_root'], configs['dataset_creation']['player_tracks_dir'])
SEGMENTS_DIR  = os.path.join(configs['global']['project_root'], configs['dataset_creation']['segments_dir'])
HOMOGRAPHY_CACHE = os.path.join(configs['global']['project_root'], configs['dataset_creation']['homography_cache'])

os.makedirs(OUTPUT_PATH, exist_ok=True)

# --- Load YOLO ---
yolo = YOLO(os.path.join(configs['global']['project_root'], configs['models']['yolo']))
print("✓ YOLO loaded.")

# --- Load Homography ---
if os.path.exists(HOMOGRAPHY_CACHE):
    with open(HOMOGRAPHY_CACHE, 'rb') as f:
        raw_matrices = pickle.load(f)
        matrices = {os.path.basename(k): v for k, v in raw_matrices.items()}
    print("✓ Homography matrices loaded.")
else:
    raise FileNotFoundError(f"Homography cache not found at {HOMOGRAPHY_CACHE}")


def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]


def process_video(video_path, H):
    """
    Process a full video frame-by-frame:
      - SACNN gates every frame as court_view or not.
      - YOLO + homography runs ONLY on court_view frames.
    Outputs a single CSV with per-frame data.
    """
    match_name = os.path.splitext(os.path.basename(video_path))[0]

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  ✗ Error opening {video_path}")
        return None

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    segment_file = os.path.join(SEGMENTS_DIR, f"{match_name}.json")
    segments = None
    if os.path.exists(segment_file):
        with open(segment_file, 'r') as f:
            segments = json.load(f).get('segments', [])

    print(f"\n{'='*60}")
    print(f"Processing: {match_name} ({total_frames} frames)")
    if segments:
        print(f"Using bounded segments: {len(segments)}")
    print(f"{'='*60}")

    rally_list = []
    processed_count = 0
    frame_idx = 0
    current_segment = None
    current_positions = []

    while cap.isOpened() and frame_idx < total_frames:
        if segments is not None:
            # Check bounding segments
            in_seg = False
            active_seg = None
            for seg in segments:
                if seg[0] <= frame_idx <= seg[1]:
                    in_seg = True
                    active_seg = seg
                    break
            
            if not in_seg:
                if current_segment is not None:
                    rally_list.append({
                        "segment": current_segment,
                        "positions": current_positions
                    })
                    current_segment = None
                    current_positions = []

                next_start = None
                for seg in segments:
                    if seg[0] > frame_idx:
                        next_start = seg[0]
                        break
                
                if next_start is not None:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, next_start)
                    frame_idx = next_start
                    continue
                else:
                    break # No more segments
            else:
                if current_segment != active_seg:
                    if current_segment is not None:
                        rally_list.append({
                            "segment": current_segment,
                            "positions": current_positions
                        })
                    current_segment = active_seg
                    current_positions = []

        ret, frame = cap.read()
        if not ret:
            break

        row = {
            'frame': frame_idx,
            'p1_cx': None, 'p1_cy': None,
            'p1x1': None, 'p1y1': None, 'p1x2': None, 'p1y2': None,
            'p2_cx': None, 'p2_cy': None,
            'p2x1': None, 'p2y1': None, 'p2x2': None, 'p2y2': None,
        }

        processed_count += 1

        # Step 2: YOLO player detection (only on segment frames)
        players_pos = []
        results = yolo.predict(frame, verbose=False)

        for result in results:
            if result.boxes is not None and len(result.boxes) > 0:
                for box in result.boxes:
                    if int(box.cls) == 0:  # class 0 = person
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        feet_center_x = int((x1 + x2) / 2)
                        feet_center_y = int(y2)
                        court_coords = pixel_to_court(feet_center_x, feet_center_y, H)
                        court_x, court_y = court_coords[0], court_coords[1]
                        if is_on_court(court_x, court_y):
                            # Convert types so json serialization succeeds
                            players_pos.append((float(court_x), float(court_y), float(x1), float(y1), float(x2), float(y2)))

        if len(players_pos) > 0:
            players_pos.sort(key=lambda p: p[1])  # sort by court_y: lower = far side

            # P1 = far side (lower court_y)
            row['p1_cx'], row['p1_cy'] = players_pos[0][0], players_pos[0][1]
            row['p1x1'], row['p1y1'] = players_pos[0][2], players_pos[0][3]
            row['p1x2'], row['p1y2'] = players_pos[0][4], players_pos[0][5]

            if len(players_pos) == 2:
                # P2 = near side (higher court_y)
                row['p2_cx'], row['p2_cy'] = players_pos[1][0], players_pos[1][1]
                row['p2x1'], row['p2y1'] = players_pos[1][2], players_pos[1][3]
                row['p2x2'], row['p2y2'] = players_pos[1][4], players_pos[1][5]

        if current_segment is not None:
            current_positions.append(row)
        else:
            # Fallback if segments json was missing (just append as a single block)
            current_positions.append(row)

        frame_idx += 1

        if processed_count % 500 == 0:
            print(f"  ... Processed {processed_count} frames | Current frame: {frame_idx}/{total_frames} ({(frame_idx/total_frames)*100:.1f}%)")

    cap.release()

    # Append the last segment if video ends while inside it
    if current_segment is not None:
        rally_list.append({
            "segment": current_segment,
            "positions": current_positions
        })
    elif not segments and current_positions:
        # Fallback if no segments file existed
        rally_list.append({
            "segment": [0, total_frames - 1],
            "positions": current_positions
        })

    save_path = os.path.join(OUTPUT_PATH, f'{match_name}_players.json')
    with open(save_path, 'w') as f:
        json.dump({"rally": rally_list}, f, indent=2)

    print(f"✓ {match_name}: {processed_count} frames processed inside segments")
    print(f"  Saved to: {save_path}")
    return save_path


def main():
    video_files = sorted(
        [f for f in os.listdir(VIDEO_DIR) if f.lower().endswith(('.mp4', '.mov', '.avi'))],
        key=natural_sort_key
    )

    print(f"\n Found {len(video_files)} videos to process\n")

    for video_file in video_files:
        video_path = os.path.join(VIDEO_DIR, video_file)

        H = matrices.get(video_file)
        if H is None:
            print(f"  [SKIP] No homography matrix for {video_file}")
            continue

        process_video(video_path, H)

    print("\n Third Pass Complete! Player positions logged.")


if __name__ == "__main__":
    main()