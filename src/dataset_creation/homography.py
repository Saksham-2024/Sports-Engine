import cv2
import numpy as np
import pickle
import os
import json
import yaml

with open('../../configs/configs.yaml', 'r') as f:
    configs = yaml.safe_load(f)

CWD           = os.path.dirname(os.path.abspath(__file__))
VIDEO_DIR     = os.path.join(configs['global']['project_root'], configs['global']['video_dir'])
SEGMENTS_PATH = os.path.join(configs['global']['project_root'], configs['dataset_creation']['segments_dir'])
OUTPUT_DIR    = os.path.join(configs['global']['project_root'], configs['global']['output_dir'])

os.makedirs(SEGMENTS_PATH, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
videos = [f for f in os.listdir(VIDEO_DIR) if f.lower().endswith(('.mp4', '.avi', '.mov'))]

# ── Real world reference points ───────────────────────────────────────────────
_court = configs['dataset_creation']['court']
court_width = _court['width']
court_length = _court['length']
net_height = _court['net_height']
# 3D court coords for solvePnP (metres, z=0 is court surface)
# Click order: TL corner, TR corner, BR corner, BL corner, Left net post, Right net post
COURT_3D_POINTS = np.array([
    [0,    0,    0   ],   # top-left corner
    [court_width,    0,    0   ],   # top-right corner
    [court_width, court_length,    0   ],   # bottom-right corner
    [0,    court_length, 0   ],   # bottom-left corner
    [0,    court_length/2,  net_height],   # left net post
    [court_width, court_length/2,  net_height],   # right net post
], dtype=np.float32)

# 2D real-world coords for homography (first 4 points only, z=0 plane)
COURT_2D_REAL = np.array([
    [0,    0   ],
    [court_width, 0   ],
    [court_width, court_length],
    [0,    court_length],
], dtype=np.float32)


def approximate_camera_matrix(frame_width, frame_height):
    """Reasonable intrinsic approximation for broadcast cameras."""
    f  = float(frame_width)          # focal length ≈ image width in pixels
    cx = frame_width  / 2.0
    cy = frame_height / 2.0
    return np.array([[f, 0, cx],
                     [0, f, cy],
                     [0, 0,  1]], dtype=np.float64)


def calibrate(time, video_path):
    """
    Opens one frame and asks the user to click 6 points:
      Points 1-4 : court corners (TL, TR, BR, BL)  — used for homography
      Points 5-6 : net posts (left, right)           — used for solvePnP only

    Returns
    -------
    success      : bool
    H            : (3,3) homography matrix  — drop-in replacement, unchanged API
    camera_pos   : (3,) camera world position in metres  (None if solvePnP fails)
    K            : (3,3) intrinsic matrix
    R            : (3,3) rotation matrix
    tvec         : (3,1) translation vector
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening video file: {video_path}")
        return False, np.zeros((3, 3), dtype=np.float64), None, None, None, None

    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, time)
        ret, frame = cap.read()
        cap.release()
        if not ret:
            print(f"Error reading frame {time} from video {video_path}")
            return False, np.zeros((3, 3), dtype=np.float64), None, None, None, None

        h, w    = frame.shape[:2]
        K       = approximate_camera_matrix(w, h)
        dist    = np.zeros((4, 1), dtype=np.float64)
        points  = []
        display = frame.copy()

        labels = [
            '1: TL corner', '2: TR corner', '3: BR corner', '4: BL corner',
            '5: Left net post top', '6: Right net post top'
        ]

        WIN_NAME = 'Calibrate - click 6 points'

        def click_event(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN and len(points) < 6:
                points.append((x, y))
                cv2.circle(display, (x, y), 6, (0, 0, 255), -1)
                cv2.putText(display, labels[len(points) - 1], (x + 8, y - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1)
                cv2.imshow(WIN_NAME, display)
                print(f"  [{len(points)}/6] {labels[len(points)-1]} at ({x}, {y})")

        print("\nClick in order: TL, TR, BR, BL court corners, then Left + Right net posts.")
        print("Keys: Enter=confirm | S=skip frame | R=undo last click | ESC=skip video")
        cv2.putText(display, "Enter=OK | S=skip | R=undo | ESC=skip video",
                    (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        cv2.namedWindow(WIN_NAME, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(WIN_NAME, click_event)
        cv2.imshow(WIN_NAME, display)

        skip_frame = False
        while True:
            key = cv2.waitKey(50) & 0xFF
            if key == 27:  # ESC → skip entire video
                points.clear()
                break
            if key in (ord('s'), ord('S')):  # S → skip this frame, try next
                points.clear()
                skip_frame = True
                break
            if key in (ord('r'), ord('R')) and points:  # R → undo last click
                points.pop()
                display[:] = frame  # redraw clean frame
                cv2.putText(display, "Enter=OK | S=skip | R=undo | ESC=skip video",
                            (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                for i, pt in enumerate(points):
                    cv2.circle(display, pt, 6, (0, 0, 255), -1)
                    cv2.putText(display, labels[i], (pt[0]+8, pt[1]-8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1)
                cv2.imshow(WIN_NAME, display)
                print(f"  Undo → {len(points)} points remaining")
            if key in (13, 32) and len(points) == 6:  # Enter or Space
                break
        cv2.destroyAllWindows()

        if skip_frame:
            return False, np.zeros((3, 3), dtype=np.float64), None, None, None, None

        print(f"Clicked {len(points)} points: {points}")

        if len(points) != 6:
            print(f"Expected 6 points, got {len(points)}. Please try again.")
            return False, np.zeros((3, 3), dtype=np.float64), None, None, None, None

        # ── Homography from first 4 corners  ──
        corner_px = np.array(points[:4], dtype=np.float64)
        H, _      = cv2.findHomography(corner_px, COURT_2D_REAL)
        print("\nHomography matrix (H):\n", H)

        # ── solvePnP from all 6 points ──
        all_px = np.array(points, dtype=np.float32)
        ok, rvec, tvec = cv2.solvePnP(
            COURT_3D_POINTS, all_px, K, dist,
            flags=cv2.SOLVEPNP_ITERATIVE
        )

        if not ok:
            print("solvePnP failed — homography saved, camera pose unavailable.")
            return True, H, None, K, None, None

        R, _       = cv2.Rodrigues(rvec)
        camera_pos = (-R.T @ tvec).flatten()
        print(f"\nCamera position in court coords (metres):")
        print(f"  x={camera_pos[0]:.3f}  y={camera_pos[1]:.3f}  z={camera_pos[2]:.3f}")

        return True, H, camera_pos, K, R, tvec

    except Exception as e:
        print(f"Error during calibration for {video_path}: {e}")
        return False, np.zeros((3, 3), dtype=np.float64), None, None, None, None


def _cached_basenames(d):
    """Return set of basenames from a dict whose keys may be relative or absolute paths."""
    return {os.path.basename(k) for k in d.keys()}


def gather_matrices(existing_matrices, existing_camera_poses):
    """
    Runs calibrate() only for videos NOT already in the caches.

    Parameters
    ----------
    existing_matrices     : {video_path: H}          — previously cached homographies
    existing_camera_poses : {video_path: dict}        — previously cached camera poses

    Returns
    -------
    matrices      : {video_path: H}          — merged (old + new)
    camera_poses  : {video_path: dict}       — merged (old + new)
    cache_updated : bool                     — True if any new entries were added
    """
    matrices     = dict(existing_matrices)
    camera_poses = dict(existing_camera_poses)
    cache_updated = False

    already_done = _cached_basenames(matrices)

    for video in sorted(videos):
        video_path = os.path.join(VIDEO_DIR, video)

        if video in already_done:
            print(f"⏩ Skipping {video} — already calibrated.")
            continue

        # Use segments JSON for a good calibration frame (first gameplay frame)
        seg_json = os.path.join(SEGMENTS_PATH, video.split('.')[0] + '.json')
        start_frame = 7500   # reasonable default for broadcast footage
        if os.path.exists(seg_json):
            try:
                with open(seg_json, 'r') as f:
                    segments = json.load(f).get('segments', [])
                if segments and len(segments[0]) >= 1:
                    start_frame = segments[0][0]  # first segment start frame
            except Exception as e:
                print(f"  Warning: could not read segments JSON for {video}: {e}")

        attempts = 30
        while attempts > 0:
            print(f"\nCalibrating {video} at frame {start_frame} ...")
            success, H, camera_pos, K, R, tvec = calibrate(start_frame, video_path)

            if success:
                matrices[video] = H       # basename key for portability
                cache_updated = True
                if camera_pos is not None:
                    camera_poses[video] = {   # basename key for portability
                        'camera_pos': camera_pos,
                        'K':    K,
                        'R':    R,
                        'tvec': tvec,
                    }
                    print(f"✅ Camera pose saved for {video}.")
                else:
                    print(f"⚠️  Homography saved but camera pose failed for {video}.")
                break
            else:
                print(f"Calibration failed for {video}. Trying next frame...")
                attempts    -= 1
                start_frame += 250

    return matrices, camera_poses, cache_updated


# ── Entry point ───────────────────────────────────────────────────────────────

homography_cache_file   = os.path.join(configs['global']['project_root'], configs['dataset_creation']['homography_cache'])
camera_pose_cache_file  = os.path.join(configs['global']['project_root'], configs['dataset_creation']['camera_pose_cache'])

# ── Load existing caches (cache-aware) ────────────────────────────────────────
matrices = {}
camera_poses = {}

if os.path.exists(homography_cache_file):
    with open(homography_cache_file, 'rb') as f:
        matrices = pickle.load(f)
    print(f" Loaded {len(matrices)} homography matrices from cache.")
else:
    print(" No homography cache found. Starting fresh.")

if os.path.exists(camera_pose_cache_file):
    with open(camera_pose_cache_file, 'rb') as f:
        camera_poses = pickle.load(f)
    print(f" Loaded {len(camera_poses)} camera poses from cache.")
else:
    print(" No camera pose cache found. Starting fresh.")

# ── Report status ─────────────────────────────────────────────────────────────
cached_videos = _cached_basenames(matrices)
all_videos = set(videos)
uncached = sorted(all_videos - cached_videos)

print(f"\n{'='*60}")
print(f"  Videos on disk:          {len(all_videos)}")
print(f"  Already calibrated:      {len(cached_videos)}")
print(f"  Remaining to calibrate:  {len(uncached)}")
if uncached:
    print(f"  Uncached: {', '.join(uncached)}")
print(f"{'='*60}\n")

if not uncached:
    print(" All videos are already calibrated. Nothing to do.")
else:
    matrices, camera_poses, cache_updated = gather_matrices(matrices, camera_poses)

    if cache_updated:
        with open(homography_cache_file, 'wb') as f:
            pickle.dump(matrices, f)
        print(f"\n Homography matrices saved → {homography_cache_file}")
        print(f"   Total entries: {len(matrices)}")

        with open(camera_pose_cache_file, 'wb') as f:
            pickle.dump(camera_poses, f)
        print(f" Camera poses saved       → {camera_pose_cache_file}")
        print(f"   Total entries: {len(camera_poses)}")
    else:
        print("\n  No new calibrations were completed.")