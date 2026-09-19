import cv2
import numpy as np
import pickle
import os
import json
from src.core.utils.configs import PathResolver
from src.core.utils.logger import get_logger
from abc import abstractmethod, ABC


class CourtPoints(ABC):
    def __init__(self, court):
        self.court = court
        self.court_width = self.court['width']
        self.court_length = self.court['length']
        self.net_height = self.court['net_height']
        self.COURT_3D_POINTS = np.array([
            [0, 0, 0],   # top-left corner
            [self.court_width, 0, 0],   # top-right corner
            [self.court_width, self.court_length, 0],   # bottom-right corner
            [0, self.court_length, 0],   # bottom-left corner
            [0, self.court_length/2, self.net_height],   # left net post
            [self.court_width, self.court_length/2, self.net_height],   # right net post
        ], dtype=np.float32)

        # 2D real-world coords for homography (first 4 points only, z=0 plane)
        self.COURT_2D_REAL = np.array([
            [0, 0],
            [self.court_width, 0],
            [self.court_width, self.court_length],
            [0, self.court_length],
        ], dtype=np.float32)

        self.points = []

    
    def get_suitable_display(self, segments_path, video, video_path, matrices, camera_poses):
        """
        Attempts calibration for a single video, retrying on different frames.
        Updates matrices / camera_poses in-place and returns cache_updated flag.
        """
        seg_json = os.path.join(segments_path, video.split('.')[0] + '.json')
        start_frame = 7500   # reasonable default for broadcast footage
        if os.path.exists(seg_json):
            try:
                with open(seg_json, 'r') as f:
                    segments = json.load(f).get('segments', [])
                if segments and len(segments[0]) >= 1:
                    start_frame = segments[0][0]  # first segment start frame
            except Exception as e:
                print(f"  Warning: could not read segments JSON for {video}: {e}")

        cache_updated = False
        attempts = 30
        while attempts > 0:
            print(f"\nCalibrating {video} at frame {start_frame} ...")
            self.points = []   # reset click buffer before each attempt
            values = self._click_points(start_frame, video_path)
            if not values:
                print(f"Calibration failed for {video}. Trying next frame...")
                attempts    -= 1
                start_frame += 250
                continue

            if values[-1] == "camera_pos":
                K, R, tvec, camera_pos = values[:-1]
                camera_poses[video] = {
                    'camera_pos': camera_pos,
                    'K':    K,
                    'R':    R,
                    'tvec': tvec,
                }
                cache_updated = True
                print(f"✅ Camera pose saved for {video}.")
            else:
                H = values[0]
                matrices[video] = H
                cache_updated = True
                print(f"✅ Homography saved for {video}.")

            break   # success — stop retrying

        return matrices, camera_poses, cache_updated
          
    def _click_points(self, start_frame, video_path):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error opening video file: {video_path}")
            return []
        try:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            ret, frame = cap.read()
            cap.release()
            if not ret:
                print(f"Error reading frame {start_frame} from video {video_path}")
                return []
            
            display = frame.copy()
            labels = [
                '1: TL corner', '2: TR corner', '3: BR corner', '4: BL corner',
                '5: Left net post top', '6: Right net post top'
            ]

            WIN_NAME = 'Calibrate - click 6 points'
            def click_event(event, x, y, flags, param):
                if event == cv2.EVENT_LBUTTONDOWN and len(self.points) < 6:
                    self.points.append((x, y))
                    cv2.circle(display, (x, y), 6, (0, 0, 255), -1)
                    cv2.putText(display, labels[len(self.points) - 1], (x + 8, y - 8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1)
                    cv2.imshow(WIN_NAME, display)
                    print(f"  [{len(self.points)}/6] {labels[len(self.points)-1]} at ({x}, {y})")
            
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
                    self.points.clear()
                    break
                if key in (ord('s'), ord('S')):  # S → skip this frame, try next
                    self.points.clear()
                    skip_frame = True
                    break
                if key in (ord('r'), ord('R')) and self.points:  # R → undo last click
                    self.points.pop()
                    display[:] = frame  # redraw clean frame
                    cv2.putText(display, "Enter=OK | S=skip | R=undo | ESC=skip video",
                                (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                    for i, pt in enumerate(self.points):
                        cv2.circle(display, pt, 6, (0, 0, 255), -1)
                        cv2.putText(display, labels[i], (pt[0]+8, pt[1]-8),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1)
                    cv2.imshow(WIN_NAME, display)
                    print(f"  Undo → {len(self.points)} points remaining")
                if key in (13, 32) and len(self.points) == 6:  # Enter or Space
                    break
            cv2.destroyAllWindows()

            if skip_frame:
                return []
            
            print(f"Clicked {len(self.points)} points: {self.points}")

            if len(self.points) != 6:
                print(f"Expected 6 points, got {len(self.points)}. Please try again.")
                return []
            
            values = self.calculate(self.points)
            return values
            
        except Exception as e:
            print(f"Error during calibration for {video_path}: {e}")
            return []

    @abstractmethod
    def calculate(self, points):
        """Compute camera matrices from the 6 clicked pixel points."""
        pass

class HomographyCalculator(CourtPoints):
    def __init__(self, court_dims, segments_path, video_name):
        super().__init__(court_dims)
        self.segments_path = segments_path
        self.video_name = video_name
        
    def calculate(self, points):
        corner_px = np.array(points[:4], dtype=np.float64)
        H, _      = cv2.findHomography(corner_px, self.COURT_2D_REAL)
        return [H, "homography"]

class CameraPositionCalculator(CourtPoints):
    def __init__(self, court_dims, frame_width, frame_height, segments_path, video_name):
        super().__init__(court_dims)
        self.segments_path = segments_path
        self.video_name = video_name
        self.logger = get_logger(__name__)
        self.f = float(frame_width)          # focal length ≈ image width in pixels
        self.cx = frame_width / 2.0
        self.cy = frame_height / 2.0
        self.camera_intrinsic_matrice = np.array([[self.f, 0, self.cx],
                                                  [0, self.f, self.cy],
                                                  [0, 0,  1]], dtype=np.float64)
        self.dist = np.zeros((4, 1), dtype=np.float64)

    def calculate(self, points):
        all_px = np.array(points, dtype=np.float32)
        ok, rvec, tvec = cv2.solvePnP(
            self.COURT_3D_POINTS, all_px, self.camera_intrinsic_matrice, self.dist, flags=cv2.SOLVEPNP_ITERATIVE
        )
        if not ok:
            self.logger.info(f"Camera position failed for {self.video_name}.")
            return []
        R, _  = cv2.Rodrigues(rvec)
        camera_pos = (-R.T @ tvec).flatten()
        print(f"\nCamera position in court coords (metres):")
        print(f"  x={camera_pos[0]:.3f}  y={camera_pos[1]:.3f}  z={camera_pos[2]:.3f}")
        return [self.camera_intrinsic_matrice, R, tvec, camera_pos, "camera_pos"]


class CalibrationPipeline:
    """Orchestrates homography and camera-pose calibration for all videos."""

    def __init__(self):
        self.resolver      = PathResolver()
        self.logger        = get_logger(__name__)
        self.video_dir        = self.resolver.get_path("global.video_dir")
        self.homography_cache = self.resolver.get_path("dataset_creation.homography_cache")
        self.camera_pos_cache = self.resolver.get_path("dataset_creation.camera_pose_cache")
        self.segments_path    = self.resolver.get_path("dataset_creation.segments_dir")
        self.court_dims       = self.resolver.get_config("dataset_creation.court")

    @staticmethod
    def _cached_basenames(d):
        """Return set of basenames from a dict whose keys may be relative or absolute paths."""
        return {os.path.basename(k) for k in d.keys()}

    def _load_caches(self):
        """Load homography and camera-pose caches from disk (or start fresh)."""
        matrices     = {}
        camera_poses = {}

        if os.path.exists(self.homography_cache):
            with open(self.homography_cache, 'rb') as f:
                matrices = pickle.load(f)
            self.logger.info(f" Loaded {len(matrices)} homography matrices from cache.")
        else:
            self.logger.info(" No homography cache found. Starting fresh.")

        if os.path.exists(self.camera_pos_cache):
            with open(self.camera_pos_cache, 'rb') as f:
                camera_poses = pickle.load(f)
            self.logger.info(f" Loaded {len(camera_poses)} camera poses from cache.")
        else:
            self.logger.info(" No camera pose cache found. Starting fresh.")

        return matrices, camera_poses

    def _save_caches(self, matrices, camera_poses):
        """Persist updated caches to disk."""
        with open(self.homography_cache, 'wb') as f:
            pickle.dump(matrices, f)
        self.logger.info(f"Homography matrices saved → {self.homography_cache} ({len(matrices)} entries)")

        with open(self.camera_pos_cache, 'wb') as f:
            pickle.dump(camera_poses, f)
        self.logger.info(f"Camera poses saved → {self.camera_pos_cache} ({len(camera_poses)} entries)")

    def _get_remaining_videos(self, videos, matrices, camera_poses):
        """Return dict of {video: [tasks]} for videos not yet fully calibrated."""
        homography_cached  = self._cached_basenames(matrices)
        camera_pose_cached = self._cached_basenames(camera_poses)

        remaining = {}
        for vid in videos:
            needs_h  = vid not in homography_cached
            needs_cp = vid not in camera_pose_cached
            if needs_h and needs_cp:
                remaining[vid] = ["homography", "camera_pose"]
            elif needs_h:
                remaining[vid] = ["homography"]
            elif needs_cp:
                remaining[vid] = ["camera_pose"]
        return remaining

    def _run(self):
        videos = sorted(
            f for f in os.listdir(self.video_dir)
            if f.lower().endswith(('.mp4', '.avi', '.mov'))
        )

        matrices, camera_poses = self._load_caches()
        remaining_videos = self._get_remaining_videos(videos, matrices, camera_poses)

        if not remaining_videos:
            self.logger.info("All videos are already calibrated. Nothing to do.")
            return

        cache_updated = False
        for vid, process in remaining_videos.items():
            video_path = os.path.join(self.video_dir, vid)

            # Peek at frame dimensions (needed for CameraPositionCalculator)
            _cap = cv2.VideoCapture(video_path)
            frame_w = int(_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_h = int(_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            _cap.release()

            if "homography" in process:
                h_calc = HomographyCalculator(self.court_dims, self.segments_path, vid)
                matrices, camera_poses, updated = h_calc.get_suitable_display(
                    self.segments_path, vid, video_path, matrices, camera_poses
                )
                cache_updated = cache_updated or updated

            if "camera_pose" in process:
                c_calc = CameraPositionCalculator(
                    self.court_dims, frame_w, frame_h, self.segments_path, vid
                )
                matrices, camera_poses, updated = c_calc.get_suitable_display(
                    self.segments_path, vid, video_path, matrices, camera_poses
                )
                cache_updated = cache_updated or updated

        if cache_updated:
            self._save_caches(matrices, camera_poses)
        else:
            self.logger.info("No new calibrations were completed.")

