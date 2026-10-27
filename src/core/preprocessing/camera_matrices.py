import cv2
import numpy as np
import pickle
import os
import json
from typing import List, Optional, Tuple, Dict, Any
from src.core.utils.configs import PathResolver
from src.core.utils.logger import get_logger


# ── 1. CourtPoints (ABC) ─────────────────────────────────────────────────────
# Sole responsibility: hold the fixed 3D / 2D court coordinate arrays.

class CourtPoints:
    """Holds the canonical 3D world-space and 2D real-world court coordinates."""

    def __init__(self, court: Dict[str, float]):
        self.court_width = court['width']
        self.court_length = court['length']
        self.net_height = court['net_height']

        # 3D real-world coords for solving PnP
        self.COURT_3D_POINTS = np.array([
            [0, 0, 0],   # TL corner
            [self.court_width, 0, 0],   # TR corner
            [self.court_width, self.court_length, 0],   # BR corner
            [0, self.court_length, 0],   # BL corner
            [0, self.court_length / 2, self.net_height],  # left net post
            [self.court_width, self.court_length / 2, self.net_height],  # right net post
        ], dtype=np.float32)

        # 2D real-world coords for homography (z = 0 plane, first 4 corners only)
        self.COURT_2D_POINTS = np.array([
            [0, 0],
            [self.court_width, 0],
            [self.court_width, self.court_length],
            [0, self.court_length],
        ], dtype=np.float32)


# ── 2. CalibrationUI ─────────────────────────────────────────────────────────
# Sole responsibility: display a video frame and collect 6 clicked pixel points.

class CalibrationUI:
    """
    Opens an OpenCV window for a given frame and returns 6 clicked pixel
    coordinates. Returns None if the user skips or aborts.
    """

    LABELS = [
        '1: TL corner', '2: TR corner', '3: BR corner', '4: BL corner',
        '5: Left net post top', '6: Right net post top'
    ]
    WIN_NAME = 'Calibrate – click 6 points'

    def __init__(self):
        self.logger = get_logger(__name__)

    def collect_points(self, frame: np.ndarray) -> Optional[List[Tuple[int, int]]]:
        """
        Display *frame* and wait for the user to click 6 court points.

        Returns:
            List of 6 (x, y) tuples on success.
            None if the user skips (S) or aborts (ESC).
        """
        points: List[Tuple[int, int]] = []
        display = frame.copy()

        def click_event(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN and len(points) < 6:
                points.append((x, y))
                cv2.circle(display, (x, y), 6, (0, 0, 255), -1)
                cv2.putText(display, self.LABELS[len(points) - 1], (x + 8, y - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1)
                cv2.imshow(self.WIN_NAME, display)
                self.logger.info(f"  [{len(points)}/6] {self.LABELS[len(points)-1]} at ({x}, {y})")

        self.logger.info("Click in order: TL, TR, BR, BL court corners, then Left + Right net posts.")
        self.logger.info("Keys: Enter/Space = confirm | S = skip frame | R = undo last click | ESC = abort video")

        cv2.putText(display, "Enter=OK | S=skip | R=undo | ESC=abort",
                    (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        cv2.namedWindow(self.WIN_NAME, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.WIN_NAME, click_event)
        cv2.imshow(self.WIN_NAME, display)

        skip_frame = False
        aborted    = False
        while True:
            key = cv2.waitKey(50) & 0xFF
            if key == 27:                                    # ESC → abort entire video
                aborted = True
                break
            if key in (ord('s'), ord('S')):                  # S → skip this frame
                skip_frame = True
                break
            if key in (ord('r'), ord('R')) and points:       # R → undo last click
                points.pop()
                display[:] = frame
                cv2.putText(display, "Enter=OK | S=skip | R=undo | ESC=abort",
                            (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                for i, pt in enumerate(points):
                    cv2.circle(display, pt, 6, (0, 0, 255), -1)
                    cv2.putText(display, self.LABELS[i], (pt[0] + 8, pt[1] - 8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1)
                cv2.imshow(self.WIN_NAME, display)
                self.logger.info(f"  Undo → {len(points)} points remaining")
            if key in (13, 32) and len(points) == 6:         # Enter or Space → confirm
                break

        cv2.destroyAllWindows()

        if aborted or skip_frame:
            return None

        if len(points) != 6:
            self.logger.warning(f"Expected 6 points, got {len(points)}. Discarding.")
            return None

        self.logger.info(f"Collected points: {points}")
        return points


# ── 3. Pure Calculators ───────────────────────────────────────────────────────
# Sole responsibility: take List[Tuple] of pixel points → return matrix(es).

class HomographyCalculator(CourtPoints):
    """Pure math: 4 pixel corners → 3×3 homography matrix H."""

    def __init__(self, court_dims: Dict[str, float]):
        super().__init__(court_dims)
        self.logger = get_logger(__name__)

    def calculate(self, points: List[Tuple[int, int]]) -> Optional[np.ndarray]:
        """
        Args:
            points: 6 (x, y) pixel tuples (only the first 4 corners are used).
        Returns:
            H (3, 3) float64 homography matrix, or None on failure.
        """
        corner_px = np.array(points[:4], dtype=np.float64)
        H, mask = cv2.findHomography(corner_px, self.COURT_2D_POINTS)
        if H is None:
            self.logger.warning("findHomography returned None.")
            return None
        return H


class CameraPositionCalculator(CourtPoints):
    """Pure math: 6 pixel points + frame dims → K, R, tvec, camera_pos."""

    def __init__(self, court_dims: Dict[str, float], frame_width: int,
                 frame_height: int, video_name: str):
        super().__init__(court_dims)
        self.video_name = video_name
        self.logger     = get_logger(__name__)

        f  = float(frame_width)           # focal length ≈ image width (pixels)
        cx = frame_width  / 2.0
        cy = frame_height / 2.0
        self.K    = np.array([[f, 0, cx],
                              [0, f, cy],
                              [0, 0,  1]], dtype=np.float64)   # intrinsic matrix
        self.dist = np.zeros((4, 1), dtype=np.float64)

    def calculate(self, points: List[Tuple[int, int]]) -> Optional[Dict[str, Any]]:
        """
        Args:
            points: 6 (x, y) pixel tuples.
        Returns:
            Dict with keys {K, R, tvec, camera_pos} on success, or None on failure.
        """
        all_px = np.array(points, dtype=np.float32)
        ok, rvec, tvec = cv2.solvePnP(
            self.COURT_3D_POINTS, all_px, self.K, self.dist,
            flags=cv2.SOLVEPNP_ITERATIVE
        )
        if not ok:
            self.logger.warning(f"solvePnP failed for {self.video_name}.")
            return None

        R, _       = cv2.Rodrigues(rvec)          # (3, 3) rotation matrix
        camera_pos = (-R.T @ tvec).flatten()       # (3,) camera position in world coords
        self.logger.info(
            f"Camera position (m): x={camera_pos[0]:.3f}  "
            f"y={camera_pos[1]:.3f}  z={camera_pos[2]:.3f}"
        )
        return {
            'K':          self.K,
            'R':          R,
            'tvec':       tvec,
            'camera_pos': camera_pos,
        }


# ── 4. CalibrationPipeline (Orchestrator) ────────────────────────────────────
# Sole responsibility: coordinate path resolution, segment loading, UI, calculators, and cache I/O.

class CalibrationPipeline:
    """
    Orchestrates the full calibration workflow:
      1. Resolve paths
      2. Load caches
      3. Determine which videos still need calibration
      4. For each video: load segments → call CalibrationUI → call Calculator
      5. Update & save caches
    """

    def __init__(self):
        self.resolver         = PathResolver()
        self.logger           = get_logger(__name__)
        self.video_dir        = self.resolver.get_path("global.video_dir")
        self.homography_cache = self.resolver.get_path("dataset_creation.homography_cache")
        self.camera_pos_cache = self.resolver.get_path("dataset_creation.camera_pose_cache")
        self.segments_path    = self.resolver.get_path("dataset_creation.segments_dir")
        self.court_dims       = self.resolver.get_config("dataset_creation.court")
        self.ui               = CalibrationUI()

    # ── Cache helpers ─────────────────────────────────────────────────────────

    @staticmethod
    def _cached_basenames(d: dict) -> set:
        return {os.path.basename(k) for k in d.keys()}

    def _load_caches(self) -> Tuple[dict, dict]:
        """Load homography and camera-pose caches from disk, or return empty dicts."""
        matrices     = {}
        camera_poses = {}

        if os.path.exists(self.homography_cache):
            with open(self.homography_cache, 'rb') as f:
                matrices = pickle.load(f)
            self.logger.info(f"Loaded {len(matrices)} homography matrices from cache.")
        else:
            self.logger.info("No homography cache found. Starting fresh.")

        if os.path.exists(self.camera_pos_cache):
            with open(self.camera_pos_cache, 'rb') as f:
                camera_poses = pickle.load(f)
            self.logger.info(f"Loaded {len(camera_poses)} camera poses from cache.")
        else:
            self.logger.info("No camera pose cache found. Starting fresh.")

        return matrices, camera_poses

    def save_caches(self, matrices: dict, camera_poses: dict) -> None:
        """Persist both caches to disk."""
        with open(self.homography_cache, 'wb') as f:
            pickle.dump(matrices, f)
        self.logger.info(f"Homography cache saved → {self.homography_cache} ({len(matrices)} entries)")

        with open(self.camera_pos_cache, 'wb') as f:
            pickle.dump(camera_poses, f)
        self.logger.info(f"Camera pose cache saved → {self.camera_pos_cache} ({len(camera_poses)} entries)")

    # ── Segment helper ────────────────────────────────────────────────────────

    def _get_start_frame(self, video: str) -> int:
        """
        Load the segments JSON for *video* and return the first frame of the
        first segment. Falls back to 7 500 if the file is missing or malformed.
        """
        seg_json = os.path.join(self.segments_path, video.split('.')[0] + '.json')
        if os.path.exists(seg_json):
            try:
                with open(seg_json, 'r') as f:
                    segments = json.load(f).get('segments', [])
                if segments and len(segments[0]) >= 1:
                    return segments[0][0]
            except Exception as e:
                self.logger.warning(f"Could not read segments JSON for {video}: {e}")
        return 7500   # broadcast footage default

    # ── Per-video calibration ─────────────────────────────────────────────────

    def _load_frame(self, video_path: str, frame_no: int) -> Optional[np.ndarray]:
        """Read a single frame from *video_path* at *frame_no*. Returns None on failure."""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            self.logger.error(f"Cannot open video: {video_path}")
            return None
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
        ret, frame = cap.read()
        cap.release()
        if not ret:
            self.logger.error(f"Cannot read frame {frame_no} from {video_path}")
            return None
        return frame

    def _calibrate_homography(self, vid: str, video_path: str,
                              start_frame: int) -> Optional[np.ndarray]:
        """
        Attempt up to 30 frames to collect 6 points and compute H.
        Returns the (3, 3) homography matrix or None.
        """
        calc    = HomographyCalculator(self.court_dims)
        frame_no = start_frame

        for _ in range(30):
            self.logger.info(f"Homography calibration for {vid} at frame {frame_no} …")
            frame = self._load_frame(video_path, frame_no)
            if frame is None:
                frame_no += 250
                continue

            points = self.ui.collect_points(frame)
            if points is None:
                frame_no += 250
                continue

            H = calc.calculate(points)
            if H is not None:
                self.logger.info(f"✅ Homography computed for {vid}.")
                return H
            frame_no += 250

        self.logger.warning(f"Homography calibration failed for {vid} after 30 attempts.")
        return None

    def _calibrate_camera_pose(self, vid: str, video_path: str,
                               start_frame: int) -> Optional[Dict[str, Any]]:
        """
        Attempt up to 30 frames to collect 6 points and compute the camera pose.
        Returns a pose dict {K, R, tvec, camera_pos} or None.
        """
        cap      = cv2.VideoCapture(video_path)
        frame_w  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_h  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        calc     = CameraPositionCalculator(self.court_dims, frame_w, frame_h, vid)
        frame_no = start_frame

        for _ in range(30):
            self.logger.info(f"Camera-pose calibration for {vid} at frame {frame_no} …")
            frame = self._load_frame(video_path, frame_no)
            if frame is None:
                frame_no += 250
                continue

            points = self.ui.collect_points(frame)
            if points is None:
                frame_no += 250
                continue

            pose = calc.calculate(points)
            if pose is not None:
                self.logger.info(f"✅ Camera pose computed for {vid}.")
                return pose
            frame_no += 250

        self.logger.warning(f"Camera-pose calibration failed for {vid} after 30 attempts.")
        return None

    # ── Remaining-videos helper ───────────────────────────────────────────────

    def _get_remaining_videos(self, videos: List[str],
                              matrices: dict,
                              camera_poses: dict) -> Dict[str, List[str]]:
        """Return {video: [tasks]} for any video not yet fully cached."""
        h_cached  = self._cached_basenames(matrices)
        cp_cached = self._cached_basenames(camera_poses)

        remaining = {}
        for vid in videos:
            needs_h  = vid not in h_cached
            needs_cp = vid not in cp_cached
            if needs_h and needs_cp:
                remaining[vid] = ["homography", "camera_pose"]
            elif needs_h:
                remaining[vid] = ["homography"]
            elif needs_cp:
                remaining[vid] = ["camera_pose"]
        return remaining

    # ── Entry point ───────────────────────────────────────────────────────────

    def run(self):
        videos = sorted(
            f for f in os.listdir(self.video_dir)
            if f.lower().endswith(('.mp4', '.avi', '.mov'))
        )

        matrices, camera_poses = self._load_caches()
        remaining              = self._get_remaining_videos(videos, matrices, camera_poses)

        if not remaining:
            self.logger.info("All videos already calibrated. Nothing to do.")
            return

        cache_updated = False

        for vid, tasks in remaining.items():
            video_path  = os.path.join(self.video_dir, vid)
            start_frame = self._get_start_frame(vid)

            if "homography" in tasks:
                H = self._calibrate_homography(vid, video_path, start_frame)
                if H is not None:
                    matrices[vid]  = H
                    cache_updated  = True

            if "camera_pose" in tasks:
                pose = self._calibrate_camera_pose(vid, video_path, start_frame)
                if pose is not None:
                    camera_poses[vid] = pose
                    cache_updated     = True

        if cache_updated:
            self.save_caches(matrices, camera_poses)
        else:
            self.logger.info("No new calibrations were completed.")
