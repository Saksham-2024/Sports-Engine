import cv2
import numpy as np
import pickle
import os

video_dir = './unlabeled_videos'
videos = [f for f in os.listdir(video_dir)]

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
        
        points = []

        # Click the 4 corners of the court        
        # Order: top-left, top-right, bottom-right, bottom-left

        def click_event(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                points.append((x, y))
                cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)
                cv2.imshow('Mark 4 Court Corners', frame)

        cv2.imshow('Mark 4 Court Corners', frame)
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


def gather_matrices():
    matrices = {}
    for video in videos:
        H = np.zeros((3,3), dtype=np.float64)
        video_path = video_dir + video
        start_frame = 7500
        # get the stroke that has normal camera angle to calculate homography
        i = 30
        while i > 0:
            ret, H = homography(start_frame, video_path)
            if ret:
                matrices[video_path] = H
                break
            else:
                print(f'error calculating homography for match {video}, mark 4 corners again')
                i -= 1
                start_frame += 250
            
    return matrices

homography_cache_file = 'homography_cache.pkl'
matrices = gather_matrices()
with open(homography_cache_file, 'wb') as f:
    pickle.dump(matrices, f)
print("Homography matrices calculated and cached.")