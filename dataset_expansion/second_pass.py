import subprocess
import os
import sys
import time
import threading

CONDA_PYTHON = "/home/saksham/miniconda3/envs/hit_detector_env/bin/python"
REPO_PATH = "/home/saksham/projects and programming/BTech_Project/Automated-Hit-frame-Detection-for-Badminton-Match-Analysis"
CONFIG_PATH = os.path.join(REPO_PATH, "configs", "ai_coach.yaml")

CWD = os.path.abspath("../dataset_expansion")
VIDEO_DIR = os.path.join(CWD, 'unlabeled_videos')
JOINT_PATH = os.path.join(CWD, 'output/joints')
RALLY_PATH = os.path.join(CWD, 'output/rallies')
VIDEO_SAVE_PATH = os.path.join(CWD, 'output/videos')

stop_event = threading.Event()

def get_progress():
    total_videos = len([f for f in os.listdir(VIDEO_DIR) 
                        if f.lower().endswith(('.mp4', '.avi', '.mov'))])
    
    
    if not os.path.exists(RALLY_PATH):
        processed_count = 0
    else:
        processed_count = len([d for d in os.listdir(RALLY_PATH) 
                               if os.path.isdir(os.path.join(RALLY_PATH, d))])
    
    return processed_count, total_videos

def print_status_bar():
    while not stop_event.is_set():
        curr, tot = get_progress()
        
        percent = (curr / tot * 100) if tot > 0 else 0
        bar = '█' * int(20 * curr // tot) if tot > 0 else ''
        sys.stdout.write(f'\rOverall Match Progress: |{bar:<20}| {percent:.1f}% ({curr}/{tot})')
        sys.stdout.flush()
        
        time.sleep(2)

def run_batch_inference():
    """Executes the prediction script once to process the whole folder."""
    command = [
            CONDA_PYTHON, 
            "src/main.py", 
            "--yaml_path", CONFIG_PATH
        ]
    try:
        print(f"[RUNNING] Starting Batch Inference for all videos...")
        subprocess.run(
            command, 
            cwd=REPO_PATH,       
            check=True, 
            text=True
        )
        print("[SUCCESS] All videos in the directory have been processed.")
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Batch inference failed: {e}")
    finally:
        stop_event.set()

if __name__ == "__main__":
    for p in [JOINT_PATH, RALLY_PATH, VIDEO_SAVE_PATH]:
        os.makedirs(p, exist_ok=True)

    ml_thread = threading.Thread(target=run_batch_inference)
    progress_thread = threading.Thread(target=print_status_bar)
    
    ml_thread.start()
    progress_thread.start()

    ml_thread.join()
    progress_thread.join()

    print("Processing Complete")