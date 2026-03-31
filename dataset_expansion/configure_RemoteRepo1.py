import os
import yaml

REPO_PATH = "/home/saksham/projects and programming/BTech_Project/Automated-Hit-frame-Detection-for-Badminton-Match-Analysis"

CWD = os.path.abspath("../dataset_expansion")
CONFIG_PATH = os.path.join(REPO_PATH, "configs", "ai_coach.yaml")
VIDEO_DIR = os.path.join(CWD, 'unlabeled_videos')
JOINT_PATH = os.path.join(CWD, 'output/joints')
RALLY_PATH = os.path.join(CWD, 'output/rallies')
VIDEO_SAVE_PATH = os.path.join(CWD, 'output/videos')

def update_repo_config():
    """Updates the YAML to point to the FOLDER containing all videos."""
    with open(CONFIG_PATH, 'r') as f:
        config = yaml.safe_load(f)

    config['model']['sacnn_path'] = REPO_PATH + '/src' + config['model']['sacnn_path'][1:]
    config['model']['court_kpRCNN_path'] = REPO_PATH + '/src' + config['model']['court_kpRCNN_path'][1:]
    config['model']['kpRCNN_path'] = REPO_PATH + '/src' + config['model']['kpRCNN_path'][1:]
    config['model']['opt_path'] = REPO_PATH + '/src' + config['model']['opt_path'][1:]
    config['model']['scaler_path'] = REPO_PATH + '/src' + config['model']['scaler_path'][1:]

    config['video_directory'] = VIDEO_DIR 
    config['video_save_path'] = VIDEO_SAVE_PATH
    config['joint_save_path'] = JOINT_PATH
    config['rally_save_path'] = RALLY_PATH
    
    with open(CONFIG_PATH, 'w') as f:
        yaml.dump(config, f, sort_keys=False, default_flow_style=False)
    
update_repo_config()