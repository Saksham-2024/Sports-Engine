import argparse
import os
import cv2
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import yaml
from src.core.utils.configs import PathResolver
from src.core.utils.logger import get_logger
from src.core.models.SACNN import SACNN, ShotAngleQueue

class FormSegments:
    def __init__(self):
        self._initialization()
        #self.run_segmentation()
    
    def _initialization(self):
        '''setup model, paths, configs, loggers etc'''
        self.resolver = PathResolver()
        self.logger = get_logger(__name__)
        self.video_dir = self.resolver.get_path("global.video_dir")
        self.output_dir = self.resolver.get_path("dataset_creation.segments_dir")
        self.model_path = self.resolver.get_path("models.sacnn")
        os.makedirs(self.output_dir, exist_ok=True)

    def _setup_model(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.logger.info(f"Using device: {self.device}")

        self.sacnn = SACNN().to(self.device)
        self.sacnn.load_state_dict(torch.load(self.model_path, map_location=self.device, weights_only=True))
        self.sacnn.eval()

        self.preprocess = transforms.Compose([
            transforms.Resize((216, 384)),
            transforms.CenterCrop((216, 216)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def _traverse_video(self, cap, frame_count, total_frames, frame_rate):
        segments = []
        sa_queue = ShotAngleQueue(max_len=5)
        current_segment_start = None
        while cap.isOpened() and frame_count < total_frames:
            ret, frame = cap.read()
            if not ret: break

            if frame_count % frame_rate == 0:
                pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                tensor = self.preprocess(pil_img).unsqueeze(0).to(self.device)
                
                with torch.no_grad():
                    out = self.sacnn(tensor)
                    sa = torch.argmax(out, dim=1).item()

                frame_info, sa_condition = sa_queue.push([sa, frame_count])
                
                if frame_info:
                    past_sa, past_frame = frame_info[0], frame_info[1]
                    
                    if sa_condition == 1: # 0 -> 1 (Start of gameplay)
                        current_segment_start = past_frame
                        
                    elif sa_condition == 3: # 1 -> 0 (End of gameplay)
                        if current_segment_start is not None:
                            segments.append([current_segment_start, past_frame])
                            current_segment_start = None

            if frame_count % 5000 == 0:
                print(f"  ... {frame_count}/{total_frames} ({(frame_count/total_frames)*100:.1f}%) | Found segments: {len(segments)}")
            frame_count += 1     

        cap.release()
        # Handle case where video ends mid-segment
        if current_segment_start is not None:
            segments.append([current_segment_start, frame_count])
        return segments

    def _handle_single_segment(self, segments):
        # professional match videos switch camera angles in between rallies and thus can be easily identified as time between 2 rallies
        # thus, we can count number of rallies. In amateur, there is no camera switching, therefore only one segment is identified.
        # break these segments into 30/60 sec segments at 25 fps
        if len(segments) == 1:
            start = segments[0][0]
            end = segments[0][1]
            new_segments = []
            while start < end:
                new_segments.append([start, min(start + 750, end)])
                start += 750
            segments = new_segments
        
        return segments

    def _segment_single_video(self, n):
        vid_name = f'match{n}'
        vid_path = os.path.join(self.video_dir, f'{vid_name}.mp4')
        if not os.path.exists(vid_path):
            self.logger.info(f"Video {vid_name} deos'nt exist.")
            return None, None
        
        out_path = os.path.join(self.output_dir, f'{vid_name}.json')
        if os.path.exists(out_path):
            self.logger.info(f"Video {vid_name} has already been segmented")
            return None, None
        
        cap = cv2.VideoCapture(vid_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        time_rate = 0.1
        frame_rate = round(int(fps) * time_rate)        
        frame_count = 0

        segments = self._traverse_video(cap, frame_count, total_frames, frame_rate)
        segments = self._handle_single_segment(segments)
        return segments, out_path

    def _write_segments(self, segments, out_path):
        with open(out_path, 'w') as f:
            json.dump({"segments": segments}, f, indent=2)

    def run_segmentation(self):
        self._setup_model()
        for i in range(1, len(os.listdir(self.video_dir))+1):
            segments, out_path = self._segment_single_video(i)
            if segments is None:
                continue
            self._write_segments(segments, out_path)
            self.logger.info(f"✓ match{i} complete: {len(segments)} gameplay segments found.")
