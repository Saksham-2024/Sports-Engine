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

class SACNN(nn.Module):
    def __init__(self):
        super(SACNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 2))
        self.bn1 = nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 2))
        self.bn2 = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv3 = nn.Conv2d(64, 32, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=(2, 2))
        self.bn3 = nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.l1 = nn.Linear(27 * 27 * 32, 2)
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, x):
        x = self.bn1(self.pool1(self.conv1(x)))
        x = F.relu(x)
        x = self.bn2(self.pool2(self.conv2(x)))
        x = F.relu(x)
        x = self.bn3(self.pool3(self.conv3(x)))
        x = F.relu(x)
        x = x.view(-1, 27 * 27 * 32)
        x = self.l1(x)
        x = F.relu(x)
        out = self.dropout(x)
        return out

class ShotAngleQueue(object):
    def __init__(self, max_len):
        self.max_len = max_len
        self.queue = []
        self.last_sa = 0
    
    def push(self, frame_info):
        sa_condition = None
        if len(self.queue) < self.max_len:
            self.queue.append(frame_info)
            return None, None
        else:
            first_info = self.queue.pop(0)
            sa, sa_condition = self.__check_sa_conditon(first_info[0])
            self.last_sa = sa
            first_info[0] = sa
            self.queue.append(frame_info)
            return first_info, sa_condition

    def __check_sa_conditon(self, sa):
        '''
        return sa, cond in {0, 1, 2, 3}
        cond :  last sa  ->   sa
          0  :      0    ->   0
          1  :      0    ->   1
          2  :      1    ->   1
          3  :      1    ->   0
        '''
        sum_val = sa
        if self.last_sa == 1 and sa == 0:
            for info in self.queue:
                sum_val += info[0]
            if sum_val <= (self.max_len / 2):
                return 0, 3  # Flip to 0
            else:
                return 1, 2  # Keep 1
        elif self.last_sa == 0 and sa == 1:
            for info in self.queue:
                sum_val += info[0]
            if sum_val >= (self.max_len / 2):
                return 1, 1  # Flip to 1
            else:
                return 0, 0  # Keep 0
        elif self.last_sa == 1 and sa == 1:
            return 1, 2
        elif self.last_sa == 0 and sa == 0:
            return 0, 0

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
