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

with open('../../configs/configs.yaml', 'r') as f:
    configs = yaml.safe_load(f)

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

def run_segmentation():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load Model
    sacnn = SACNN().to(device)
    model_path = os.path.join(configs['global']['project_root'], configs['models']['sacnn'])
    sacnn.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    sacnn.eval()

    preprocess = transforms.Compose([
        transforms.Resize((216, 384)),
        transforms.CenterCrop((216, 216)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    video_dir  = os.path.join(configs['global']['project_root'], configs['global']['video_dir'])
    output_dir = os.path.join(configs['global']['project_root'], configs['dataset_creation']['segments_dir'])
    os.makedirs(output_dir, exist_ok=True)

    for i in range(1, len(os.listdir(video_dir))):
        vid_name = f'match{i}'
        vid_path = os.path.join(video_dir, f'{vid_name}.mp4')
        if not os.path.exists(vid_path):
            continue

        out_path = os.path.join(output_dir, f'{vid_name}.json')
        if os.path.exists(out_path):
            print(f"Skipping {vid_name}, already segmented.")
            continue

        print(f"\n🚀 Segmenting: {vid_name}")
        cap = cv2.VideoCapture(vid_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        time_rate = 0.1
        frame_rate = round(int(fps) * time_rate)
        
        sa_queue = ShotAngleQueue(max_len=5) # 5 corresponds to their default queue length
        
        segments = []
        current_segment_start = None
        
        frame_count = 0
        while cap.isOpened() and frame_count < total_frames:
            ret, frame = cap.read()
            if not ret: break

            if frame_count % frame_rate == 0:
                pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                tensor = preprocess(pil_img).unsqueeze(0).to(device)
                
                with torch.no_grad():
                    out = sacnn(tensor)
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

        with open(out_path, 'w') as f:
            json.dump({"segments": segments}, f, indent=2)
            
        print(f"✓ {vid_name} complete: {len(segments)} gameplay segments found.")

if __name__ == '__main__':
    run_segmentation()
