import os
import torch
import torchvision
from torchvision.models.detection import keypointrcnn_resnet50_fpn, KeypointRCNN_ResNet50_FPN_Weights

# 1. Define your desired location and filename
# Replace this with your actual B.Tech Project path
save_directory = "/home/saksham/projects and programming/BTech_Project/Automated-Hit-frame-Detection-for-Badminton-Match-Analysis/src/models/weights"
file_name = "kpRCNN.pth"
full_path = os.path.join(save_directory, file_name)

# 2. Ensure the directory exists
if not os.path.exists(save_directory):
    os.makedirs(save_directory)
    print(f"Created directory: {save_directory}")

print("Starting download... This file is approximately 226 MB.")

# 3. Load the model with standard COCO weights
# 'DEFAULT' corresponds to KeypointRCNN_ResNet50_FPN_Weights.COCO_V1
model = keypointrcnn_resnet50_fpn(weights=KeypointRCNN_ResNet50_FPN_Weights.DEFAULT)

# 4. Save the model weights (state_dict) to your desired path
torch.save(model.state_dict(), full_path)

print(f"Successfully saved weights to: {full_path}")