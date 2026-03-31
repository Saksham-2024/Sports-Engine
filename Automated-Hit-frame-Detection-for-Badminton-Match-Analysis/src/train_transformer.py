import os
import json
import pickle
import zipfile
import shutil
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm
import warnings
from sklearn.exceptions import InconsistentVersionWarning

warnings.filterwarnings("ignore", category=InconsistentVersionWarning)

# Import your model
from models.transformer import OptimusPrime

# --- CONFIGURATION ---
DATA_ROOT = "/home/saksham/projects and programming/BTech_Project/dataset"
TRAIN_DIR = DATA_ROOT + "/KSeq_train_dataset"
TEST_DIR = DATA_ROOT + "/KSeq_test_dataset"
SCALER_PATH = "./models/weights/scaler.pickle"
CHECKPOINT_PATH = "./models/weights/opt_checkpoint.pt"
FINAL_PATH = "./models/weights/opt_path.pt"

# --- DATASET CLASS ---
class KSeqDataset(Dataset):
    def __init__(self, root_dir, scaler_path, max_len=600):
        self.max_len = max_len
        self.files = list(Path(root_dir).rglob("*.json"))
        
        with open(scaler_path, 'rb') as f:
            self.scaler = pickle.load(f)
        print(f"Loaded {len(self.files)} rallies from {root_dir}. Scaler initialized.")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        with open(self.files[idx], 'r') as f:
            data = json.load(f)

        frames = data['frames']
        seq_len = min(len(frames), self.max_len)

        features = np.zeros((self.max_len, 2, 17, 2), dtype=np.float32)
        labels = np.full((self.max_len,), 3, dtype=np.int64) # 3 is Pad

        for i in range(seq_len):
            raw_joints = np.array(frames[i]['joint']) # [2, 17, 2]
            scaled_flat = self.scaler.transform(raw_joints.reshape(1, -1))
            features[i] = scaled_flat.reshape(2, 17, 2)
            labels[i] = frames[i]['label']

        return torch.from_numpy(features), torch.from_numpy(labels)

# --- EVALUATION ---
def evaluate(model, loader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for feats, labs in loader:
            feats, labs = feats.to(device), labs.to(device)
            src_pad_mask = model.create_src_pad_mask(feats)
            out = model(feats, src_pad_mask=src_pad_mask)
            preds = torch.argmax(out, dim=2).view(-1)
            labs = labs.view(-1)
            mask = labs != 3 # Ignore padding
            correct += (preds[mask] == labs[mask]).sum().item()
            total += mask.sum().item()
    return 100 * correct / total if total > 0 else 0

# --- TRAINING ---
def run_training(resume_from_checkpoint=True):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # The paper architecture [cite: 163, 219, 220]
    model = OptimusPrime(
        num_tokens=4,
        dim_model=2048,
        num_heads=8,
        num_encoder_layers=8,
        dim_feedforward=2048
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=1e-5) # Paper LR: 1e-5 [cite: 163]
    criterion = nn.CrossEntropyLoss(ignore_index=3)

    start_epoch = 0
    if resume_from_checkpoint and os.path.exists(CHECKPOINT_PATH):
        print(f"Loading checkpoint from {CHECKPOINT_PATH}...")
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"Resuming from Epoch {start_epoch}")

    # Pointing to unzipped data folders
    train_ds = KSeqDataset(TRAIN_DIR, SCALER_PATH)
    test_ds = KSeqDataset(TEST_DIR, SCALER_PATH)

    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False)

    for epoch in range(start_epoch, 100): # 100 Epochs total [cite: 163]
        model.train()
        epoch_loss = 0

        # LR Decay 90% after 70th epoch [cite: 163]
        if epoch >= 70:
            for g in optimizer.param_groups: g['lr'] = 1e-6

        pbar = tqdm(train_loader, desc=f"Epoch {epoch:02d}/100", unit="rally")

        for feats, labs in pbar:
            feats, labs = feats.to(device), labs.to(device)
            optimizer.zero_grad()

            src_pad_mask = model.create_src_pad_mask(feats)
            out = model(feats, src_pad_mask=src_pad_mask)

            loss = criterion(out.view(-1, 4), labs.view(-1))
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        val_acc = evaluate(model, test_loader, device)
        print(f"Finished Epoch {epoch} | Avg Loss: {epoch_loss/len(train_loader):.4f} | Val Accuracy: {val_acc:.2f}%")

        # Save checkpoint every epoch locally
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, CHECKPOINT_PATH)

    torch.save(model.state_dict(), FINAL_PATH)
    print(f"Training Complete. Weights saved to {FINAL_PATH}")

if __name__ == "__main__":
    run_training()