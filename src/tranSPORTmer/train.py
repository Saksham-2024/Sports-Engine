import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import json
import os
import numpy as np
from tqdm import tqdm
import pickle
import yaml
from transformer import create_model

with open('../../configs/configs.yaml', 'r') as f:
    configs = yaml.safe_load(f)


class TrajectoryDataset(Dataset):
    def __init__(self, data_file):
        with open(data_file, 'rb') as f:
            data = pickle.load(f)
        self.input_traj = torch.from_numpy(data['input_trajectories']).float()
        self.input_masks = torch.from_numpy(data['input_masks']).float()
        self.target_traj = torch.from_numpy(data['target_trajectories']).float()
        self.target_masks = torch.from_numpy(data['target_masks']).float()
        self.metadata = data['metadata']
    
    def __len__(self):
        return len(self.input_traj)
    
    def __getitem__(self, idx):
        return {
            'input_traj': self.input_traj[idx],
            'input_mask': self.input_masks[idx],
            'target_traj': self.target_traj[idx],
            'target_mask': self.target_masks[idx],
            'metadata': self.metadata[idx]
        }

class ADELoss(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, pred, target, target_mask):
        error = torch.norm(pred - target, p=2, dim=-1)
        valid_mask = target_mask == 0
        if valid_mask.sum() == 0:
            return torch.tensor(0.0, device=pred.device, requires_grad=True)
        return error[valid_mask].mean()

def compute_metrics(pred, target, target_mask):
    valid_mask = target_mask == 0 
    pred_xy = pred[..., :2]
    target_xy = target[..., :2]
    pred_z = pred[..., 2:3]
    target_z = target[..., 2:3]
    
    error_xy = torch.norm(pred_xy - target_xy, p=2, dim=-1)
    ade_xy = error_xy[valid_mask].mean() if valid_mask.sum() > 0 else torch.tensor(0.0)
    
    error_z = torch.abs(pred_z.squeeze(-1) - target_z.squeeze(-1))
    ade_z = error_z[valid_mask].mean() if valid_mask.sum() > 0 else torch.tensor(0.0)
    
    error_overall = torch.norm(pred - target, p=2, dim=-1)
    ade_overall = error_overall[valid_mask].mean() if valid_mask.sum() > 0 else torch.tensor(0.0)
    
    fde_errors = error_overall[:, -1, :] 
    fde_valid = valid_mask[:, -1, :]
    fde = fde_errors[fde_valid].mean() if fde_valid.sum() > 0 else torch.tensor(0.0)
    
    return {
        'ade_xy': ade_xy.item(), 'ade_z': ade_z.item(), 'ade_overall': ade_overall.item(), 'fde': fde.item()
    }

def train_epoch(model, loader, optimizer, loss_fn, device):
    model.train()
    total_loss = 0
    num_batches = 0
    
    for batch in tqdm(loader, desc='Training'):
        in_traj = batch['input_traj'].to(device)
        # Squeeze 4D mask [B, T, N, C] to 3D [B, T, N]
        in_mask = batch['input_mask'][:, :, :, 0].to(device)
        
        tgt_traj = batch['target_traj'].to(device)
        tgt_mask = batch['target_mask'][:, :, :, 0].to(device)
        
        pred = model(in_traj, in_mask)
        loss = loss_fn(pred, tgt_traj, tgt_mask)
        
        # Skip batch update if all targets were masked out
        if loss.item() == 0.0:
            continue
            
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
    
    return total_loss / max(num_batches, 1)

def validate(model, loader, loss_fn, device):
    model.eval()
    total_loss = 0
    num_batches = 0
    all_metrics = {'ade_xy': [], 'ade_z': [], 'ade_overall': [], 'fde': []}
    
    with torch.no_grad():
        for batch in tqdm(loader, desc='Validating'):
            in_traj = batch['input_traj'].to(device)
            in_mask = batch['input_mask'][:, :, :, 0].to(device)
            tgt_traj = batch['target_traj'].to(device)
            tgt_mask = batch['target_mask'][:, :, :, 0].to(device)
            
            pred = model(in_traj, in_mask)
            loss = loss_fn(pred, tgt_traj, tgt_mask)
            
            if loss.item() == 0.0:
                continue
                
            total_loss += loss.item()
            num_batches += 1
            
            metrics = compute_metrics(pred, tgt_traj, tgt_mask)
            for key in all_metrics:
                all_metrics[key].append(metrics[key])
    
    avg_loss = total_loss / max(num_batches, 1)
    avg_metrics = {k: np.mean(v) if v else 0.0 for k, v in all_metrics.items()}
    return avg_loss, avg_metrics

def main():
    config = {
        'input_dim': configs['tranSPORTmer']['model']['input_dim'],
        'd_model': configs['tranSPORTmer']['model']['d_model'],
        'num_heads': configs['tranSPORTmer']['model']['num_heads'],
        'd_ff': configs['tranSPORTmer']['model']['d_ff'],
        'dropout': configs['tranSPORTmer']['model']['dropout'],
        'num_agents': configs['tranSPORTmer']['model']['num_agents'],
        'seq_len': configs['tranSPORTmer']['model']['seq_len'],
        'target_seq_len': configs['tranSPORTmer']['model']['target_seq_len']
    }
    train_config = {'epochs': configs['tranSPORTmer']['training']['epochs'], 
                    'batch_size': configs['tranSPORTmer']['training']['batch_size'], 
                    'learning_rate': configs['tranSPORTmer']['training']['learning_rate'], 
                    'device': 'cuda' if torch.cuda.is_available() else 'cpu'}
    
    device = torch.device(train_config['device'])
    
    train_dataset = TrajectoryDataset(Path(os.path.join(configs['global']['project_root'], configs['tranSPORTmer']['data']['training_data_dir'], 'train_windows.pkl')))
    val_dataset = TrajectoryDataset(Path(os.path.join(configs['global']['project_root'], configs['tranSPORTmer']['data']['training_data_dir'], 'val_windows.pkl')))
    
    train_loader = DataLoader(train_dataset, batch_size=train_config['batch_size'], shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=train_config['batch_size'], shuffle=False, num_workers=0)
    
    model = create_model(config).to(device)
    loss_fn = ADELoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=train_config['learning_rate'])
    
    best_val_loss = float('inf')
    checkpoint_file = Path(os.path.join(configs['global']['project_root'], configs['tranSPORTmer']['weights']['best_weights']))
    checkpoint_file.parent.mkdir(parents=True, exist_ok=True)
    
    for epoch in range(1, train_config['epochs'] + 1):
        train_loss = train_epoch(model, train_loader, optimizer, loss_fn, device)
        val_loss, val_metrics = validate(model, val_loader, loss_fn, device)
        
        print(f"\nEpoch {epoch}/{train_config['epochs']}")
        print(f"  Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")
        print(f"  Metrics: ADE={val_metrics['ade_overall']:.4f}, FDE={val_metrics['fde']:.4f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), checkpoint_file)

if __name__ == '__main__':
    main()