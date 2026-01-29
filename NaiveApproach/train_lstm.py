import joblib
import json
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from lstm import ShotLSTM 

SRC_DIR = "processed_data/"
SEED = 42

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IGNORE_INDEX = -100

BATCH_SIZE = 32
LR = 1e-3
EPOCHS = 50
HIDDEN_DIM = 256
NUM_LAYERS = 2
GRAD_CLIP = 1.0

TRAIN_RATIO = 0.75
VAL_RATIO = 0.15
TEST_RATIO = 0.10

PATIENCE = 6             
USE_CLASS_WEIGHTS = False

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if DEVICE.type == "cuda":
    torch.cuda.manual_seed_all(SEED)

src_dir = "processed_data/"
sequences = joblib.load(src_dir + "sequences.pkl")
shot_label_encoder = joblib.load(src_dir + "shot_label_encoder.pkl")
features = json.load(open(src_dir + "feature_list.json"))

num_classes = len(shot_label_encoder.classes_)
input_dim = len(features)

print(f"Loaded {len(sequences)} sequences.")
print(f"Feature dimension: {input_dim}")
print(f"Number of classes: {num_classes}")

class RallyDataset(Dataset):
    def __init__(self, sequences):
        self.sequences = sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        X, y = self.sequences[idx]
        return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.long)

def collate_fn(batch):
    Xs, ys = zip(*batch)
    lengths = torch.tensor([len(x) for x in Xs], dtype=torch.long)

    X_pad = pad_sequence(Xs, batch_first=True, padding_value=0.0)
    y_pad = pad_sequence(ys, batch_first=True, padding_value=IGNORE_INDEX)

    return X_pad, y_pad, lengths

n = len(sequences)
indices = list(range(n))
random.shuffle(indices)

n_train = int(n * TRAIN_RATIO)
n_val = int(n * VAL_RATIO)
train_idx = indices[:n_train]
val_idx = indices[n_train:n_train + n_val]
test_idx = indices[n_train + n_val:]

train_seqs = [sequences[i] for i in train_idx]
val_seqs   = [sequences[i] for i in val_idx]
test_seqs  = [sequences[i] for i in test_idx]

print(f"Split: train={len(train_seqs)}, val={len(val_seqs)}, test={len(test_seqs)}")

train_loader = DataLoader(RallyDataset(train_seqs), batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
val_loader   = DataLoader(RallyDataset(val_seqs)  , batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
test_loader  = DataLoader(RallyDataset(test_seqs) , batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

def compute_class_weights(sequences, num_classes, ignore_index=IGNORE_INDEX):
    counts = np.zeros(num_classes, dtype=np.int64)
    for X, y in sequences:
        y_flat = y.flatten()
        y_flat = y_flat[y_flat != ignore_index]
        if y_flat.size:
            binc = np.bincount(y_flat, minlength=num_classes)
            counts += binc
    
    counts = counts.astype(np.float64) + 1.0
    weights = 1.0 / counts
    weights = weights / weights.sum() * num_classes  
    return torch.tensor(weights, dtype=torch.float32)

if USE_CLASS_WEIGHTS:
    class_weights = compute_class_weights(train_seqs, num_classes)
    print("Class weights computed.")
else:
    class_weights = None

model = ShotLSTM(input_dim=input_dim, hidden_dim=HIDDEN_DIM, num_classes=num_classes, num_layers=NUM_LAYERS).to(DEVICE)
if class_weights is not None:
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(DEVICE), ignore_index=IGNORE_INDEX)
else:
    criterion = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)

optimizer = torch.optim.Adam(model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)

def flatten_preds_targets(logits, targets):
    preds = logits.argmax(dim=-1).cpu().numpy().reshape(-1)
    targets = targets.cpu().numpy().reshape(-1)
    mask = targets != IGNORE_INDEX
    return preds[mask], targets[mask]

def evaluate(loader, model, device):
    model.eval()
    losses = []
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for X, y, lengths in loader:
            X, y, lengths = X.to(device), y.to(device), lengths.to(device)
            logits = model(X, lengths)
            B, T, C = logits.shape
            loss = criterion(logits.reshape(B*T, C), y.reshape(B*T))
            losses.append(loss.item())
            p, t = flatten_preds_targets(logits, y)
            if len(t) > 0:
                all_preds.append(p)
                all_targets.append(t)
    if len(losses) == 0:
        return 0.0, 0.0, 0.0
    mean_loss = float(np.mean(losses))
    if len(all_targets) == 0:
        return mean_loss, 0.0, 0.0
    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)
    acc = accuracy_score(all_targets, all_preds)
    f1 = f1_score(all_targets, all_preds, average="macro", zero_division=0)
    return mean_loss, acc, f1

best_val_f1 = -1.0
patience_counter = 0

for epoch in range(1, EPOCHS + 1):
    model.train()
    train_losses = []
    for X, y, lengths in train_loader:
        X, y, lengths = X.to(DEVICE), y.to(DEVICE), lengths.to(DEVICE)
        optimizer.zero_grad()
        logits = model(X, lengths)
        B, T, C = logits.shape
        loss = criterion(logits.reshape(B*T, C), y.reshape(B*T))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()
        train_losses.append(loss.item())

    train_loss = float(np.mean(train_losses))
    val_loss, val_acc, val_f1 = evaluate(val_loader, model, DEVICE)
    scheduler.step(val_loss)

    print(f"Epoch {epoch}/{EPOCHS} | TrainLoss: {train_loss:.4f} | ValLoss: {val_loss:.4f} | ValAcc: {val_acc:.4f} | ValF1: {val_f1:.4f}")

    if val_f1 > best_val_f1:
        best_val_f1 = val_f1
        torch.save({
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "epoch": epoch,
            "feature_cols": features,
            "shot_classes": shot_label_encoder.classes_.tolist()
        }, "best_lstm_checkpoint.pt")
        patience_counter = 0
        print(f"  Saved new best model (ValF1={val_f1:.4f})")
    else:
        patience_counter += 1
        if patience_counter >= PATIENCE:
            print("Early stopping triggered.")
            break

print("Loading best model for final evaluation on test set...")
ckpt = torch.load("best_lstm_checkpoint.pt", map_location=DEVICE)
model.load_state_dict(ckpt["model_state"])

test_loss, test_acc, test_f1 = evaluate(test_loader, model, DEVICE)
print(f"TEST  Loss: {test_loss:.4f}  Acc: {test_acc:.4f}  F1: {test_f1:.4f}")

all_preds = []
all_targets = []
with torch.no_grad():
    for X, y, lengths in test_loader:
        X, y, lengths = X.to(DEVICE), y.to(DEVICE), lengths.to(DEVICE)
        logits = model(X, lengths)
        p, t = flatten_preds_targets(logits, y)
        if len(t) > 0:
            all_preds.append(p); all_targets.append(t)

if len(all_targets) > 0:
    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)
    cm = confusion_matrix(all_targets, all_preds, labels=list(range(num_classes)))
    print("Confusion matrix shape:", cm.shape)

torch.save(model.state_dict(), "lstm_model.pt")
print("Training complete. Model saved to lstm_model.pt")