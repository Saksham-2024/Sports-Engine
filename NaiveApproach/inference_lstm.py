# inference_lstm.py

import joblib
import json
import torch
import random
import numpy as np
from lstm import ShotLSTM

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
src_dir = "processed_data/"

sequences = joblib.load(src_dir + "sequences.pkl")
shot_label_encoder = joblib.load(src_dir + "shot_label_encoder.pkl")
feature_list = json.load(open(src_dir + "feature_list.json"))

num_classes = len(shot_label_encoder.classes_)
input_dim = len(feature_list)

model = ShotLSTM(
    input_dim=input_dim,
    hidden_dim=256,
    num_classes=num_classes,
    num_layers=3
).to(DEVICE)

model.load_state_dict(torch.load("lstm_model.pt", map_location=DEVICE))
model.eval()

print("LSTM model loaded.")

rally_idx = random.randint(0, len(sequences) - 1)
X, y = sequences[rally_idx]

T = X.shape[0]
t = random.randint(0, T - 2)

print(f"\nRally index: {rally_idx}")
print(f"Using shots [0 → {t}] as context")

X_context = torch.tensor(X[:t+1], dtype=torch.float32).unsqueeze(0).to(DEVICE)
lengths = torch.tensor([t+1]).to(DEVICE)


with torch.no_grad():
    logits = model(X_context, lengths)
    pred_id = logits[0, -1].argmax().item()
    pred_label = shot_label_encoder.inverse_transform([pred_id])[0]

true_label = shot_label_encoder.inverse_transform([y[t]])[0]

print(f"Ground truth next shot : {true_label}")
print(f"Predicted next shot   : {pred_label}")
