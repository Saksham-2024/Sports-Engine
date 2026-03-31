import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.preprocessing import StandardScaler #type: ignore
from sklearn.utils.class_weight import compute_class_weight #type: ignore
import matplotlib.pyplot as plt #type: ignore
import seaborn as sns #type: ignore
from sklearn.metrics import confusion_matrix, classification_report #type: ignore

print("Loading 3D Tensor...")
data = np.load('5.2-features_tensor_doubled.npz', allow_pickle=True)
X_raw = data['X']
y_raw = data['y']
num_classes = len(np.unique(y_raw))

print("Standardizing features...")
N_strokes, seq_len, num_features = X_raw.shape
X_flat = X_raw.reshape(-1, num_features)

nan_count = np.isnan(X_flat).sum()
inf_count = np.isinf(X_flat).sum()
print(f"Diagnostic: Found {nan_count} NaNs and {inf_count} Infs hiding in the data!")

X_flat = np.nan_to_num(X_flat, nan=0.0, posinf=0.0, neginf=0.0)

scaler = StandardScaler()
X_scaled_flat = scaler.fit_transform(X_flat)
X_scaled = X_scaled_flat.reshape(N_strokes, seq_len, num_features)

X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
y_tensor = torch.tensor(y_raw, dtype=torch.long)

dataset = TensorDataset(X_tensor, y_tensor)

train_size = int(0.7 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

print(f"Training on {train_size} strokes | Testing on {test_size} strokes")

class BadmintonLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout=0.50):
        super(BadmintonLSTM, self).__init__()
        
        self.lstm = nn.LSTM(
            input_size=input_size, 
            hidden_size=hidden_size, 
            num_layers=num_layers, 
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True # <--- Turn on dual-direction reading
        )
        self.dropout = nn.Dropout(dropout)
        # Double the input to the Linear layer (Forward Memory + Backward Memory)
        self.fc = nn.Linear(hidden_size * 2, num_classes)
        
    def forward(self, x):
        # We now pull 'hn' (the hidden states) alongside 'out'
        out, (hn, cn) = self.lstm(x)
        # hn shape: (num_layers * 2, batch_size, hidden_size)
        # Grab the absolute final memory state of the forward and backward passes
        forward_memory = hn[-2, :, :]
        backward_memory = hn[-1, :, :]
        # Smash them together into one complete sequence understanding
        final_memory_state = torch.cat((forward_memory, backward_memory), dim=1)
    
        # Apply our dropout handicap
        final_memory_state = self.dropout(final_memory_state)
        
        return self.fc(final_memory_state)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Training on device: {device}")

model = BadmintonLSTM(  
    input_size=106, 
    hidden_size=64,
    num_layers=2, 
    num_classes=num_classes,
    dropout=0.30
).to(device)

unique_classes, class_counts = np.unique(y_raw, return_counts=True)
stroke_names = data['classes']

print("\n Class Distribution Check:")
for name, count in zip(stroke_names, class_counts):
    print(f"{name:15s}: {count} strokes")

class_wts = compute_class_weight('balanced', classes=unique_classes, y=y_raw)
weights_tensor = torch.tensor(class_wts, dtype=torch.float32).to(device)

print(f"\n Applied Class Weights: {np.round(class_wts, 2)}")
criterion = nn.CrossEntropyLoss(weight=weights_tensor)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)

EPOCHS = 50
best_test_acc = 0.0
print("\n Starting Training Engine...")
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    correct_train = 0
    total_train = 0
    
    for batch_X, batch_y in train_loader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        # Calculate training accuracy
        _, predicted = torch.max(outputs.data, 1)
        total_train += batch_y.size(0)
        correct_train += (predicted == batch_y).sum().item()
        
    train_acc = 100 * correct_train / total_train
    
    # Evaluate on Test Set every 5 epochs
    if (epoch + 1) % 5 == 0 or epoch == 0:
        model.eval()
        correct_test = 0
        total_test = 0
        with torch.no_grad():
            for batch_X, batch_y in test_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                outputs = model(batch_X)
                _, predicted = torch.max(outputs.data, 1)
                total_test += batch_y.size(0)
                correct_test += (predicted == batch_y).sum().item()
                
        test_acc = 100 * correct_test / total_test
        print(f"Epoch [{epoch+1}/{EPOCHS}] | Loss: {total_loss/len(train_loader):.4f} | Train Acc: {train_acc:.2f}% | Test Acc: {test_acc:.2f}%")
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            torch.save(model.state_dict(), 'badminton_lstm_best.pth')
            print(f"New best model saved with {test_acc:.2f}% accuracy!")
            
        scheduler.step(test_acc)
        # Step the scheduler based on Test Accuracy
        scheduler.step(test_acc)

torch.save(model.state_dict(), 'badminton_lstm_v1.pth')
print("\n Training Complete. Model saved to 'badminton_lstm_v1.pth'")

print("\nGenerating Confusion Matrix...")
model.eval()
all_preds = []
all_true = []

with torch.no_grad():
    for batch_X, batch_y in test_loader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        outputs = model(batch_X)
        _, predicted = torch.max(outputs.data, 1)
        
        all_preds.extend(predicted.cpu().numpy())
        all_true.extend(batch_y.cpu().numpy())

stroke_names = data['classes']
cm = confusion_matrix(all_true, all_preds)

plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=stroke_names, 
            yticklabels=stroke_names)
plt.title('Badminton CNN-BiLSTM Confusion Matrix', fontsize=16)
plt.ylabel('True Stroke', fontsize=14)
plt.xlabel('Predicted Stroke', fontsize=14)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
print("\nDetailed Classification Report:")
print(classification_report(all_true, all_preds, target_names=stroke_names))