import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler

print("Loading original training data...")
data = np.load('5.2-features_tensor_doubled.npz', allow_pickle=True)
X_raw = data['X']

print("Reshaping and cleaning data...")
N_strokes, seq_len, num_features = X_raw.shape
X_flat = X_raw.reshape(-1, num_features)

X_flat = np.nan_to_num(X_flat, nan=0.0, posinf=0.0, neginf=0.0)

print("Fitting the StandardScaler...")
scaler = StandardScaler()
scaler.fit(X_flat)

joblib.dump(scaler, 'scaler.pkl')
print("Success! 'scaler.pkl' has been created and saved.")