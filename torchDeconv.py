import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import time
import pandas as pd
import mne
from pydeconv.utils import analyze_data

# Print initial GPU memory usage
if torch.cuda.is_available():
    print(f"Initial Allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
    print(f"Initial Reserved: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")

# ============================ Helper Functions ============================

def create_feature_vector(evts_df, column_to_filter, value, max_latency=None, cond=None):
    filtered_events = evts_df[evts_df[column_to_filter] == value]
    if cond is not None:
        cond_col, cond_val = cond
        filtered_events = filtered_events[filtered_events[cond_col] == cond_val]
    if max_latency is None:
        max_latency = int(evts_df['latency'].max()) + 1
    feature_vector = np.zeros(max_latency)
    for lat in filtered_events['latency'].values:
        lat_idx = int(lat)
        if 0 <= lat_idx < max_latency:
            feature_vector[lat_idx] = 1
    return feature_vector

def _delays(tmin, tmax, sfreq):
    return np.arange(int(np.round(tmin * sfreq)), int(np.round(tmax * sfreq)) + 1)

# Safer normalization function
def safe_normalize(data, axis=0, eps=1e-8):
    mean = np.mean(data, axis=axis, keepdims=True)
    std = np.std(data, axis=axis, keepdims=True)
    # Add small epsilon to prevent division by zero
    std = np.maximum(std, eps)
    return (data - mean) / std

# ============================ Load Data ============================

data_path = "./example_data/"
settings = analyze_data()
features = pd.read_csv(data_path + "629959_full_metadata.csv") 
raw = mne.io.read_raw_eeglab(data_path + "629959_analysis.set", preload=True)

n_channels = 20
n_times = raw.n_times
sfreq = 500

y = raw.get_data()[:n_channels, :]
# Use safer normalization
y = safe_normalize(y, axis=1)

print(f"Target data shape: {y.shape}")  # Debug: print target shape



# ============================ Feature Construction ============================

# Create delays and define kernel parameters
kernel_size = len(_delays(-.2, .5, sfreq))
pad = kernel_size // 2
delay_offset = len(_delays(-.2, 0, sfreq))

# Load features from CSV and create feature vectors
features = pd.read_csv(data_path + "629959_full_metadata.csv")
n_times = raw.n_times

inter = create_feature_vector(features, 'type', 'fixation', n_times)
effect = create_feature_vector(features, 'type', 'fixation', n_times, ('ontarget', True))
saccade = create_feature_vector(features, 'type', 'saccade', n_times)

feature_vec1 = np.roll(inter, -delay_offset + pad)
feature_vec2 = np.roll(saccade, -delay_offset + pad)
feature_vec3 = np.roll(effect, -delay_offset + pad)

# Stack three feature vectors into a (3, T) array
X = np.stack([feature_vec1, feature_vec2, feature_vec3], axis=0)
X = (X - np.mean(X, axis=1, keepdims=True)) / np.std(X, axis=1, keepdims=True)

print('X shape:', X.shape)  # (3, T)

# ============================ Torch Setup ============================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.cuda.empty_cache()

# Create feature tensor: expand to (64, 3, T)
X_tensor = torch.tensor(X, dtype=torch.float32).unsqueeze(0)   # (1, 3, T)
X_tensor = X_tensor.expand(n_channels, -1, -1).contiguous()      # (64, 3, T)
X_padded = nn.functional.pad(X_tensor, (pad, pad))               # (64, 3, T + 2*pad)

# Move to GPU now (after processing on CPU to save memory)
X_padded = X_padded.to(device)

# Create target tensor with shape (64, 1, T)
target_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(1)  # (64, 1, T)
target_tensor = target_tensor.to(device)

# ============================ Model Definition ============================

class GroupedConvModel(nn.Module):
    def __init__(self, n_channels, in_features, kernel_size):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels=n_channels * in_features,  # 64*3 = 192
            out_channels=n_channels,               # one output per channel
            kernel_size=kernel_size,
            groups=n_channels                      # each channel’s features convolved separately
        )

    def forward(self, x):
        # x is expected to be (n_channels, in_features, T)
        B, F, T = x.shape  # B=n_channels, F=in_features
        x = x.reshape(1, B * F, T)  # (1, 192, T)
        out = self.conv(x)         # (1, 64, T_out)
        return out.squeeze(0)      # (64, T_out)

model = GroupedConvModel(n_channels=n_channels, in_features=3, kernel_size=kernel_size).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
# You can use Huber loss or MSE loss; here we use MSE for simplicity.
criterion = nn.MSELoss()

# ============================ Training Loop ============================

num_epochs = 10  # Using fewer epochs for testing
loss_history = []
print("Training...")
start_time = time.time()

model.train()
for epoch in range(num_epochs):
    optimizer.zero_grad()
    output = model(X_padded)  # (64, T_out)
    # Crop output if necessary to match target (assumes symmetric padding)
    output_cropped = output[..., pad:pad + target_tensor.shape[-1]]
    loss = criterion(output_cropped, target_tensor)
    loss.backward()
    optimizer.step()
    loss_history.append(loss.item())
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.6f}")

print(f"Training completed in {time.time() - start_time:.2f} seconds")

# ============================ Plotting Kernels ============================

with torch.no_grad():
    kernels = model.conv.weight.view(n_channels, 3, -1).cpu().numpy()

plt.figure(figsize=(12, 6))
delays = _delays(-.2, .5, sfreq) / sfreq  # scale delays to seconds
for ch in range(3):
    plt.plot(delays, kernels[0, ch], label=f"Channel 0 - Feature {ch+1}")
plt.title("Learned Kernels for Channel 0")
plt.xlabel("Time (s)")
plt.legend()
plt.grid(True)
plt.show()
