import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.signal import convolve

# ---- Helper Function for delays ----
def _delays(start, stop, sfreq):
    """Create time delays array.
    
    Args:
        start (float): Start time in seconds
        stop (float): Stop time in seconds
        sfreq (float): Sampling frequency in Hz
        
    Returns:
        np.ndarray: Array of time points
    """
    return np.arange(start, stop, 1/sfreq)

# ---- Feature creation helper function ----
def create_feature_vector(events, key, value, n_times):
    """
    Create binary feature vector based on events.
    
    Args:
        events (dict or array): Events data structure
        key (str): Key to index in events
        value: Value to look for
        n_times (int): Length of the output vector
        
    Returns:
        np.ndarray: Binary feature vector of length n_times
    """
    # This is a placeholder - replace with your actual implementation
    feature_vec = np.zeros(n_times)
    for event in events:
        if event[key] == value:
            idx = event['sample']
            if 0 <= idx < n_times:
                feature_vec[idx] = 1
    return feature_vec

class MultiFeatureConvModel(nn.Module):
    """ERP estimation model using 1D convolutions.
    
    This model learns kernel shapes (impulse responses) for each feature
    and combines them to reconstruct the EEG signal.
    """
    def __init__(self, in_channels, kernel_size):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, 1, kernel_size, bias=False)
        
        # Initialize weights with small random values
        nn.init.normal_(self.conv.weight, std=0.01)
        
    def forward(self, x):
        return self.conv(x)
    
    def get_kernels(self):
        """Get the learned kernels (ERPs)."""
        return self.conv.weight.detach().cpu().squeeze().numpy()

def train_erp_model(raw, events, feature_definitions, time_window=(-0.2, 0.5), 
                    sfreq=500, num_epochs=3000, learning_rate=0.001, 
                    alpha=5, patience=10, channels=None, verbose=True):
    """
    Train the ERP estimation model.
    
    Args:
        raw: EEG data object with get_data() method (e.g., MNE Raw object)
        events: Events data structure
        feature_definitions: List of (key, value) tuples defining features to extract
        time_window: Tuple of (start_time, end_time) in seconds for the ERP window
        sfreq: Sampling frequency in Hz
        num_epochs: Maximum number of training epochs
        learning_rate: Learning rate for optimizer
        alpha: Ridge regularization strength
        patience: Early stopping patience
        channels: List of channel indices to use (None=all)
        verbose: Whether to print progress
        
    Returns:
        model: Trained model
        history: Training history
    """
    # ---- Setup ----
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if verbose:
        print(f"Using device: {device}")
    
    # Process time window parameters
    start_time, end_time = time_window
    kernel_size = len(_delays(start_time, end_time, sfreq))
    delay_len = len(_delays(start_time, 0, sfreq))
    pad = kernel_size - 1
    
    # Get EEG data
    n_times = raw.n_times
    if channels is None:
        y = raw.get_data()  # All channels
    else:
        y = raw.get_data()[channels]  # Selected channels
        
    n_channels = y.shape[0]
    n_samples = n_times  # For ridge normalization
    
    # ---- Create feature vectors ----
    feature_vecs = []
    for key, value in feature_definitions:
        feature_vec = np.roll(create_feature_vector(events, key, value, n_times), -delay_len)
        feature_vecs.append(feature_vec)
    
    X = np.stack(feature_vecs, axis=0)  # shape: (n_features, n_times)
    n_features = X.shape[0]
    
    # ---- Convert to torch and move to device ----
    X_tensor = torch.tensor(X, dtype=torch.float32).unsqueeze(0).to(device)  # (1, n_features, n_times)
    X_padded = nn.functional.pad(X_tensor, (pad, 0))  # (1, n_features, n_times + pad)
    
    # Train separate model for each channel
    all_models = []
    all_histories = []
    
    for ch in range(n_channels):
        if verbose:
            print(f"\nTraining model for channel {ch+1}/{n_channels}")
        
        # Set up channel-specific target
        target = y[ch]
        target_tensor = torch.tensor(target, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)  # (1, 1, n_times)
        
        # Set up model and optimizer
        model = MultiFeatureConvModel(in_channels=n_features, kernel_size=kernel_size).to(device)
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=5, verbose=verbose)
        
        # Loss function with Ridge penalty
        mse_loss = nn.MSELoss()
        
        def criterion(output, target):
            mse = mse_loss(output, target)
            ridge = (alpha / n_samples) * torch.sum(model.conv.weight ** 2)
            return mse + ridge
        
        # ---- Training Loop with Early Stopping ----
        loss_history = []
        best_loss = float('inf')
        best_model_state = None
        patience_counter = 0
        
        if verbose:
            print("Starting training...")
        start_time = time.time()
        
        for epoch in range(num_epochs):
            model.train()
            optimizer.zero_grad()
            
            output = model(X_padded)
            # Crop padding to match target
            output_cropped = output[..., :n_times]
            
            loss = criterion(output_cropped, target_tensor)
            loss.backward()
            optimizer.step()
            
            current_loss = loss.item()
            loss_history.append(current_loss)
            
            # Learning rate scheduler step
            scheduler.step(current_loss)
            
            # Early stopping check
            if current_loss < best_loss:
                best_loss = current_loss
                patience_counter = 0
                # Save best model state
                best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            else:
                patience_counter += 1
                
            if patience_counter >= patience:
                if verbose:
                    print(f"Early stopping at epoch {epoch+1}")
                break
                
            if verbose and (epoch + 1) % 100 == 0:
                print(f"Epoch {epoch+1}, Loss: {current_loss:.6f}")
        
        if verbose:
            print(f"Training completed in {time.time() - start_time:.2f} seconds")
        
        # Load best model
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
            model = model.to(device)
        
        all_models.append(model)
        all_histories.append(loss_history)
    
    return all_models, all_histories

def plot_erp_results(models, histories, raw, X, feature_names=None, time_window=(-0.2, 0.5), 
                     sfreq=500, channels=None, channel_names=None):
    """
    Plot ERP estimation results.
    
    Args:
        models: List of trained models (one per channel)
        histories: List of training histories
        raw: EEG data object
        X: Feature matrix (n_features, n_times)
        feature_names: List of feature names
        time_window: Tuple of (start_time, end_time) in seconds
        sfreq: Sampling frequency in Hz
        channels: List of channel indices plotted
        channel_names: List of channel names
    """
    if channels is None:
        y = raw.get_data()
        channels = list(range(y.shape[0]))
    else:
        y = raw.get_data()[channels]
    
    n_channels = len(channels)
    n_features = X.shape[0]
    n_times = X.shape[1]
    
    if feature_names is None:
        feature_names = [f"Feature {i+1}" for i in range(n_features)]
    
    if channel_names is None:
        channel_names = [f"Channel {ch+1}" for ch in range(n_channels)]
    
    # Time axis for ERPs
    start_time, end_time = time_window
    erp_times = np.linspace(start_time, end_time, len(_delays(start_time, end_time, sfreq)))
    
    # ---- Plot Loss Curves ----
    plt.figure(figsize=(10, 4))
    for ch in range(n_channels):
        plt.plot(histories[ch], label=channel_names[ch])
    plt.title("Loss Curves")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
    # ---- Plot Learned Kernels (ERPs) ----
    n_cols = min(3, n_channels)
    n_rows = (n_channels + n_cols - 1) // n_cols
    
    plt.figure(figsize=(15, n_rows * 3))
    for ch in range(n_channels):
        plt.subplot(n_rows, n_cols, ch + 1)
        learned_kernels = models[ch].get_kernels()
        
        for i in range(n_features):
            plt.plot(erp_times, learned_kernels[i], label=feature_names[i])
            
        plt.title(f"Estimated ERPs - {channel_names[ch]}")
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude")
        plt.legend()
        plt.grid(True)
        plt.axvline(x=0, color='k', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.show()
    
    # ---- Plot Model Output vs EEG ----
    device = next(models[0].parameters()).device
    
    # Create feature tensor
    X_tensor = torch.tensor(X, dtype=torch.float32).unsqueeze(0).to(device)
    kernel_size = models[0].conv.weight.shape[-1]
    pad = kernel_size - 1
    X_padded = nn.functional.pad(X_tensor, (pad, 0))
    
    plt.figure(figsize=(15, n_rows * 3))
    for ch in range(n_channels):
        plt.subplot(n_rows, n_cols, ch + 1)
        
        with torch.no_grad():
            predicted = models[ch](X_padded).squeeze().cpu().numpy()
            # Remove padding
            predicted = predicted[:n_times]
        
        plt.plot(y[ch], label="EEG Signal", alpha=0.7)
        plt.plot(predicted, label="Reconstructed", color='red')
        plt.title(f"EEG vs Model Output - {channel_names[ch]}")
        plt.legend()
        plt.grid(True)
    
    plt.tight_layout()
    plt.show()

# ---- Example Usage ----
if __name__ == "__main__":
    # This is just example code - would require actual raw data and events
    from pydeconv.utils import *
    from pydeconv.pydeconv_sims import *
    import numpy as np
    import mne

    # %matplotlib qt 
    n_seconds = 3000        # Duration of the signal in seconds
    sfreq = 500            # Sampling frequency in Hz
    sig = EEGSimulator(n_seconds, sfreq)
    # transition probabilities
    W_matrix = [[0, 0.45, 0.45, 0.1],
                [0.9, 0, 0 , .1],
                [0.9, 0, 0,.1],
                [.33,.33,.33,0]]
    kernels = {
                0: {'onsets': [0, 0.19, 0.25], 'amplitudes': [0.1, -0.05, 0.04], 'widths': [0.05, 0.05, 0.07], 'weight':0.3},
                1: {'onsets': [0, 0.19, 0.25], 'amplitudes': [0.1, -0.05, 0.04], 'widths': [0.05, 0.05, 0.07], 'weight':0.3},
                2: {'onsets': [0, 0.19, 0.25], 'amplitudes': [0.1, -0.07, 0.04], 'widths': [0.05, 0.05, 0.07], 'weight':0.3},
                3: {'onsets': [0, 0.19, 0.25], 'amplitudes': [0.1, -0.07, 0.04], 'widths': [0.05, 0.05, 0.07], 'weight':0.3},
                'modulation': {'ker_idx2mod': 1, 'mod': 'linear','dist': 'uniform', 'lims': [100, 600]}
            }

    sig.create_isi_pdf(0, sample_size=100, lims=[.01, .15], dist_type='skewed', mode=.05, skew=0, scale=.01)
    sig.create_isi_pdf(1, sample_size=100, lims=[.1, .6], dist_type='skewed', mode=.3, skew=2, scale=.05)
    sig.create_isi_pdf(2, sample_size=100, lims=[.1, .6], dist_type='skewed', mode=.3, skew=2, scale=.05)
    sig.create_isi_pdf(3, sample_size=100, lims=[.1, .6], dist_type='uniform')


    # sig.combine_isi_pdf
    # sig.plot_isi_pdf(0)
    # sig.plot_isi_pdf(1)

    ################
    sig.simulate(noise="brown",erp_ker=kernels,w_matrix=W_matrix)

        # create evts

    # Copy and modify the event data
    evts = sig.evts.copy()

    # Set the event type to filter (event_id 1 for example)
    event_id1 = 1
    event_id2 = 2

    # Filter events where `type == event_id`
    filtered_evts = evts.loc[(evts['type'] == event_id1) | (evts['type'] == event_id2)]

    # Get the number of filtered events
    n_events = len(filtered_evts)

    # Ensure that latencies are integer values
    latencies = filtered_evts['latency'].values.astype(int)

    # Create the events array for MNE
    # Column 1: Latencies
    # Column 2: Zeros (assuming no previous event values, hence zeros)
    # Column 3: Event types (all set to 1 since filtered for `event_id`)
    mne_events = np.column_stack((latencies,
                                np.zeros(n_events, dtype=int),
                                np.ones(n_events, dtype=int)))

    # Print or use `mne_events` as needed
    print(mne_events[:5])
    #create RAW
    # Creating simulated RAW

    # Parameters

    n_samples = n_seconds * sfreq  # Total number of samples
    n_channels = 1         # Number of channels (virtual channel)

    # Create random data for the virtual channel (shape: [n_samples])
    data = sig.data

    # Reshape the data to be 2D (n_channels, n_samples)
    data = data.reshape((n_channels, n_samples))

    # Create MNE info object
    ch_names = ['VirtualEEG']  # Name of the virtual channel
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types='eeg')

    # Create the Raw object from the reshaped data array
    raw = mne.io.RawArray(data, info)

    evts['type'].value_counts()

    # Update 'effect' column
    evts['effect'] = evts['type'].apply(lambda x: True if x == 2 else False if x == 1 else np.nan)

    # Replace all 2s with 1s in 'type' column
    evts['type'] = evts['type'].replace(2, 1)
    evts['type'].value_counts()
    columns = {'latencies':latencies,	'type':'type','categorical':'categorical','continuous':'continuous'}
    # evts.rename(columns=columns, inplace=True)


    # Example feature definitions (key, value pairs)
    feature_definitions = [
        ('type', 0),  # e.g., stimulus type 0
        ('type', 1),  # e.g., stimulus type 1
        ('effect', True)  # e.g., some effect present
    ]
    
    feature_names = ['Stimulus Type 0', 'Stimulus Type 1', 'Effect Present']
    
    # Train model on first 3 channels
    channels = [0, 1, 2]
    
    models, histories = train_erp_model(
        raw=raw,  # Replace with your actual data
        events=evts,  # Replace with your actual events
        feature_definitions=feature_definitions,
        time_window=(-0.2, 0.5),
        sfreq=500,
        num_epochs=3000,
        learning_rate=0.001,
        alpha=5,
        channels=channels
    )
    
    # Get feature matrix for plotting
    X = np.stack([
        create_feature_vector(evts, 'type', 0, raw.n_times),
        create_feature_vector(evts, 'type', 1, raw.n_times),
        create_feature_vector(evts, 'effect', True, raw.n_times)
    ], axis=0)
    
    # Plot results
    plot_erp_results(
        models=models,
        histories=histories,
        raw=raw,
        X=X,
        feature_names=feature_names,
        time_window=(-0.2, 0.5),
        sfreq=500,
        channels=channels
    )