# Design matrix creation and B-spline feature expansion utilities
import numpy as np
# pyrefly: ignore [missing-import]
import torch
from typing import Sequence, Optional   

def shifted_matrix(
    features: np.ndarray,
    delays: Sequence[int],
    use_gpu: bool = True,
    indices_to_keep: Optional[Sequence[int]] = None,
    output_torch: bool = False,
    train_indexes: np.ndarray = None,
    pred_indexes: np.ndarray = None,
    ) -> np.ndarray:
    """
    Build a time-shifted design matrix for given features and delays.

    This function stacks time-shifted versions of the input feature matrix along the second axis,
    optionally computing only for specified row indices to reduce memory.

    Parameters
    ----------
    features : np.ndarray, shape (n_times, n_features) or (n_times,)
        Input time series data. If 1D, it is treated as a single feature.
    delays : Sequence[int]
        Relative time shifts (in samples). Positive delays shift past values,
        negative delays shift future values, zero retains current.
    use_gpu : bool, default True
        Whether to attempt computation on CUDA device first. Falls back to CPU on OOM.
    indices_to_keep : Sequence[int], optional
        Specific time indices at which to compute rows of the shifted matrix.
        If None, computes all rows.
    output_torch : bool or float, default False
        If True, returns a PyTorch tensor instead of a NumPy array. 
        If False, returns a NumPy array.
    train_indexes : np.ndarray, optional
        Indices of training samples. If provided, only these indices are used for computation.
    pred_indexes : np.ndarray, optional
        Indices of prediction samples. If provided, only these indices are used for computation.

    Returns
    -------
    np.ndarray, shape (n_rows, n_features * n_delays)
        Design matrix where each row contains concatenated features for each delay.
    """
    # Determine device order: try GPU first, then CPU
    preferred = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")
    devices = [preferred]
    if preferred.type == "cuda":
        devices.append(torch.device("cpu"))

    # Ensure features is 2D
    feats = features.reshape(-1, 1) if features.ndim == 1 else features

    for dev in devices:
        try:
            # Move data onto device
            if not isinstance(feats, torch.Tensor):
                feats_t = torch.tensor(feats, dtype=torch.float32, device=dev)
            else:
                feats_t = feats.to(dtype=torch.float32, device=dev)
                        
            shifted = _compute_shifted(feats_t, delays, indices_to_keep)
            
            # Reshape: (n_rows, n_delays, n_features) -> (n_rows, n_features * n_delays)
            n_rows, n_delays, n_feat = shifted.shape
            mat = shifted.permute(0, 2, 1).reshape(n_rows, n_feat * n_delays)
            if train_indexes is not None and pred_indexes is not None:
                if output_torch:
                    return mat[train_indexes, :], mat[pred_indexes, :]
                else: 
                    return mat[train_indexes, :].cpu().numpy(), mat[pred_indexes, :].cpu().numpy()
            else:
                if output_torch:
                    return mat
                else: 
                    return mat.cpu().numpy()

        except RuntimeError as e:
            if dev.type == "cuda":
                print(f"CUDA OOM on device {dev}; retrying on CPU. Error: {e}")
                continue
            else:
                raise
    # If loop completes without return, something went wrong
    raise RuntimeError("shifted_matrix failed on all devices")


def _compute_shifted(
    feats_t: torch.Tensor,
    delays: Sequence[int],
    indices_to_keep: Optional[Sequence[int]]
) -> torch.Tensor:
    """
    Compute shifted matrix for given features and delays.

    Parameters
    ----------
    feats_t : torch.Tensor
        Input features tensor of shape (n_samples, n_features).
    delays : Sequence[int]
        Delays to apply to the features.
    indices_to_keep : Optional[Sequence[int]]
        Specific indices to compute the shifted matrix for.

    Returns
    -------
    torch.Tensor
        Shifted matrix of shape (n_rows, n_delays, n_features).
    """
    n_samples, n_features = feats_t.shape
    delays = torch.tensor(delays, device=feats_t.device, dtype=torch.int64)

    if indices_to_keep is not None:
        idx = torch.tensor(indices_to_keep, device=feats_t.device, dtype=torch.int64)
        idx_shifted = idx[:, None] - delays[None, :]  # Shape: (n_rows, n_delays)
    else:
        idx_shifted = torch.arange(n_samples, device=feats_t.device)[:, None] - delays[None, :]  # Shape: (n_samples, n_delays)

    # Mask for valid indices
    valid_mask = (idx_shifted >= 0) & (idx_shifted < n_samples) # Shape: (n_rows, n_delays)

    # Clamp indices to valid range (i.e: ensure values are between 0 and n_samples-1)
    idx_clipped = idx_shifted.clamp(0, n_samples - 1)
    
    # Gather features and apply the mask
    feats_exp = feats_t[idx_clipped]  # Shape: (n_rows, n_delays, n_features)
    
    # Broadcast mask to match feature dimensions (unsqueeze to add feature dimension)
    feats_exp *= valid_mask.unsqueeze(-1)  # Shape: (n_rows, n_delays, n_features)

    return feats_exp


def ensure_contiguous_and_finite(X: np.ndarray, name: Optional[str] = None) -> np.ndarray:
    """Ensure NumPy array is C-contiguous and free of NaN/Inf values.

    Parameters
    ----------
    X : np.ndarray
        The array (e.g., design matrix) to check and sanitize.
    name : str, optional
        An optional identifier (e.g., subject code) for context in print messages.

    Returns
    -------
    X_clean : np.ndarray
        A C-contiguous array with NaN/Inf values replaced by 0.0.
    """
    ctx = f" for {name}" if name else ""
    if not X.flags["C_CONTIGUOUS"]:
        print(f"[INFO] Forcing C-contiguity{ctx}")
        X = np.ascontiguousarray(X)
    if not np.isfinite(X).all():
        print(f"[INFO] Replacing NaN/Inf values in design matrix{ctx}")
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    return X


def kept_idxs(feature_tensor, tmin, tmax, sampling_rate, axis=0):
    """
    Get the indices of the interesting samples to estimate the kernel in a vectorized way.
    Uses a difference-array approach to find which indices fall into the window defined
    around each event from tmin to tmax (in seconds).

    Parameters:
    ----------
    feature_tensor : torch.Tensor
        A 1D tensor representing the feature (e.g., a binary marker for events).
    tmin : float
        Start of the kernel window relative to event time (in seconds).
    tmax : float
        End of the kernel window relative to event time (in seconds).
    sampling_rate : float
        Sampling rate in Hz (samples per second).
    axis : int
        The axis along which to find the indices. Default is 0.

    Returns:
    -------
    kept_indices : list
        Sorted list of unique indices that fall in the union of all event windows.
    """
    feature_np = feature_tensor.cpu().numpy()
    event_idxs = np.nonzero(feature_np)[axis]
    if len(event_idxs) == 0:
        return []

    L = feature_tensor.shape[axis]

    # Convert tmin and tmax to sample offsets
    start_offset = int(np.floor(tmin * sampling_rate))
    end_offset = int(np.ceil(tmax * sampling_rate))

    diff = np.zeros(L + 1, dtype=np.int32)

    # Start and end of each window in sample indices
    start_idxs = event_idxs + start_offset
    end_idxs = event_idxs + end_offset

    # Clip indices to valid range
    start_idxs = np.clip(start_idxs, 0, L)
    end_idxs = np.clip(end_idxs, 0, L)

    # Apply difference array method
    np.add.at(diff, start_idxs, 1)
    np.add.at(diff, end_idxs, -1)

    cumsum = np.cumsum(diff[:-1])
    kept_indices = np.nonzero(cumsum > 0)[0]

    return kept_indices.tolist()