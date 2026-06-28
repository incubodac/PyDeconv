import numpy as np
import pytest
from pydeconv.utils.window_rejection import basicrap, cont_ArtifactDetect, joinclosesegments

class MockEEGArray:
    """Mock EEG class mimicking minimal MNE Raw structure needed for window rejection."""
    def __init__(self, data, sfreq, ch_names=None):
        self.data = data
        self.info = {'sfreq': sfreq}
        if ch_names is not None:
            self.ch_names = ch_names
        else:
            self.ch_names = [f'EEG {i:03d}' for i in range(data.shape[0])]

    def get_data(self, picks=None):
        if picks is None:
            picks = range(self.data.shape[0])
        return self.data[picks, :]

def test_window_rejection_synthetic():
    # Define parameters for synthetic data
    sr = 500              # sample rate in Hz
    duration = 10         # duration in seconds
    n_samples = sr * duration
    n_channels = 3

    # Create base random noise
    data = np.random.normal(0, 0.1, (n_channels, n_samples))

    # Inject large amplitude artifacts (peaks) at specific times
    # Artifact 1: around 2.0s (sample 1000) on Channel 0 & 1
    data[0, 1000] = 5.0
    data[1, 1000] = 5.0
    
    # Artifact 2: around 6.0s (sample 3000) on Channel 2
    data[2, 3000] = 5.0

    # Initialize the mock EEG object
    eeg = MockEEGArray(data, sr)

    # Parameters for artifact detection
    channels = range(n_channels)
    amplitude_threshold = 2.0
    window_ms = 2000
    step_ms = 1000

    # 1. Test basicrap
    win_rej, chan_rej = basicrap(eeg, channels, amplitude_threshold, window_ms, step_ms)
    
    # Check that basicrap detected artifacts
    assert win_rej.size > 0
    # Artifact at 1000 (2.0s) should be detected in windows overlapping sample 1000
    artifact_1_detected = any(start <= 1000 < stop for start, stop in win_rej)
    artifact_2_detected = any(start <= 3000 < stop for start, stop in win_rej)
    assert artifact_1_detected
    assert artifact_2_detected

    # 2. Test cont_ArtifactDetect
    win_rej_cont = cont_ArtifactDetect(
        eeg,
        amplitudeThreshold=amplitude_threshold,
        windowsize=window_ms,
        channels=channels,
        stepsize=step_ms,
        combineSegments=1000
    )
    
    assert win_rej_cont.size > 0
    # Check that segments are successfully marked
    assert win_rej_cont.shape[0] >= 1

def test_window_rejection_no_artifacts():
    # Create flat/clean data with zero artifacts
    sr = 250
    data = np.zeros((2, 1000))
    eeg = MockEEGArray(data, sr)

    # With a high threshold, no segments should be returned
    win_rej = cont_ArtifactDetect(
        eeg,
        amplitudeThreshold=5.0,
        windowsize=1000,
        channels=[0, 1],
        stepsize=500
    )
    assert len(win_rej) == 0

def test_join_close_segments():
    # Test joining logic of close windows
    # Segments: [100, 200] and [250, 350]. Gap is 50 samples.
    WinRej = np.array([[100, 200], [250, 350]])
    chanrej = np.array([[1, 0], [0, 1]])

    # Case 1: Merge threshold (100 samples) > gap (50 samples). Should be merged.
    merged_win, merged_chan = joinclosesegments(WinRej, chanrej, shortisisam=100)
    assert len(merged_win) == 1
    assert np.array_equal(merged_win[0], [100, 350])
    # The merged channels should be logical OR of the active channels
    assert np.array_equal(merged_chan[0], [1, 1])

    # Case 2: Merge threshold (30 samples) < gap (50 samples). Should NOT be merged.
    separate_win, separate_chan = joinclosesegments(WinRej, chanrej, shortisisam=30)
    assert len(separate_win) == 2
    assert np.array_equal(separate_win[0], [100, 200])
    assert np.array_equal(separate_win[1], [250, 350])

def test_eye_gaze_warning(capsys):
    # Create mock EEG array with a GAZE channel
    sr = 250
    data = np.zeros((2, 1000))
    eeg = MockEEGArray(data, sr, ch_names=['EEG 001', 'GAZE_X'])

    # Run detection
    cont_ArtifactDetect(eeg, amplitudeThreshold=1.0, windowsize=100, channels=[0, 1])
    
    # Check that warning message was printed to stdout
    captured = capsys.readouterr()
    assert "EYE-Channels detected" in captured.out
