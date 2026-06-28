# Window-based artifact rejection and time-window exclusion utilities
import numpy as np
import mne

def cont_ArtifactDetect(EEG, amplitudeThreshold=150, windowsize=2000, channels=None, stepsize=100, combineSegments=None):
    """
    Reject commonly recorded artifactual potentials (c.r.a.p.) in continuous EEG data.

    Parameters
    ----------
    EEG : mne.io.BaseRaw
        Continuous EEG dataset.
    amplitudeThreshold : float, optional
        Threshold for identifying artifacts (peak-to-peak amplitude in uV), by default 150.
    windowsize : float, optional
        Moving window width (in milliseconds), by default 2000.
    channels : list of int or None, optional
        Channels to check for artifacts. If None, check all channels, by default None.
    stepsize : float, optional
        Moving window step (in milliseconds), by default 100.
    combineSegments : float or None, optional
        Merge adjacent bad intervals closer than this duration (in milliseconds). If None, no merging, by default None.

    Returns
    -------
    WinRej : ndarray, shape (n_segments, 2)
        Marked segments in sample indices (onset, offset).
    """
    if channels is None:
        channels = np.arange(EEG.get_data().shape[0])
    
    # Check for EYE-EEG channels
    if hasattr(EEG, 'ch_names') and any('GAZE' in chan for chan in EEG.ch_names):
        print("EYE-Channels detected. It is not recommended to include these channels "
              "in the continuousArtifactDetect() function as the scale is usually very different. "
              "Please remove before using this function or make sure you correctly "
              "indicate channels to be considered.")
    
    amp_th = amplitudeThreshold  # in uV
    winms = windowsize  # in ms
    stepms = stepsize  # in ms 
    chanArray = channels  # which channels to check?
    shortisi = combineSegments  # merge adjacent bad intervals?

    WinRej, chanrej = basicrap(EEG, chanArray, amp_th, winms, stepms)

    if WinRej.size == 0:
        print('\n No data found exceeding the threshold. No rejection was performed.\n')
    else:
        if shortisi is not None:
            shortisisam = int(np.floor(shortisi * EEG.info['sfreq'] / 1000))  # to samples
            WinRej, chanrej = joinclosesegments(WinRej, chanrej, shortisisam)
        
        throw_out = np.array([1])
        while throw_out.size > 0:
            throw_out = np.array([], dtype=int)
            
            for i in range(WinRej.shape[0] - 1):
                if throw_out.size > 0 and throw_out[-1] == i:
                    continue
                
                if WinRej[i, 1] >= WinRej[i + 1, 0]:
                    throw_out = np.append(throw_out, i + 1)
                    WinRej[i, 0] = min(WinRej[i, 0], WinRej[i + 1, 0])
                    WinRej[i, 1] = max(WinRej[i, 1], WinRej[i + 1, 1])

            WinRej = np.delete(WinRej, throw_out, axis=0)

        nbad = np.sum(WinRej[:, 1] - WinRej[:, 0])
        print('\n {} segments were marked.'.format(WinRej.shape[0]))
        print('A total of {} samples ({:.02f} percent of data) was marked as bad.\n'.format(
            nbad, nbad / EEG.get_data().shape[1] * 100
        ))

    return WinRej

def basicrap(EEG, chanArray, ampth, winms, stepms):
    """
    Basic artifact rejection algorithm checking max peak-to-peak amplitude in moving windows.

    Parameters
    ----------
    EEG : mne.io.BaseRaw
        Continuous EEG dataset.
    chanArray : list of int
        Channels to check for artifacts.
    ampth : float
        Threshold for identifying artifacts (peak-to-peak amplitude in uV).
    winms : float
        Moving window width (in milliseconds).
    stepms : float
        Moving window step (in milliseconds).

    Returns
    -------
    WinRej : ndarray, shape (n_segments, 2)
        Onset and offset sample indices of detected artifact windows.
    chanrej : ndarray, shape (n_segments, n_channels)
        Binary matrix indicating which channels exceeded the threshold in each segment.
    """
    nchan = len(chanArray)
    srate = EEG.info['sfreq']  # Sampling rate in Hz

    WinRej = np.array([], dtype=int).reshape(0, 2)
    chanrej = np.array([], dtype=int).reshape(0, nchan)

    winpts = int(winms * srate / 1000)
    steppts = int(stepms * srate / 1000)
    data = EEG.get_data(picks=chanArray)

    for start in range(0, data.shape[1] - winpts + 1, steppts):
        stop = start + winpts
        datatmp = data[:, start:stop]
        maxamp = np.max(datatmp, axis=1) - np.min(datatmp, axis=1)
        exceeded = np.where(maxamp > ampth)[0]
        if exceeded.size > 0:
            chtmp = np.zeros(nchan)
            chtmp[exceeded] = 1
            WinRej = np.vstack((WinRej, [start, stop]))
            chanrej = np.vstack((chanrej, chtmp))

    return WinRej, chanrej

def joinclosesegments(WinRej, chanrej, shortisisam):
    """
    Merge adjacent marked segments that are closer than a threshold.

    Parameters
    ----------
    WinRej : ndarray, shape (n_segments, 2)
        Onset and offset sample indices of marked segments.
    chanrej : ndarray, shape (n_segments, n_channels)
        Binary matrix indicating channel rejections per segment.
    shortisisam : int
        Maximum gap duration in samples below which segments are merged.

    Returns
    -------
    WinRej2 : ndarray, shape (n_merged_segments, 2)
        Merged onset and offset sample indices.
    ChanRej2 : ndarray, shape (n_merged_segments, n_channels)
        Binary channel rejection matrix for merged segments.
    """
    if WinRej.shape[0] == 0:
        return WinRej, chanrej

    WinRej2 = []
    ChanRej2 = []
    
    print('\nWARNING: Marked segments that are closer than {} samples will be joined together.\n'.format(shortisisam))
    
    a = WinRej[0, 0]
    b = WinRej[0, 1]
    working = 0
    chrej2 = np.zeros(chanrej.shape[1])
    nwin = WinRej.shape[0]

    for j in range(1, nwin):
        isi = WinRej[j, 0] - WinRej[j-1, 1]
        if isi < shortisisam:
            b = WinRej[j, 1]
            chantmp = np.logical_or(chanrej[j, :], chanrej[j-1, :])
            chrej2 = np.logical_or(chrej2, chantmp)
            working = 1
            if j == nwin - 1:
                WinRej2.append([a, b])
                ChanRej2.append(chrej2)
        else:
            if working == 1:
                WinRej2.append([a, b])
                ChanRej2.append(chrej2)
                working = 0
            else:
                WinRej2.append([a, b])
                ChanRej2.append(chanrej[j-1, :])
            a = WinRej[j, 0]
            b = WinRej[j, 1]
            chrej2 = np.zeros(chanrej.shape[1])

    if working == 0:
        WinRej2.append([a, b])
        ChanRej2.append(chanrej[-1, :])

    ChanRej2 = [list(map(int, c)) for c in ChanRej2]
    
    return np.array(WinRej2), np.array(ChanRej2)
