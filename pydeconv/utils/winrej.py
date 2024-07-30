import numpy as np
# import utils.load as load
import os
import mne
import matplotlib.pyplot as plt
# import utils.setup as setup
# from utils.paths import paths
from ..utils.plot_general import *
#import functions_general
#joaco code
# exp_info = setup.exp_info()
############################-----MAKe compatible with mne e.g. pick types----------

def cont_ArtifactDetect(EEG, amplitudeThreshold=150, windowsize=2000, channels=None, stepsize=100, combineSegments=None):
    """
    Reject commonly recorded artifactual potentials (c.r.a.p.) in continuous EEG data.

    Args:
        EEG (ndarray): Continuous EEG dataset.
        amplitudeThreshold (int): Threshold for identifying artifacts (values).
        windowsize (int): Moving window width (in msec).
        channels (list): Channels to check for artifacts.
        stepsize (int): Moving window step (in msec).
        combineSegments (int): Merge adjacent bad intervals.

    Returns:
        WinRej (ndarray): Marked segments.
    """
    
    if channels is None:
        channels = np.arange(EEG.get_data().shape[0])
    
    # Check for EYE-EEG channels
    if  hasattr(EEG, 'ch_names') and any('GAZE' in chan for chan in EEG.ch_names):
        print("EYE-Channels detected. It is not recommended to include these channels\
        in the continuousArtifactDetect() function as the scale is usually very different.\
        Please remove before using this function or make sure you correctly\
        indicate channels to be consider")
    
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
        WinRej

        nbad = np.sum(WinRej[:, 1] - WinRej[:, 0])
        print('\n {} segments were marked.'.format(WinRej.shape[0]))
        print('A total of {} samples ({:.02f} percent of data) was marked as bad.\n'.format(nbad, nbad / EEG.get_data().shape[1] * 100))

    return WinRej

def basicrap(EEG, chanArray, ampth, winms, stepms):
    """
    Basic artifact rejection algorithm.

    Args:
        EEG (ndarray): Continuous EEG dataset.
        chanArray (list): Channels to check for artifacts.
        ampth (int): Threshold for identifying artifacts (values).
        winms (int): Moving window width (in ms).
        stepms (int): Moving window step (in ms).

    Returns:
        WinRej (ndarray): Marked segments.
        chanrej (ndarray): Channel rejections.
    """
    nchan = len(chanArray)
    srate = EEG.info['sfreq']  # Sampling rate in Hz

    WinRej = np.array([], dtype=int).reshape(0, 2)
    chanrej = np.array([],dtype=int).reshape(0,nchan)

    winpts = int(winms * srate / 1000)
    steppts = int(stepms * srate / 1000)
    data = EEG.get_data(picks=chanArray)

    for start in range(0, data.shape[1] - winpts + 1, steppts):
        stop = start + winpts
        datatmp = data[list(chanArray), start:stop]
        #print(list(chanArray),start,stop,data.shape)
        maxamp = np.max(datatmp, axis=1) - np.min(datatmp, axis=1)
        exceeded = np.where(maxamp > ampth)[0]
        #print(exceeded,exceeded.shape)
        if exceeded.size > 0:
            # if len(exceeded) == 1:
            #     chlist = chanArray[int(exceeded)]
            #     chanrej = chlist
            # else:
                chtmp = np.zeros(nchan)
                chtmp[exceeded] = 1
                WinRej = np.vstack((WinRej, [start, stop]))
                chanrej = np.vstack((chanrej,chtmp))

    return WinRej, chanrej

def joinclosesegments(WinRej, chanrej, shortisisam):
    WinRej2 = []   # Create an empty list to store new windows
    ChanRej2 = []  # Create an empty list to store new channel rejections
    
    print('\nWARNING: Marked segments that are closer than {} samples will be joined together.\n'.format(shortisisam))
    # Print a warning message informing the user about the operation being performed.
    
    a = WinRej[0, 0]  # Initialize 'a' with the start time of the first window
    b = WinRej[0, 1]  # Initialize 'b' with the end time of the first window
    m = 0  # Initialize 'm' as 0
    working = 0  # Initialize 'working' as 0
    chrej2 = np.zeros(chanrej.shape[1])  # Create an array of zeros with the same number of columns as chanrej
    nwin = WinRej.shape[0]  # Get the number of rows in WinRej (number of windows)

    # Loop through the windows in WinRej
    for j in range(1, nwin):
        # Calculate the inter-segment interval (isi) between the current and previous windows
        isi = WinRej[j, 0] - WinRej[j-1, 1]
        if isi < shortisisam:
            # If the isi is smaller than the specified threshold (shortisisam)
            b = WinRej[j, 1]  # Update 'b' with the end time of the current window
            #chrej2 |= (chanrej[j, :] | chanrej[j-1, :])  # Perform a bitwise OR operation on channel rejections
            chantmp = np.logical_or(chanrej[j, :] , chanrej[j-1, :])
            chrej2  = np.logical_or(chrej2,chantmp)
            working = 1  # Set 'working' to 1, indicating that we're working on a segment
            if j == nwin - 1:
                WinRej2.append([a, b])  # Append the current start and end times to WinRej2
                ChanRej2.append(chrej2)  # Append the channel rejections to ChanRej2
        else:
            if working == 1:
                WinRej2.append([a, b])  # Append the start and end times to WinRej2
                ChanRej2.append(chrej2)  # Append the channel rejections to ChanRej2
                working = 0  # Reset 'working' to 0
            else:
                WinRej2.append([a, b])  # Append the start and end times to WinRej2
                ChanRej2.append(chanrej[j-1, :])  # Append the channel rejections from the previous window
            a = WinRej[j, 0]  # Update 'a' with the start time of the current window
            b = WinRej[j, 1]  # Update 'b' with the end time of the current window
            chrej2 = np.zeros(chanrej.shape[1])  # Reset chrej2 to an array of zeros
            m += 1  # Increment 'm'

    ChanRej2 = [list(map(int, c)) for c in ChanRej2]  # Convert the channel rejections to a list of integers
    
    # Return the results as NumPy arrays
    return np.array(WinRej2), np.array(ChanRej2)



# Example usage:
# WinRej = uf_continuousArtifactDetect(EEG)
if __name__=='__main__':
    class EEGArray:
        def __init__(self, data):
            self.data = data
            self.info = {}  # Initialize an empty dictionary for attributes
        def get_data(self,picks=None):
            if picks==None:
                picks=range(self.data.shape[0])
            return self.data[picks,:]
        def shape(self):
            return self.data.shape
        

    for i in range(0,6,5):
        print(i)

    sr= 500
    max_dur = 500
    max_samples = round(sr*max_dur)
    peak_at = 200

    array = np.zeros((3, max_samples))
    array[0, round(200*sr)] = 1.6
    array[2, round(300*sr)] = 1.6
    array[1, round(200*sr)] = 1.6
    array[2, round(304*sr)] = 1.6

    data       =  np.random.rand(3, max_samples)+ array
    data[0,:]  = data[0,:] -3 
    data[1,:] = data[1,:] *1.1

    times = np.linspace(0,max_dur,max_samples)
  
    EEG = EEGArray(data)
    EEG.info = {'sfreq':sr}  # Replace 1000 with your desired sample rate
    chanArray = range(3)
    ampth = 1.3
    winms = 2000
    stepms = 1000
    pepe , pipi=basicrap(EEG, chanArray, ampth, winms, stepms)
    print('\n',pepe)
    print('this is pipi\n',pipi)
    print(pipi.shape)
    fig, ax = plt.subplots()
    for i in range(3):
        plt.plot(times,data.T[:,i]+i*4)

    # Add shaded regions
    # for row in pepe:
    #     start, stop = row
    #     plt.axvspan(times[start], times[stop], alpha=0.2, color='gray', label='Shaded Region')

    # Set labels and legend
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.legend()

    # Show the plot
    # plt.show()
    print(pepe[1])
    pipo = cont_ArtifactDetect(EEG, amplitudeThreshold=ampth, windowsize=winms, channels=range(3), stepsize=stepms, combineSegments=1000)
    print(pipo)
    for row in pipo:
        start, stop = row
        plt.axvspan(times[start], times[stop], alpha=0.2, color='gray', label='Shaded Region')
    plt.show()
