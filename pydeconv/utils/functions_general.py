import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def scale_from_interval(signal_to_scale, reference_signal, interval_signal=None, interval_ref=None):
    """
    Scale signal based on matching signal with different scale.

    Parameters
    ----------
    signal_to_scale: ndarray
        The 1D signal you wish to re-scale.
    reference_signal: ndarray
        The 1D reference signal with propper scaling.
    interval_signal: {'list', 'tuple'}, optional
        The signal interval to use for scaling. if None, the whole signal is used. Default to None.
    interval_ref: {'list', 'tuple'}, optional
        The reference signal interval to use for scaling. if None, the whole signal is used. Default to None.

    Returns
    -------
    signal_to_scale: ndarray
      The re-scaled signal.
    """

    signal_to_scale -= np.nanmin(signal_to_scale[interval_signal[0]:interval_signal[1]])
    signal_to_scale /= np.nanmax(signal_to_scale[interval_signal[0]:interval_signal[1]])
    signal_to_scale *= (
            np.nanmax(reference_signal[interval_ref[0]:interval_ref[1]])
            - np.nanmin(reference_signal[interval_ref[0]:interval_ref[1]]))
    signal_to_scale += np.nanmin(reference_signal[interval_ref[0]:interval_ref[1]])

    return signal_to_scale


def remove_missing(x, y, time, missing):
    mx = np.array(x == missing, dtype=int)
    my = np.array(y == missing, dtype=int)
    x = x[(mx + my) != 2]
    y = y[(mx + my) != 2]
    time = time[(mx + my) != 2]
    return x, y, time


def find_nearest(array, values):
    """
    Find the nearest element in an array to a given value

    Parameters
    ----------
    array: ndarray
        The 1D array to look in for the nearest value.
    values: int, float, list or 1D array
        If int or float, use that value to find neares elemen in array and return index and element as int and array.dtype
        If list or array, iterate over values and return arrays of indexes and elements nearest to each value.

    Returns
    -------
    idx: int
      The index of the element in the array that is nearest to the given value.
    element: int float
        The nearest element to the specified value
    """
    array = np.asarray(array)

    if isinstance(values, float) or isinstance(values, int):
        idx = (np.abs(array - values)).argmin()
        return idx, array[idx]

    elif len(values):
        idxs = []
        elements = []
        for value in values:
            idx = (np.abs(array - value)).argmin()
            idxs.append(idx)
            elements.append(array[idx])
        return np.asarray(idxs), np.asarray(elements)


def find_previous(array, value):
    """
    Find the nearest element in an array to a given value

    Parameters
    ----------
    array: ndarray
        The 1D array to look in for the nearest value.

    Returns
    -------
    idx: int
      The index of the element in the array that is nearest to the given value.
    """

    array = np.asarray(array)
    idx = np.max(np.where(array - value <= 0)[0])
    return idx, array[idx]



def find_first_within(array, low_bound, up_bound):
    """
    Find the first element from an array in a certain interval

    Parameters
    ----------
    array: ndarray
        The 1D array to look in for the nearest value.
    low_bound: float
        the lower boundary of the search interval
    up_bound: float
        the upper boundary of the search interval

    Returns
    -------
    idx: int
      The index of the element in the array that is nearest to the given value.
    value: float
      The value of the array in the found index.
    """

    array = np.asarray(array)
    elements = np.where(np.logical_and((array > low_bound), (array < up_bound)))[0]
    try:
        idx = np.min(elements)
        return idx, array[idx]
    except:
        return False, False


def find_last_within(array, low_bound, up_bound):
    """
    Find the first element from an array in a certain interval

    Parameters
    ----------
    array: ndarray
        The 1D array to look in for the nearest value.
    low_bound: float
        the lower boundary of the search interval
    up_bound: float
        the upper boundary of the search interval

    Returns
    -------
    idx: int
      The index of the element in the array that is nearest to the given value.
    value: float
      The value of the array in the found index.
    """

    array = np.asarray(array)
    elements = np.where(np.logical_and((array > low_bound), (array < up_bound)))[0]
    try:
        idx = np.max(elements)
        return idx, array[idx]
    except:
        return False, False


def first_trial(evt_buttons):
    """
    Get event corresponding to last green button press before 1st trial begins

    Parameters
    ----------
    evt_buttons: ndarray
        The 1D array containing the responses from the MEG data.

    Returns
    -------
    first_trial: int
      The number of trial corresponding to the last green button press before the 1st trial begins.
    """

    for i, button in enumerate(evt_buttons):
        if button != 'green':
            return i


def flatten_list(ls):
    flat_list = [element for sublist in ls for element in sublist]
    return flat_list


def align_signals(signal_1, signal_2):
    '''
    Find samples shift that aligns two matching signals by Pearson correlation.

    Parameters
    ----------
    signal_1: ndarray
        1D array signal. Must be longer than signal_2, if not, the samples shift will be referenced to signal_2.

    signal_2: ndarray
        1D array signal.

    Returns
    -------
    max_sample: int
      Sample shift of maximum correlation between signals.
    '''

    invert = False

    start_samples = len(signal_1) - len(signal_2)
    # invert signals to return samples shift referenced to the longer signal
    if start_samples < 0:
        print('Signal_2 is longer. Inverting reference.')
        save_signal = signal_1
        signal_1 = signal_2
        signal_2 = save_signal
        invert = True

    corrs = []
    for i in range(start_samples):
        print("\rProgress: {}%".format(int((i + 1) * 100 / start_samples)), end='')
        df = pd.DataFrame({'x1': signal_1[i:i + len(signal_2)], 'x2': signal_2})
        corrs.append(df.corr()['x1']['x2'])
        # if df.corr()['x1']['x2'] > 0.5 and all(np.diff(corrs[-50:]) < 0): # Original parameters
        if any(np.array(corrs) > 0.9) and all(np.diff(corrs[-100:]) < 0):
            print(f'\nMaximal correlation sample shift found in sample {i}')
            break
    max_sample = np.argmax(corrs)
    print(f'Maximum correlation of {np.max(corrs)}')

    if invert:
        max_sample = -max_sample

    return max_sample, corrs


def ch_name_map(orig_ch_name):
    if orig_ch_name[-5:] == '-4123':
        new_ch_name = orig_ch_name[:-5]
    else:
        new_ch_name = orig_ch_name
    return new_ch_name


def pick_chs(chs_id, info):
    '''
    :param chs_id: 'mag'/'LR'/'parietal/occipital/'frontal'/sac_chs/parietal+'
        String identifying the channels to pick.
    :param info: class attribute
        info attribute from the evoked data.
    :return: picks: list
        List of chosen channel names.
    '''

    if chs_id == 'mag':
        picks = 'mag'
    elif chs_id == 'sac_chs':
        picks = ['MLF14', 'MLF13', 'MLF12', 'MLF11', 'MRF11', 'MRF12', 'MRF13', 'MRF14', 'MZF01']
    elif chs_id == 'LR':
        right_chs = ['MRT51', 'MRT52', 'MRT53']
        left_chs = ['MLT51', 'MLT52', 'MLT53']
        picks = right_chs + left_chs

    else:
        ids = chs_id.split('_')
        all_chs = info.ch_names
        picks = []
        for id in ids:
            if id == 'parietal':
                picks += [ch_name for ch_name in all_chs if 'M' in ch_name and 'P' in ch_name]
            elif id == 'parietal+':
                picks += [ch_name for ch_name in all_chs if 'M' in ch_name and 'P' in ch_name]
                picks += ['MLT25', 'MLT26', 'MLT27', 'MLO24', 'MLO23', 'MLO22', 'MLO21', 'MLT15', 'MLT16',
                                 'MLO14', 'MLO13', 'MLO12', 'MLO11',
                                 'MZO01',
                                 'MRT25', 'MRT26', 'MRT27', 'MRO24', 'MRO23', 'MRO22', 'MRO21', 'MRT15', 'MRT16',
                                 'MRO14', 'MRO13', 'MRO12', 'MRO11']
            elif id == 'occipital':
                picks += [ch_name for ch_name in all_chs if 'M' in ch_name and 'O' in ch_name]
            elif id == 'frontal':
                picks += [ch_name for ch_name in all_chs if 'M' in ch_name and 'F' in ch_name]
            elif id == 'temporal':
                picks += [ch_name for ch_name in all_chs if 'M' in ch_name and 'T' in ch_name]
            elif id == 'central':
                picks += [ch_name for ch_name in all_chs if 'M' in ch_name and 'C' in ch_name]

            # Subset from picked chanels
            elif id == 'L':
                picks = [ch_name for ch_name in picks if 'M' in ch_name and 'L' in ch_name]
            elif id == 'R':
                picks = [ch_name for ch_name in picks if 'M' in ch_name and 'R' in ch_name]

    return picks


def get_freq_band(band_id):
    '''
    :param band_id: str ('Delta/Theta/Alpha/Beta/Gamma
        String determining the frequency bands to get.

    :return: l_freq: int
        Lower edge of frequency band.
    :return: h_freq: int
        High edge of frequency band.
    '''
    if type(band_id) == str:

        # Get multiple frequency bands
        bands = band_id.split('_')
        l_freqs = []
        h_freqs = []

        for band in bands:
            if band == 'Delta':
                l_freq = 1
                h_freq = 4
            elif band == 'Theta':
                l_freq = 4
                h_freq = 8
            elif band == 'Alpha':
                l_freq = 8
                h_freq = 12
            elif band == 'Beta':
                l_freq = 12
                h_freq = 30
            elif band == 'Gamma':
                l_freq = 30
                h_freq = 45
            elif band == 'HGamma':
                l_freq = 45
                h_freq = 100
            elif band == 'Broad':
                l_freq = 0.5
                h_freq = 100

            l_freqs.append(l_freq)
            h_freqs.append(h_freq)

        l_freq = np.min(l_freqs)
        h_freq = np.max(h_freqs)

    elif type(band_id) == tuple:
        l_freq = band_id[0]
        h_freq = band_id[1]

    elif band_id == None:

        l_freq = None
        h_freq = None

    return l_freq, h_freq


def get_time_lims(epoch_id, map=None):
    '''

    :param epoch_id: str
        String with the name of the epochs to select.
    :param map: dict
        Dictionary of dictionaries indicating the times associated to each type of epoch id.
        Keys should be 'fix', 'sac', and within those keys, a dictionary with keys 'tmin', 'tmax', 'plot_xlim' with their corresponding values.

    :return: tmin: float
        time corresponding to time start of the epochs.
    :return: tmax: float
        time corresponding to time end of the epochs.
    :return: plot_xlim: tuple of float
        time start and end to plot.

    '''
    done = False
    if map:
        for key in map.keys():
            if key in epoch_id:
                tmin = map[key]['tmin']
                tmax = map[key]['tmax']
                plot_xlim = map[key]['plot_xlim']
                done = True
    if not done:
        print('Using default time values')
        if 'fix' in epoch_id:
            tmin = -0.1
            tmax = 0.2
            plot_xlim = (tmin, tmax)
        elif 'sac' in epoch_id:
            tmin = -0.05
            tmax = 0.1
            plot_xlim = (tmin, tmax)
        else:
            tmin = -0.1
            tmax = 0.1
            plot_xlim = (-0.05, 0.1)

    return tmin, tmax, plot_xlim


def get_item(epoch_id):

    if 'tgt' in epoch_id:  # 1 for target, 0 for item, None for none
        tgt = 1
    elif 'it' in epoch_id:
        tgt = 0
    else:
        tgt = None

    return tgt


def get_dir(epoch_id):

    if '_sac' in epoch_id:
        dir = epoch_id.split('_sac')[0]
    else:
        dir = None

    return dir


def get_screen(epoch_id):

    screens = ['emap', 'cross1', 'ms', 'vs', 'cross2']
    screen = epoch_id.split('_')[-1]

    if screen not in screens:
        screen = None

    return screen


def get_mss(epoch_id):

    if 'mss' in epoch_id:
        mss = epoch_id.split('mss')[-1][0]
    else:
        mss = None

    return mss


def get_condition_trials(subject, mss=None, corr_ans=None, tgt_pres=None):
    bh_data = subject.bh_data
    if corr_ans:
        bh_data = bh_data.loc[subject.corr_ans == 1]
    elif corr_ans == False:
        bh_data = bh_data.loc[subject.corr_ans == 0]
    if mss:
        bh_data = bh_data.loc[bh_data['Nstim'] == mss]
    if tgt_pres:
        bh_data = bh_data.loc[bh_data['Tpres'] == 1]
    elif tgt_pres == False:
        bh_data = bh_data.loc[bh_data['Tpres'] == 0]

    trials = list(bh_data.index + 1)  # +1 for 0th index

    return trials, bh_data