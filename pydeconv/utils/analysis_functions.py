import mne
import numpy as np
import sys
import os
from ..utils.functions_general import *
import matplotlib.pyplot as plt
# from utils.paths import paths
import os
import time
#import save
# from ..utils.load import *
# from utils.setup import *
#DAC code

#Joaco code
def define_events(evts_df,
                  event_type,
                  present=None,
                  correct=None,    
                  mss=None,
                  trials=None,
                  phase=None,
                  dur=None,
                  item_type=None,
                  dir=None,
                  rank=None
                ):

    print('Defining events to epoch NOTE: trials with incorrect searchimage size were balcklisted!!!')

    metadata_sup = None
    metadata     = evts_df.copy()
    if 'fixation' in event_type:
        metadata = metadata.loc[(metadata.bad == 0) & (metadata.inrange == True) ]
    
    if correct:
        metadata = metadata.loc[(metadata.correct == correct)]
    if present:
        metadata = metadata.loc[(metadata.present == present)]
    if phase:
        metadata = metadata.loc[(metadata.phase == phase)]
    if type(mss)==int:
        metadata = metadata.loc[(metadata.mss == mss)]
    if type(mss)==str:
        msss=mss.split('_')
        metadata = metadata.loc[(metadata.mss == int(msss[0])) | (metadata.mss == int(msss[1]))]

    if type(dur)==int:
        metadata = metadata.loc[(metadata.duration >= dur)]
    if 'fixation' in event_type:
        if item_type == 'ontarget':
            metadata = metadata.loc[(metadata['ontarget'] == True)]
        elif item_type == 'ondistractor':
            metadata = metadata.loc[(metadata['ondistractor'] == True)]
        elif item_type == 'onstm':
            metadata = metadata.loc[~(metadata['stm'].isna())]
    if 'saccade' in event_type:
        if dir:
            metadata = metadata.loc[(metadata['dir'] == dir)] #NEEDS modification to be use
        else:
            metadata = metadata.loc[(metadata['type'] == 'saccade')] #NEEDS modification to be use
    if type(rank)==int:
        metadata = metadata.loc[(metadata['rank'] >= rank)]
    if type(rank)==str:
        rankst = rank.split('_')
        if rankst[0] == 'more':
            metadata = metadata.loc[(metadata['rank'] >= int(rankst[1]))]
        elif rankst[0] == 'less':
            metadata = metadata.loc[(metadata['rank'] <= int(rankst[1]))]

    metadata.reset_index(drop=True, inplace=True)

    events_samples = metadata['latency'] - 1 #latency column comes from (UBA preprocess in EEGLAB 2022 and 2023 NOTT proly too)!!!!!!

    events = np.zeros((len(events_samples), 3)).astype(int)
    events[:, 0] = events_samples
    events[:, 2] = 1

    events_id = dict(event_type=1)
    print('events defined')
    

    return metadata, events, events_id, metadata_sup

def balcklisted_trials(suj,metadata):
    from utils.paths import paths
    import pandas as pd
    path = paths()
    exp_path = path.experiment_path()

    item_pos = path.item_pos_path()
    bh_data = suj.load_bh_csv()
    evts = suj.load_event_struct()
    image_names = bh_data['searchimage'].drop_duplicates()
    image_names = image_names.str.split('cmp_', expand=True)[1]
    image_names = image_names.str.split('.jpg', expand=True)[0]
    image_names = list(image_names)
    df = pd.read_csv(item_pos)
    tr=0
    metadata_tmp = metadata.copy()
    for image_name in image_names[:-1]:
        tr+=1
        scale_x = 1
        scale_y = 1
        img = plt.imread(exp_path + 'cmp_' + image_name + '.jpg')    
        if (img.shape[0] != 1024) or (img.shape[1] !=1280):      
            # Calculate scaling factors
            metadata_tmp = metadata_tmp.loc[metadata_tmp.trial!=tr]

    return metadata_tmp


def epoch_data(subject,
                event_type,
                present=None,
                correct=None,    
                mss=None,
                trials=None,
                phase=None,
                dur=None,
                item_type=None,
                dir=None,
                rank=None,
                tmin=-.2,
                tmax=.3,
                baseline=(None, 0),
                reject=None,
                save_data=False,
                epochs_save_path=None,
                epochs_data_fname=None,
                reduced_head=None):
    '''
    :param subject:
    :param mss:
    :param corr_ans:
    :param tgt_pres:
    :param epoch_id:
    :param meg_data:
    :param tmin:
    :param tmax:
    :param baseline: tuple
    Baseline start and end times.
    :param reject: float|str|bool
    Peak to peak amplituyde reject parameter. Use 'subject' for subjects default calculated for short fixation epochs.
     Use False for no rejection. Default to 4e-12 for magnetometers.
    :param save_data:
    :param epochs_save_path:
    :param epochs_data_fname:
    :return:
    '''

    eeg = subject.load_analysis_eeg()
    eeg = subject.load_electrode_positions(eeg)
    evts_df = subject.load_metadata()
    if subject.experiment == 'UBA':
        chns = 128
    elif subject.experiment == 'UON':
        chns = 64
    # Sanity check to save data
    if reduced_head is not None:
        eeg =  subject.load_analysis_eeg(reduced=True)
        eeg =  eeg.set_montage('biosemi64') #not for UBA 128
        chns = 64

    if save_data and (not epochs_save_path or not epochs_data_fname):
        raise ValueError('Please provide path and filename to save data. Else, set save_data to false.')


    # Define events
    _, ev, _, _ =   define_events(evts_df,
                                                                event_type,
                                                                present,
                                                                correct,    
                                                                mss,
                                                                trials,
                                                                phase,
                                                                dur,
                                                                item_type,
                                                                dir,
                                                                rank)
    
    
    # Reject based on channel amplitude
    if reject == None:
        # Not setting reject parameter will set to default subject value
        reject = dict(eeg=200e-6)# Check this value!!!!!!!!!!!!24/8/23 DAC
    elif reject == False:
        # Setting reject parameter to False uses No rejection (None in mne will not reject)
        reject = None
    elif reject == 'subject':
        reject = dict(eeg=250e-6)
    elif isinstance(reject,float):
        reject_dic = dict(eeg= reject)

    # Epoch data
    epochs = mne.Epochs(raw=eeg, events=ev, event_id=None, tmin=tmin, tmax=tmax, reject=reject_dic,
                        event_repeated='drop', metadata=None, preload=True, baseline=baseline,picks=range(chns))
    # Drop bad epochs
    #print(ev)
    epochs.drop_bad()
    #print(eeg)

    # if metadata_sup is not None:
    #     metadata_sup = metadata_sup.loc[(metadata_sup['id'].isin(epochs.metadata['event_name']))].reset_index(drop=True)
    #     epochs.metadata = metadata_sup

    if save_data:
        # Save epoched data
        epochs.reset_drop_log_selection()
        os.makedirs(epochs_save_path, exist_ok=True)
        epochs.save(epochs_save_path + epochs_data_fname, overwrite=True)

    return epochs, ev


def time_frequency(epochs, l_freq, h_freq, freqs_type, n_cycles_div=4., return_itc=True, save_data=False, trf_save_path=None,
                   power_data_fname=None, itc_data_fname=None):

    # Sanity check to save data
    if save_data and (not trf_save_path or not power_data_fname or not itc_data_fname):
        raise ValueError('Please provide path and filename to save data. Else, set save_data to false.')

    # Compute power over frequencies
    print('Computing power and ITC')
    if freqs_type == 'log':
        freqs = np.logspace(*np.log10([l_freq, h_freq]), num=40)
    elif freqs_type == 'lin':
        freqs = np.linspace(l_freq, h_freq, num=h_freq - l_freq + 1)  # 1 Hz bands
    n_cycles = freqs / n_cycles_div  # different number of cycle per frequency
    power, itc = mne.time_frequency.tfr_morlet(epochs, freqs=freqs, n_cycles=n_cycles, use_fft=True,
                                               return_itc=return_itc, decim=3, n_jobs=None, verbose=True)

    if save_data:
        # Save trf data
        os.makedirs(trf_save_path, exist_ok=True)
        power.save(trf_save_path + power_data_fname, overwrite=True)
        itc.save(trf_save_path + itc_data_fname, overwrite=True)

    return power, itc


def get_plot_tf(tfr, plot_xlim=(None, None), plot_max=True, plot_min=True):
    if plot_xlim:
        tfr_crop = tfr.copy().crop(tmin=plot_xlim[0], tmax=plot_xlim[1])
    else:
        tfr_crop = tfr.copy()

    timefreqs = []

    if plot_max:
        max_ravel = tfr_crop.data.mean(0).argmax()
        freq_idx = int(max_ravel / len(tfr_crop.times))
        time_percent = max_ravel / len(tfr_crop.times) - freq_idx
        time_idx = round(time_percent * len(tfr_crop.times))
        max_timefreq = (tfr_crop.times[time_idx], tfr_crop.freqs[freq_idx])
        timefreqs.append(max_timefreq)

    if plot_min:
        min_ravel = tfr_crop.data.mean(0).argmin()
        freq_idx = int(min_ravel / len(tfr_crop.times))
        time_percent = min_ravel / len(tfr_crop.times) - freq_idx
        time_idx = round(time_percent * len(tfr_crop.times))
        min_timefreq = (tfr_crop.times[time_idx], tfr_crop.freqs[freq_idx])
        timefreqs.append(min_timefreq)

    timefreqs.sort()

    return timefreqs


def head_reducer(raw):
    #load standard montages and check if raw has 128 chs or more
    if len(raw.pick(['eeg']).ch_names) < 128:
        print('eeg data has less than 128 channels')
        return
    bio128=mne.channels.make_standard_montage("biosemi128")
    bio64=mne.channels.make_standard_montage("biosemi64")
    #functions to determine the matching channels
    def calculate_distance(arr1, arr2):
        # Assuming arr1 and arr2 are numpy arrays
        return np.linalg.norm(arr1 - arr2)

    def find_close_channels(dict1, dict2, threshold):
        close_channel_pairs = []
        close_dict ={}
        for key1, value1 in dict1.items():
            # Check if the corresponding key exists in the second OrderedDict
            for key2 ,value2 in dict2.items():
                distance = calculate_distance(value1, value2)

                if distance < threshold:
                    close_channel_pairs.append((key1, key2))
                    close_dict[key1]=key2

        return close_channel_pairs , close_dict
    #from montages get an orderDict with positions and check matching channels within thr
    threshold = 1e-10
    dict64 = bio64.get_positions()['ch_pos']
    dict128 = bio128.get_positions()['ch_pos']

    close_channel_pairs ,close_dict= find_close_channels(dict64, dict128, threshold)
    print(f"{len(close_channel_pairs)} Pairs of keys closer than {threshold}:")
    #channels to interpolate from 64 cap
    cap64_chs_to_add = [ch for ch in bio64.ch_names if ch not in close_dict.keys()]
    #channels to remove after interpolation
    cap158_chs_to_remove = [ch for ch in bio128.ch_names if ch not in close_dict.values()]
    #create custom montage
    dict158 = dict128
    #add channels from 64 montage that don't match any of the 128 location
    for new_chs in cap64_chs_to_add:
        dict158[new_chs] = dict64[new_chs]
    #these are the fiducials for both biosemi montages
    nas = bio64.get_positions()['nasion']
    lpa = bio64.get_positions()['lpa']
    rpa = bio64.get_positions()['rpa']

    custom_montage = mne.channels.make_dig_montage(ch_pos=dict158,coord_frame='head',nasion=nas,lpa=lpa,rpa=rpa)
    custom_montage.plot()

    #add new channels
    sfreq = raw.info['sfreq']
    n_samples = raw.n_times

    # Create new channels with all zero values (30 is now hardcoded but it can be easyly change to be expected nchan-matching_chs_number)
    new_channel_data = np.zeros((30, n_samples))

    # Create an info structure for the new channels
    new_channel_info = mne.create_info(ch_names=cap64_chs_to_add, sfreq=sfreq, ch_types=['eeg']*30)

    # Create a RawArray object with the new channel data
    new_channel_raw = mne.io.RawArray(new_channel_data, new_channel_info)

    # Add the new channel to the Raw object
    raw.add_channels([new_channel_raw], force_update_info=True)
    tointer = raw.copy().set_montage(custom_montage, on_missing='ignore')
    # check if fail tointer.info['dig']
    tointer.info['bads'] = cap64_chs_to_add
    #[eeg.info.ch_names.append(bad) for bad in cap64_chs_to_add]
    tointer = tointer.pick_channels(custom_montage.ch_names)
    raw_interpolated = tointer.interpolate_bads()
    #remove unwanted channels
    raw_interpolated.drop_channels(cap158_chs_to_remove)
    #change names of the remainig channels
    dict_close = {key: val for val , key in close_dict.items()}
    mne.rename_channels(raw_interpolated.info, dict_close)
    #reorder channels to the biosemi original order
    raw_interpolated = raw_interpolated.reorder_channels(bio64.ch_names)


    return raw_interpolated


def saccades_intercept_evts_prev(evts,fixations_intercept_metadata):
    #this function selects, based in the fixation intercepts, the previous saccades
    #to be consider as a second intercept in the model
    evts_fix_n_sac = evts.loc[(evts.type=='fixation') | (evts.type=='saccade')]
    evts_fix_n_sac.reset_index(drop=True, inplace=True)

    # Find the indices of df1 corresponding to the 'latency' values in df2
    indices_to_select = evts_fix_n_sac[evts_fix_n_sac['latency'].isin(fixations_intercept_metadata['latency'])].index - 1 #atempt to correct collinearity original version looked at the prev. sacc.!!!

    # Filter df1 based on the calculated indices
    intercept_saccades_metadata = evts_fix_n_sac.loc[indices_to_select]
    
    events_samples = intercept_saccades_metadata['latency'] - 1 #latency column comes from (UBA preprocess in EEGLAB 2022 and 2023 NOTT proly too)!!!!!!
    events = np.zeros((len(events_samples), 3)).astype(int)
    events[:, 0] = events_samples
    events[:, 2] = 1

    return intercept_saccades_metadata , events 

def saccades_intercept_evts_next(evts,fixations_intercept_metadata):
    #this function selects, based in the fixation intercepts, the next saccades 
    #to be consider as a second intercept in the model
    evts_fix_n_sac = evts.loc[(evts.type=='fixation') | (evts.type=='saccade')]
    evts_fix_n_sac.reset_index(drop=True, inplace=True)

    # Find the indices of df1 corresponding to the 'latency' values in df2
    indices_to_select = evts_fix_n_sac[evts_fix_n_sac['latency'].isin(fixations_intercept_metadata['latency'])].index + 1 #atempt to correct collinearity original version looked at the prev. sacc.!!!

    # Filter df1 based on the calculated indices
    intercept_saccades_metadata = evts_fix_n_sac.loc[indices_to_select]
    
    events_samples = intercept_saccades_metadata['latency'] - 1 #latency column comes from (UBA preprocess in EEGLAB 2022 and 2023 NOTT proly too)!!!!!!
    events = np.zeros((len(events_samples), 3)).astype(int)
    events[:, 0] = events_samples
    events[:, 2] = 1

    return intercept_saccades_metadata , events

def saccades_intercept_evts_nearest(evts,fixations_intercept_metadata):
    #this function selects, based in the fixation intercepts, the next saccades 
    #to be consider as a second intercept in the model
    evts_fix_n_sac = evts.loc[(evts.type=='fixation') | (evts.type=='saccade')]
    evts_fix_n_sac.reset_index(drop=True, inplace=True)

    # Find the indices of df1 corresponding to the 'latency' values in df2
    indices_to_select_1 = evts_fix_n_sac[evts_fix_n_sac['latency'].isin(fixations_intercept_metadata['latency'])].index - 1 #atempt to correct collinearity original version looked at the prev. sacc.!!!
    indices_to_select_2 = evts_fix_n_sac[evts_fix_n_sac['latency'].isin(fixations_intercept_metadata['latency'])].index + 1 #atempt to correct collinearity original version looked at the prev. sacc.!!!

    selected_indexes = np.union1d(indices_to_select_1, indices_to_select_2)
    # Filter df1 based on the calculated indices
    intercept_saccades_metadata = evts_fix_n_sac.loc[selected_indexes]
    
    events_samples = intercept_saccades_metadata['latency'] - 1 #latency column comes from (UBA preprocess in EEGLAB 2022 and 2023 NOTT proly too)!!!!!!
    events = np.zeros((len(events_samples), 3)).astype(int)
    events[:, 0] = events_samples
    events[:, 2] = 1

    return intercept_saccades_metadata , events
 
def add_active_to_evts(evts,thr_dur=0):
    evts_tmp = evts
    evts_tmp['active_search'] = False
    phases_to_active = ['vs']
    #phase_to_active can be only vs 

    
    if 'inrange' not in evts_tmp.columns:
        raise ValueError("Column 'inrange' does not exist in the DataFrame. Run the corresponding function to add it.")

    for phase in phases_to_active:
        for tr in range(1,210):
            evts_fix_vs_pres = evts_tmp[(evts_tmp['type']=='fixation') & (evts_tmp['phase']== phase) & (evts_tmp['inrange'] == True) 
                                        & (evts_tmp['trial']== tr) & (evts_tmp['present']== True)]
            if len(evts_fix_vs_pres.loc[(evts_fix_vs_pres.ontarget) & (evts_fix_vs_pres.duration >=thr_dur)])>0:
                active_tmp = ~(evts_fix_vs_pres['ontarget'] & (evts_fix_vs_pres['duration'] >= thr_dur)).cumsum().astype(bool)  
            
                evts_tmp.loc[active_tmp.index, 'active_search'] = active_tmp
                

    #NOW DO IT WITH MEM
    return evts_tmp

if __name__ == '__main__':
    import utils.setup as setup
    import utils.load as load
    info = setup.exp_info()
    suj  = load.subject(info,3)
    evts = suj.load_metadata()
    _, ev, _, _ =     define_events(evts,
                                    event_type='fixation',
                                    present=True,
                                    correct=None,    
                                    mss=4,
                                    trials=None,
                                    phase='vs',
                                    dur=150,
                                    item_type='ontarget',
                                    dir=None,
                                    rank=None)
    print(ev.shape)
