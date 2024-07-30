import logging
import os
import numpy as np


def start_stop_samples_trigg(evts,trigg):
    trigg_samples = evts[evts['type']==trigg]['latency'].to_numpy().astype(int)
    dur_samples   = evts[evts['type']==trigg]['duration'].to_numpy().astype(int)
    
    return trigg_samples, trigg_samples + dur_samples


def start_samples_trigg(evts,trigg):
    
    if type(trigg)==str:
        trigg_samples = evts[evts['type']==trigg]['latency'].to_numpy().astype(int)
    elif type(trigg)==list:
        ev   = evts.loc[evts['type'].isin(trigg)]
        trigg_samples = ev['latency'].to_numpy().astype(int)
    
    return trigg_samples

def expand_evts_struct(evts):
    
    
    return evts

def closest_tuple(tuples, threshold, point):
    import numpy as np
    from scipy.spatial.distance import cdist
    distances = cdist(np.array(tuples), np.array([point]))
    within_threshold = distances < threshold
    if np.any(within_threshold):
        closest_idx = np.argmin(distances[within_threshold])
        closest_tuple_idx = np.where(within_threshold)[0][closest_idx]
        return True, closest_tuple_idx
    else:
        return False, None



def add_trial_info_to_events(evts,bh_data,thr):
    #so far inrage column is only refering to vs phase and presented image at that experimental phase
    from utils.paths import paths
    import utils.setup as setup
    import matplotlib.image as mpimg
    import pandas as pd


    threshold = thr
    image_names = bh_data['searchimage'].drop_duplicates().str.split('cmp_', expand=True)[1].str.split('.jpg', expand=True)[0].to_list()
    path =    paths()
    exp_path = path.experiment_path()
    targets  = bh_data.loc[::6,['st5']] #first column has image name and second T/A(absent)
    target_files = targets['st5'].str.lstrip('memstim').str.lstrip('/')[:-1] #target filenames
    exp_path = path.experiment_path()
    info = setup.exp_info()
    screensize = info.screen_size#[ 1920,1080 ]
    item_pos = path.item_pos_path()
    df = pd.read_csv(item_pos)

        
    
    import numpy as np
    def key2bool(val):
        if val == 'right':
            return True
        elif val == 'left':
            return False
        else:
            return val

    #modify evts dataframe 
    column = ['phase','istarget','isdistractor','']
    phases = {'cross1','mem','cross2','vs','bad_ET'}
    emvs   = {'fixation','saccade'}
    tr=0
    evts['trial']          = np.nan
    evts['phase']          = np.nan
    evts['mss']            = np.nan
    evts['ontarget']       = np.nan
    evts['ondistractor']   = np.nan
    evts['present']        = np.nan
    evts['correct']        = np.nan
    evts['stm']            = np.nan
    evts['inrange']        = np.nan

    msss     = list(bh_data.loc[::6,'Nstim'])
    T_and_stim = bh_data.loc[5::6,['st5_cat','st5','key_resp.corr']].dropna( how='all')
    T_and_stim.loc[T_and_stim.st5=='memstim/dog1962.png']=T_and_stim.loc[T_and_stim.st5=='memstim/dog1962.png'].replace({0:1,1:0}) #mod 24 ovt DAC to flip dof1962 correct value
    presents= list((T_and_stim.loc[:,'st5_cat'] =='T' ) & ~(T_and_stim.loc[:,'st5']=='memstim/dog1962.png'))    
    pressed = bh_data.loc[5::6,'key_resp.keys']
    #corrects = list(presents[:-1]== pressed.map(key2bool)) #changed 24 oct DAC to work with UON It should work as well with UBA
    corrects  = list(T_and_stim.loc[:,'key_resp.corr'].dropna( how='all')==1)

    #define start and stop latencies for phase events then loop over events and label them
    cross1_start_samp, cross1_stop_samp = start_stop_samples_trigg(evts,'cross1')
    mem_start_samp, mem_stop_samp       = start_stop_samples_trigg(evts,'mem')
    cross2_start_samp, cross2_stop_samp = start_stop_samples_trigg(evts,'cross2')
    vs_start_samp, vs_stop_samp         = start_stop_samples_trigg(evts,'vs')

    for index, row  in evts.iterrows():
        if evts.at[index,'type']=='cross1':
            tr+=1
            image_name   = image_names[tr-1]
            img = mpimg.imread(exp_path + 'cmp_' + image_name + '.jpg')
            xim = (screensize[0]-img.shape[1])/2
            yim = (screensize[1]-img.shape[0])/2
            trial_stims  = df[df['folder']==image_name]
            ##### external bounding for inrange ####
            height, width, _ = img.shape
            ext_width = width + 2 * threshold
            ext_height = height + 2 * threshold
            x_bounds = [-threshold,-threshold+ext_width]
            y_bounds = [-threshold,-threshold+ext_height]
            ########################################
            records = trial_stims.to_records(index=False)
            item_pos = [(record[6]+record[5]/2, record[7]+record[4]/2) for record in records]
            if presents[tr-1]:
                target_pos = trial_stims[trial_stims['stm']==target_files.iloc[tr-1]][['height','width','pos_x','pos_y']].to_records(index=False)
                target_pos = target_pos[0]
                target_pos = target_pos[2]+target_pos[1]/2,target_pos[3]+target_pos[0]/2
            else:
                target_pos = None    
            try:
                if presents[tr-1]:
                    if not target_pos:
                        raise ValueError("ima_pos is empty, but flag is True")
                else:
                    if target_pos:
                        raise ValueError("ima_pos is not empty, but flag is False")
            except ValueError as e:
                print(f"Sanity check failed: {str(e)}")
                
        elif evts.at[index,'type'] in emvs:
            evts.at[index,'trial'] = tr
            evts.at[index,'mss']   = msss[tr-1]
            evts.at[index,'present']   = presents[tr-1]
            evts.at[index,'correct']   = corrects[tr-1]
            
            if cross1_start_samp[tr-1] < evts.at[index,'latency'] < cross1_stop_samp[tr-1]:
                evts.at[index,'phase'] = 'cross1'
            elif mem_start_samp[tr-1] < evts.at[index,'latency'] < mem_stop_samp[tr-1]:
                evts.at[index,'phase'] = 'mem'
            elif cross2_start_samp[tr-1] < evts.at[index,'latency'] < cross2_stop_samp[tr-1]:
                evts.at[index,'phase'] = 'cross2'
            elif vs_start_samp[tr-1] < evts.at[index,'latency'] < vs_stop_samp[tr-1]:
                evts.at[index,'phase'] = 'vs'
                point = (evts.at[index,'fix_avgpos_x']-xim,evts.at[index,'fix_avgpos_y']-yim)
                
                if (evts.at[index,'type']=='fixation') and (x_bounds[0] < point[0] < x_bounds[1]) and  (y_bounds[0] < point[1] < y_bounds[1]):
                    evts.at[index,'inrange']        = True
                elif (evts.at[index,'type']=='fixation') and not ((x_bounds[0] < point[0] < x_bounds[1]) and  (y_bounds[0] < point[1] < y_bounds[1])):
                    evts.at[index,'inrange']        = False
            
                flag, closest_id = closest_tuple(item_pos, threshold, point)
                if not flag:
                    continue
                elif item_pos[closest_id]==target_pos:
                    evts.at[index,'ontarget']      = True
                    evts.at[index,'ondistractor']  = False
                    evts.at[index,'stm']           = trial_stims['stm'].iloc[closest_id]
                else:
                    evts.at[index,'ontarget']      = False
                    evts.at[index,'ondistractor']  = True
                    evts.at[index,'stm']           = trial_stims['stm'].iloc[closest_id]
                


    
    cross1_counts = len(evts[(evts['type']=='fixation') & (evts['phase']=='cross1')])
    mem_counts    = len(evts[(evts['type']=='fixation') & (evts['phase']=='mem')])
    cross2_counts = len(evts[(evts['type']=='fixation') & (evts['phase']=='cross2')])
    vs_counts     = len(evts[(evts['type']=='fixation') & (evts['phase']=='vs')])
    mean_dur_vs   = evts[(evts['type']=='fixation') & (evts['phase']=='vs')]['duration'].mean()
    answer_acc = 100*sum(corrects)/len(corrects)
    
    print(f'percentage of correct answers : {answer_acc:.1f}')
    print(f'fixations in cross1 phase : {cross1_counts}\n')
    print(f'fixations in mem phase    : {mem_counts}\n')
    print(f'fixations in cross2 phase : {cross2_counts}\n')
    print(f'fixations in vs phase     : {vs_counts}\n')
    print(evts['type'].value_counts())
    total_captured_fixs = sum((evts['ondistractor']) | (evts['ontarget']))     #taking into account inrange is still missing
    total_item_fixed = 100*total_captured_fixs/vs_counts
    on_targets  = sum((evts['ontarget']==True))
    on_distractors = sum((evts['ondistractor']==True))
    

      
    print(f'total fixations on items    : {total_captured_fixs}')
    print(f'fixations on targets  : {on_targets}')
    print(f'fixations on distractors  : {on_distractors}') 
    print(f'percentage of capture fixations in vs {total_item_fixed:.1f}%')

    logger = logging.getLogger()
    logger.info("Percentage of correct answers: %.1f %%", answer_acc)
    logger.info("Cross1: %d", cross1_counts)
    logger.info("Mem: %d", mem_counts)
    logger.info("Cross2: %d", cross2_counts)
    logger.info("VS: %d", vs_counts)
    logger.info("Total fixations on items (vs): %d", total_captured_fixs)
    logger.info("Total fixations on targets: %d", on_targets)
    logger.info("Total fixations on distractors: %d", on_distractors)
    logger.info("Percentage of capture fixations (vs): %.1f %%", total_item_fixed)

    stats = [answer_acc, vs_counts, total_captured_fixs ,on_targets, total_item_fixed ,mean_dur_vs]
    return evts ,stats
   

def plot_fix_durs_mem_vs(evts):
    from matplotlib import pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 6))
    main_phases = [ 'mem', 'vs']
    for phase in main_phases:
        ev = evts[(evts['type'] == 'fixation') & (evts['phase'] == phase)]
        n_fixations = len(ev)
        ax.hist(ev['duration'], bins=80, alpha=0.5, density=True, label=f'{phase} (N={n_fixations})')

    ax.set_xlabel('Duration (ms)')
    ax.set_ylabel('Density')
    ax.set_title('Normalized Fixation Duration Distribution')
    ax.legend()
    plt.show()

def plot_fix_durs_all_phases(evts):
    from matplotlib import pyplot as plt
    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(8, 6))
    axs = axs.flatten()
    phas = ['cross1','mem','cross2','vs']
    for i in range(4):
        ev     = evts[(evts['type']=='fixation') & (evts['phase']==phas[i])]
        n_fixations = len(ev)
        axs[i].hist(ev['duration'], bins=80)
        axs[i].set_xlabel('Duration (ms)')
        axs[i].set_ylabel('Frequency')
        axs[i].set_title(phas[i])
        axs[i].legend([f'N={n_fixations}'])

    plt.tight_layout()
    plt.show()  
     
def plot_trial(eeg,suj,tr,show_borders=True):
    #so far circles radius and external bound are hard coded to 80 
    from utils.paths import paths
    import utils.setup as setup
    import matplotlib.image as mpimg
    from matplotlib import pyplot as plt 
    import matplotlib.patches as patches
    import pandas as pd

    path =    paths()
    exp_path = path.experiment_path()
    item_pos = path.item_pos_path()
    #suj  = load.subject(info,0)
    info = setup.exp_info()
    screensize = info.screen_size
    bh_data     = suj.load_bh_csv()
    evts = suj.load_event_struct()
    image_names = bh_data['searchimage'].drop_duplicates()
    image_names = image_names.str.split('cmp_', expand=True)[1]
    image_names = image_names.str.split('.jpg', expand=True)[0]

    star_samp, stop_samp = start_stop_samples_trigg(evts,'vs')
    image_names = list(image_names)
    image_name  = image_names[tr-1]
    y,x = suj.get_et_data(eeg,[star_samp[tr-1],stop_samp[tr-1]])
    img = mpimg.imread(exp_path + 'cmp_' + image_name + '.jpg')

    target_width = 1024
    target_height = screensize[1]
    if img.shape != (1024, 1280, 3):      
        # Calculate scaling factors
        scale_x = target_width / img.shape[0] 
        scale_y = target_height / img.shape[1] 
        
    
    xim = (screensize[0]-img.shape[1])/2
    yim = (screensize[1]-img.shape[0])/2
    print(xim,yim)

    fig, ax = plt.subplots()##############  SIZE
    #perhaps I should have moved the image boundaries and not the scanpath
    ax.imshow(img)
    ax.plot(y*1e6-xim,x*1e6-yim,'black')

    #################check fix evt marks correspondence with scanpath#############
    fix_start_samps = start_samples_trigg(evts,'fixation')
    fixs_lats = [x for x in fix_start_samps if star_samp[tr-1]  <   x  <  stop_samp[tr-1]]



    for i in fixs_lats:
        try:
            xf = eeg[info.et_channel_names[0],i][0]
            yf = eeg[info.et_channel_names[1],i][0]
        except:
            xf = eeg[info.et_channel_namesL[0],i][0]
            yf = eeg[info.et_channel_namesL[1],i][0]
        if show_borders:
            ax.scatter(xf*1e6-xim,yf*1e6-yim,s=50,color='blue')
        ####add fixations positions from evts data###########
        #point = evts[evts['latency']==i][['fix_avgpos_x','fix_avgpos_y']]
        x_ev = evts[evts['latency']==i]['fix_avgpos_x']
        y_ev = evts[evts['latency']==i]['fix_avgpos_y']
        ax.scatter(x_ev-xim,y_ev-yim,s=40,color='g')


    #############################################################################
    df = pd.read_csv(item_pos)

    if show_borders:
        for index, row in df[df['folder']==image_name].iterrows():
        # Create a rectangle patch for the bounding box
            rect = patches.Rectangle((row['pos_x'], row['pos_y']), row['width'], row['height'], linewidth=2, edgecolor='r', facecolor='none')
            
            # Add the rectangle patch to the plot
            ax.add_patch(rect)
    #########add centers#######
    #df is position csv image_name is searchimage
    trial_stims = df[df['folder']==image_name]
    records = trial_stims.to_records(index=False)
    # Extract the (x, y) values from each record as a tuple using a list comprehension
    centers_list = [(record[6]+record[5]/2, record[7]+record[4]/2) for record in records]
    #closest_tuple(centers_list, 40, (419,500)) add circles
    if show_borders:
        for i in range(len(centers_list)):
            plt.scatter(centers_list[i][0],centers_list[i][1],color='black')
        # Step 4: Add the circle to the plot
            circle = patches.Circle((centers_list[i][0], centers_list[i][1]), 80, edgecolor='red', facecolor='none', linewidth=2, alpha=0.5)
            ax.add_patch(circle)
    
    ##########add frame bounds######
    height, width, _ = img.shape
    width = width + 2 * 80
    height = height + 2 * 80
    print(width,height)
    # Draw the frame around the image
    frame_color = 'red'  # You can change the color here, e.g., 'blue', 'green', 'black', etc.
    frame_width = 1   # You can change the width of the frame here
    if show_borders:
        frame_rect = plt.Rectangle((-80, -80), width, height, linewidth=frame_width, edgecolor=frame_color, facecolor='none')
        ax.add_patch(frame_rect)
    return fig
    

def create_full_metadata(info,sub_id,metadata_path,capturing_thr,save_evts=False):
    full_file_path = os.path.join(metadata_path,f'{sub_id}_full_metadata.csv')
    if not os.path.exists(full_file_path):
        suj  = load.subject(info, sub_id)
        eeg  = suj.load_analysis_eeg()
        #eeg  = suj.load_electrode_positions(eeg)
        evts = suj.load_event_struct()
        bh_data     = suj.load_bh_csv()
        evts, stats  = add_trial_info_to_events(evts,bh_data,capturing_thr)
        evts = add_bad_ET_to_evts(evts)
        evts = add_multi_to_evts(evts)
        evts = add_rank_to_evts(evts)#only for vs 
        evts = add_refix_to_evts(evts)
        if save_evts:
        # Save epoched data     
            evts.to_csv(full_file_path, index=False)
            logger = logging.getLogger()
            logger.info("saving full metadata for subject: %s ", sub_id)
        return evts, stats
    #add remaining functions to add multi rank and refix


def calculate_rms(evoked, channels, tmin, tmax):
    # Extract the data from the specified channels
    channel_data = evoked.copy().pick_channels(channels).data

    # Get the time indices corresponding to tmin and tmax
    tmin_idx = int(evoked.time_as_index(tmin))
    tmax_idx = int(evoked.time_as_index(tmax))

    # Slice the data from tmin to tmax
    sliced_data = channel_data[:, tmin_idx:tmax_idx]

    # Calculate the RMS along the time axis
    rms = np.sqrt(np.mean(sliced_data**2, axis=1))

    return rms

def add_bad_ET_to_evts(evts):
    #create a funtion with this to asses if a fixation belongs to bad_ET segment
    #add bad column, these evetns are based on eye movements recognition they usually happend outside the trials
    interesting_event_type = {'fixation', 'saccade'}
    bad_start, bad_stop = start_stop_samples_trigg(evts,'bad_ET')
        
    evts_tmp = evts
    evts_tmp['bad'] = 0

        
    # Check if latency_value falls between any pair of start and end values
    for start, end in zip(bad_start, bad_stop):#[ind:ind+1]
        # print('pepe')
        # print('bad interval',start,end)
        for index, row in evts_tmp.iterrows():
            latency_value = row['latency']
            # print('event latency',latency_value)
            # print('event type',row['type'])
            # print((start <= latency_value) and (latency_value <= end))
            # print(row['type'] in interesting_event_type)
            if (start < latency_value) and (latency_value < end) and (row['type'] in interesting_event_type):             
                evts_tmp.loc[index, 'bad'] = 1
    return evts_tmp
            
def add_multi_to_evts(evts):
    evts_tmp = evts
    phases_to_multi = ['vs']
    #phase_to_multi can be vs or mem
    evts_tmp['multi'] = np.nan 
    stm_id = evts_tmp.columns.get_loc('stm')
    multi_id = evts_tmp.columns.get_loc('multi')

    for phase in phases_to_multi:
        evts_fix_vs = evts_tmp[(evts_tmp['type']=='fixation') & (evts_tmp['phase']== phase)]
        #THIS CODE SHOULD RUN ON A SLICE OF ONLY FIXATIONS FROM VS OR MSS 


        #-----------------------------------
        # Initialize the 'multi' column with zeros
        # Counter variable for consecutive runs
        count = 1

        # Loop through the DataFrame
        for i in range(1, len(evts_fix_vs)):
            if (evts_fix_vs['stm'].iloc[i] == evts_fix_vs['stm'].iloc[i - 1]) and (evts_fix_vs['trial'].iloc[i] == evts_fix_vs['trial'].iloc[i - 1]) and (evts_fix_vs['stm'].iloc[i] != np.nan):
                # Increment counter for consecutive runsi
                count += 1
            else:
                # Assign the counter value to the 'multi' column
                # Reset the counter for a new run
                if (count == 1):
                    if isinstance(evts_fix_vs.iloc[i-1,stm_id], str):
                        #print(i, count)
                        evts_fix_vs.iloc[i-1,multi_id] = 0
                else:
                    evts_fix_vs.iloc[i - count:i,multi_id] = range(1, count + 1)
                    count = 1
        # Assign the last run counter value to the remaining consecutive values
        if count == 1:
            if isinstance(evts_fix_vs.iloc[len(evts_fix_vs)-1,stm_id], str):
                evts_fix_vs.iloc[len(evts_fix_vs)-1, multi_id] = 0
        else:
            evts_fix_vs.iloc[len(evts_fix_vs) - count:len(evts_fix_vs), multi_id] = range(1, count + 1)

        # ADD NEW MULTI DATA TO EVTS DATA
        evts_tmp.loc[evts_fix_vs.index,:] = evts_fix_vs
    #NOW DO IT WITH MEM
    return evts_tmp

def add_rank_to_evts(evts):
    evts_tmp = evts
    phases_to_multi =  ['vs']#['vs','mem']
    #phase_to_multi can be vs or mem
    evts_tmp['rank'] = np.nan 
    rank_count = 0
    trial_id = evts_tmp.columns.get_loc('trial')
    rank_id = evts_tmp.columns.get_loc('rank')

    if 'inrange' not in evts_tmp.columns:
        raise ValueError("Column 'inrange' does not exist in the DataFrame. Run the corresponding function to add it.")

    for phase in phases_to_multi:
        for tr in range(1,210):
            evts_fix_tmp = evts_tmp[(evts_tmp['type']=='fixation') & (evts_tmp['phase']== phase) & (evts_tmp['trial']== tr)  & (evts_tmp['inrange']== True)]
            #THIS CODE SHOULD RUN ON A SLICE OF ONLY FIXATIONS FROM VS OR MSS 
            rank_range = len(evts_fix_tmp)
            evts_fix_tmp.iloc[:rank_range,rank_id] = range(1,rank_range+1)
            evts_tmp.loc[evts_fix_tmp.index,:] = evts_fix_tmp
            #print(evts_fix_tmp)
    return evts_tmp

def add_refix_to_evts(evts):
    evts_tmp = evts
    evts_tmp['refix'] = np.nan 
    phases_to_multi = ['vs']
    #phase_to_multi can be vs or mem
    stm_id = evts_tmp.columns.get_loc('stm')
    refix_id = evts_tmp.columns.get_loc('refix')
    multi_id = evts_tmp.columns.get_loc('multi')

    
    if 'inrange' not in evts_tmp.columns:
        raise ValueError("Column 'inrange' does not exist in the DataFrame. Run the corresponding function to add it.")
#pp=evts[(evts['type']=='fixation') & (evts['phase']== 'vs') & (evts['trial']== 3)  & (evts['inrange']== True) & ((evts['multi']==0) | (evts['multi']== 1))] 

    for phase in phases_to_multi:
        for tr in range(1,210):
            evts_fix_vs = evts_tmp[(evts_tmp['type']=='fixation') & (evts_tmp['phase']== phase) & (evts_tmp['inrange'] == True)  & (evts['trial']== tr)]
            pp = evts_tmp[(evts_tmp['type']=='fixation') & (evts_tmp['phase']== phase) & (evts_tmp['inrange'] == True)  & (evts['trial']== tr) & ((evts['multi']==0) | (evts['multi']== 1))]

########
            #evts_fix_vs['refix'] = np.nan 
            ppna= pp.dropna(subset=['stm'])
            ppna[ppna.duplicated(subset='stm', keep=False)]
            refixed_stms = ppna.loc[ppna.duplicated(subset='stm', keep=False),'stm'].unique()
########    
            # Loop through the trial DataFrame
            for st in refixed_stms:                         #loop over refixed stims
                refix_count = 1                             #This stim was refixated at least one time
                evts_st_tmp = evts_fix_vs.loc[evts_fix_vs.stm == st]          #Filter only rows to that stim
                for i in range(len(evts_st_tmp)-1):         #I go over the filtered data except the last row
                    if evts_st_tmp.iloc[i,multi_id]==0:        #If the stim is a single fixated item...
                        evts_st_tmp.iloc[i,refix_id] = refix_count 
                        refix_count = refix_count + 1 
                        #print('entra en o if',i,refix_count)#then starts the count and add 1 (there sure be more refixes)
                    else:
                        if evts_st_tmp.iloc[i,multi_id]==evts_st_tmp.iloc[i+1,multi_id]-1: 
                            evts_st_tmp.iloc[i,refix_id] = refix_count
                        else:
                            evts_st_tmp.iloc[i,refix_id] = refix_count
                            refix_count = refix_count + 1      
                            
                            #print('entra en else')#then starts the count and add 1 (there sure be more refixes)
                            
                    #if evts_st_tmp.iloc[-1,multi].values==0:
                    evts_st_tmp.iloc[-1,refix_id] = refix_count




                    evts_fix_vs.loc[evts_st_tmp.index,:] = evts_st_tmp

            evts_fix_vs.loc[~(evts_fix_vs['stm'].isin(refixed_stms) | evts_fix_vs['stm'].isna()), 'refix'] = 0


            # ADD NEW MULTI DATA TO EVTS DATA
            evts_tmp.loc[evts_fix_vs.index,:] = evts_fix_vs
    #NOW DO IT WITH MEM
    return evts_tmp

def plot_time_series_with_navigation(x,y,window,jump): 
    import numpy as np
    import matplotlib.pyplot as plt
    from ipywidgets import Button, VBox
    from IPython.display import display, clear_output

    # Generate example data
    # x = np.linspace(0, 1000, 1000)
    # y = np.sin(x)

    # Define the window size and jump size
    window_size = window
    jump_size = jump

    # Define the current start index
    start_index = 0

    # Define the plotting function
    def plot_window():
        end_index = start_index + window_size

        # Clear previous plot
        clear_output(wait=True)

        # Plot the windowed data
        plt.plot(x[start_index:end_index], y[start_index:end_index])
        plt.xlabel('Time')
        plt.ylabel('Value')
        plt.title('Time Series Window')

        # Set the x-axis limits based on the current window
        plt.xlim(x[start_index], x[end_index])

        # Show the plot
        plt.show()

    # Define the button callback functions
    def next_window(_):
        global start_index
        if start_index + jump_size + window_size <= len(x):
            start_index += jump_size
            plot_window()
            display(buttons_box)

    def previous_window(_):
        global start_index
        if start_index - jump_size >= 0:
            start_index -= jump_size
            plot_window()
            display(buttons_box)

    # Create the Next and Back buttons
    next_button = Button(description='Next')
    back_button = Button(description='Back')

    # Register the button callback functions
    next_button.on_click(next_window)
    back_button.on_click(previous_window)

    # Create a vertical box layout for the buttons
    buttons_box = VBox([back_button, next_button])

    # Display the initial plot and button box
    plot_window()
    display(buttons_box)



def find_different_keys(dict1, dict2):
    different_keys = []

    for key in dict1.keys():
        if key in dict2 and dict1[key] != dict2[key]:
            different_keys.append(key)

    return different_keys

def find_equal_keys(dict1, dict2):
    equal_keys = []

    for key in dict1.keys():
        if key in dict2 and dict1[key] == dict2[key]:
            equal_keys.append(key)

    return equal_keys

def get_effect_name_from_cond_dict(effect_dict):
    diff=find_different_keys(effect_dict['condition_1'],effect_dict['condition_2'])
    equa=find_equal_keys(effect_dict['condition_1'],effect_dict['condition_2'])

    diff_effect_str = []
    equ_string =  []

    for i in equa:
        equ_string.append(i+ '_' + str(effect_dict['condition_1'][i])+'_')
    equ_string_res = ''
    for item in equ_string:
        if item == equ_string[-1]:
            equ_string_res += item 
        else:
            equ_string_res += item + '_'

    for i in diff:
        diff_effect_str.append(i+ '_' + str(effect_dict['condition_2'][i])+'-'+str(effect_dict['condition_1'][i]))
    result_string = ""
    for item in diff_effect_str:
        if item == diff_effect_str[-1]:
            result_string += item 
        else:
            result_string += item + '_'
    return  equ_string_res + result_string 

def pick_chs(chs_id, info):
    '''
    :param chs_id: 'mag'/'LR'/'parietal/occipital/'frontal'/sac_chs/parietal+'
        String identifying the channels to pick.
    :param info: class attribute
        info attribute from the evoked data.
    :return: picks: list
        List of chosen channel names.
    '''

    if chs_id == 'eeg':
        picks = 'eeg'
    # elif chs_id == 'LR':
    #     right_chs = ['MRT51', 'MRT52', 'MRT53']
    #     left_chs = ['MLT51', 'MLT52', 'MLT53']
    #     picks = right_chs + left_chs
    else:
        ids = chs_id.split('_')
        all_chs = info.ch_names
        picks = []
        for id in ids:
            if len(id) == 1:
                picks += [ch_name for ch_name in all_chs if id in ch_name]
            elif id in all_chs:
                picks += [id]
            elif id == 'CF':
                picks += ['C15', 'C16', 'C17', 'C18', 'C19', 'C20', 'C28','C29']
            elif id == 'RAL':
                picks += ['C5','C6','C7','C8','C9','B27','B28']
            elif id == 'LAL':
                picks += ['D5','D6','D7','D8','D9','C30','C31']
            elif id == 'LAM':
                picks += ['D2','D3','D4','D12','D13','C24','C25']
            elif id == 'RAM':
                picks += ['C2','C3','C4','C11','C12','B31','B32']
            elif id == 'LPM':
                pass#picks += ['D2','D3','D4','D12','D13','C24','C25']
            elif id == 'RPM':
                pass#picks += ['C2','C3','C4','C11','C12','B31','B32']
            elif id == 'LOT':
                pass#picks += ['D2','D3','D4','D12','D13','C24','C25']
            elif id == 'ROT':
                pass#picks += ['C2','C3','C4','C11','C12','B31','B32']
            elif id == 'CO':
                picks += ['A14','A15','A22','A23','A24','A27','A28']

            # Subset from picked chanels
            elif id == 'L':
                picks = [ch_name for ch_name in picks if 'M' in ch_name and 'L' in ch_name]
            elif id == 'R':
                picks = [ch_name for ch_name in picks if 'M' in ch_name and 'R' in ch_name]

    return picks

def fig(fig, path, fname):
    """
    Save figure fig with given filename to given path.

    Parameters
    ----------
    fig: figure
        Instance of figure to save
    path: str
        Path to save directory
    fname: str
        Filename of file to save
    """

    # Make dir
    os.makedirs(path, exist_ok=True)
    # Save
    fig.savefig(path + fname + '.png')

    # Create svg directory
    svg_path = path + '/svg/'
    os.makedirs(svg_path, exist_ok=True)
    # Save
    fig.savefig(svg_path + fname + '.svg')

def is_inside_window(event_latency,windows):
    for window in windows:
        if event_latency >= window[0] and event_latency <= window[1]:
            return True
    return False

def plot_bb(eeg,suj,show_borders=True):
    #so far circles radius and external bound are hard coded to 80 
    from utils.paths import paths
    import utils.setup as setup
    import matplotlib.image as mpimg
    from matplotlib import pyplot as plt 
    import matplotlib.patches as patches
    import pandas as pd

    path =    paths()
    exp_path = path.experiment_path()
    item_pos = path.item_pos_path()
    #suj  = load.subject(info,0)
    info = setup.exp_info()
    screensize = info.screen_size
    bh_data     = suj.load_bh_csv()
    evts = suj.load_event_struct()
    image_names = bh_data['searchimage'].drop_duplicates()
    image_names = image_names.str.split('cmp_', expand=True)[1]
    image_names = image_names.str.split('.jpg', expand=True)[0]

    star_samp, stop_samp = start_stop_samples_trigg(evts,'vs')
    image_names = list(image_names)
    image_name  = image_names[0]
    #img = mpimg.imread(exp_path + 'cmp_' + image_name + '.jpg')

    target_width = 1024
    target_height = 1280
    blank_image = np.zeros((target_width,target_height, 3), dtype=np.uint8)

    fig, ax = plt.subplots()##############  SIZE
    ax.imshow(blank_image)

    df = pd.read_csv(item_pos)
    if show_borders:
        for image_name in image_names[:1]:
            scale_x =1
            scale_y =1  
            img = mpimg.imread(exp_path + 'cmp_' + image_name + '.jpg')    
            if  img.shape != (1024,1280,3):      
                # Calculate scaling factors
                scale_x = target_width / img.shape[0] 
                scale_y = target_height / img.shape[1] 
            for index, row in df[df['folder']==image_name].iterrows():

            # Create a rectangle patch for the bounding box
                rect = patches.Rectangle((row['pos_x']*scale_x, row['pos_y']*scale_y), row['width'], row['height'], linewidth=.4, edgecolor='r', facecolor='r',alpha=1)
                
                # Add the rectangle patch to the plot
                ax.add_patch(rect)
    #########add centers#######
    # #df is position csv image_name is searchimage
    # trial_stims = df[df['folder']==image_name]
    # records = trial_stims.to_records(index=False)
    # # Extract the (x, y) values from each record as a tuple using a list comprehension
    # centers_list = [(record[6]+record[5]/2, record[7]+record[4]/2) for record in records]
    # #closest_tuple(centers_list, 40, (419,500)) add circles

    return fig


def plot_bb_hmap(eeg, suj, show_borders=True):
    from utils.paths import paths
    import utils.setup as setup
    import matplotlib.image as mpimg
    from matplotlib import pyplot as plt 
    import matplotlib.patches as patches
    import pandas as pd

    path = paths()
    exp_path = path.experiment_path()
    item_pos = path.item_pos_path()
    info = setup.exp_info()
    target_width = 1024
    target_height = 1280
    blank_image = np.zeros((target_width, target_height), dtype=np.float32)

    bh_data = suj.load_bh_csv()
    evts = suj.load_event_struct()
    image_names = bh_data['searchimage'].drop_duplicates()
    image_names = image_names.str.split('cmp_', expand=True)[1]
    image_names = image_names.str.split('.jpg', expand=True)[0]
    image_names = list(image_names)

    fig, ax = plt.subplots(figsize=(10, 8))  # Adjust the figure size as needed

    df = pd.read_csv(item_pos)
    for image_name in image_names[:-1]:
        scale_x = 1
        scale_y = 1  
        img = plt.imread(exp_path + 'cmp_' + image_name + '.jpg')    
        if img.shape != (1024, 1280, 3):      
            # Calculate scaling factors
            scale_x = target_width / img.shape[0] 
            scale_y = target_height / img.shape[1] 
        for index, row in df[df['folder'] == image_name].iterrows():
            # Add 1 to each occupied pixel in the rectangle
            x, y, w, h = row['pos_x']*scale_x, row['pos_y']*scale_y, row['width'], row['height']
            blank_image[int(y):int(y+h), int(x):int(x+w)] += 1

    # Normalize the image by dividing by the number of image names
    blank_image /= len(image_names[:-1])

    # if show_borders:
    #     for image_name in image_names:
    #         scale_x = 1
    #         scale_y = 1  
    #         img = plt.imread(exp_path + 'cmp_' + image_name + '.jpg')    
    #         if img.shape != (1024, 1280, 3):      
    #             # Calculate scaling factors
    #             scale_x = target_width / img.shape[0] 
    #             scale_y = target_height / img.shape[1] 
    #         for index, row in df[df['folder'] == image_name].iterrows():
    #             rect = patches.Rectangle((row['pos_x']*scale_x, row['pos_y']*scale_y), row['width'], row['height'], linewidth=.4, edgecolor='r', facecolor='r', alpha=1)
    #             ax.add_patch(rect)
    ax.imshow(blank_image, cmap='hot', interpolation='bicubic', origin='upper', extent=[0, target_height, target_width, 0])
    ax.set_xlabel('Horizontal Position [pix]')
    ax.set_ylabel('Vertical Position [pix]')
    ax.set_title('Heatmap of items across trials',fontweight='bold')
    return fig

# Convert bytes to a human-readable format
def convert_size(size_bytes):
    if size_bytes == 0:
        return "0 Bytes"
    size_name = ("Bytes", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB")
    i = int(np.floor(np.log(size_bytes) / np.log(1024)))
    p = np.power(1024, i)
    s = round(size_bytes / p, 2)
    return "{} {}".format(s, size_name[i])

# Example usage:
# plot_bb(eeg_data, suj_instance)
# plt.show()
    

if __name__ == '__main__':
    # import utils.setup as setup
    # import utils.load as load
    # info = setup.exp_info()
    # suj  = load.subject(info,0)
    # evts = suj.load_event_struct()
    # bh_data = suj.load_bh_csv()
    # #lats=start_samples_trigg(info,suj,['mem','vs'])
    # #print(len(lats))
    # # tuples = [(2, 0), (5, 5), (5, 7)]
    # # threshold = 2.5
    # # point = (5, 6)
    # # closest= closest_tuple(tuples, threshold, point)
    # # print(closest)  # Output: (1, 2)
    # #evts = add_trial_info_to_events(evts,bh_data)
    # #plot_fix_durs_mem_vs(evts)

#--------------------EFFECT DICT-------------------
    effect_dict =   { 'condition_1':{
                                'event_type':'fixation',
                                'present':True,
                                'correct':False,    
                                'mss':1,
                                'trials':None,
                                'phase':'vs',
                                'dur':80,
                                'item_type':'ontarget',
                                'dir':None
                                },
                    'condition_2':{
                                'event_type':'fixation',
                                'present':True,
                                'correct':True,    
                                'mss':4,
                                'trials':None,
                                'phase':'vs',
                                'dur':80,
                                'item_type':'ondistractor',
                                'dir':None
                                }
                    }
    # diff=find_different_keys(effect_dict['condition_1'],effect_dict['condition_2'])
    # effect_str = []
    # for i in diff:
    #     effect_str.append(i+ '_' + str(effect_dict['condition_2'][i])+'-'+str(effect_dict['condition_1'][i]))
    # result_string = ""
    # for item in effect_str:
    #     result_string += item + '_'
    print(get_effect_name_from_cond_dict(effect_dict))
