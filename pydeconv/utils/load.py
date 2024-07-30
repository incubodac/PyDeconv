# from utils.paths import paths
# from utils.setup import *
import os
import pathlib
import pickle
import mne
import pandas as pd
import matplotlib.pyplot as plt
from art import *
import logging


def config(path, fname):
    """
    Try and load the run configuration and setup information.
    If no previous configuration file was saved, setup config obj.

    Parameters
    ----------
    path: str
        The path to the directory where configuration file is stored.
    fname: str
        The filename for the configuration file.

    Returns
    -------
    config: class
        Class containgn the run configuration and setup information.
    """

    try:
        # Load
        filepath = path + fname
        f = open(filepath, 'rb')
        config = pickle.load(f)
        f.close()

        # Set save config as false
        config.update_config = False

    except:
        # Create if no previous configuration file
        config = setup.config()

    return config


def var(file_path):
    """
    Load variable from specified path

    Parameters
    ----------
    file_path: str
        The path to the file to load.

    Returns
    -------
    var: any
        The loaded variable.
    """
    # Load
    f = open(file_path, 'rb')
    var = pickle.load(f)
    f.close()

    return var


def preproc_subject(exp_info, subject_code):
    """
    Load preprocessed subject object.

    Attributes
    --------
    fixations:
    saccades:
    config?:

    Parameters
    --------
    exp_info: class
       Experiment information class

    Returns
    -------
    preproc_subject: class
        The preprocessed subject class
    """

    # Select 1st subject by default
    if subject_code == None:
        subject_id = exp_info.subjects_ids[0]
    # Select subject by index
    elif type(subject_code) == int:
        subject_id = exp_info.subjects_ids[subject_code]
    # Select subject by id
    elif type(subject_code) == str and (subject_code in exp_info.subjects_ids):
        subject_id = subject_code
    else:
        print('Subject not found.')

    # Preprocessing configuration
    preproc_path = paths().preproc_path()
    file_path = pathlib.Path(os.path.join(preproc_path, subject_id, f'Subject_data.pkl'))
    try:
        f = open(file_path, 'rb')
        preproc_subject = pickle.load(f)
        f.close()
    except:
        print(f'Directory: {os.listdir(pathlib.Path(os.path.join(preproc_path, subject_id)))}')
        raise ValueError(f'Preprocessed data for subject {subject_id} not found in {file_path}')

    return preproc_subject


def ica_subject(exp_info, subject_code):
    """
    Load ica subject object.

    Attributes
    --------
    fixations:
    saccades:
    config?:

    Parameters
    --------
    exp_info: class
       Experiment information class

    Returns
    -------
    ica_subject: class
        The ica subject class
    """

    # Select 1st subject by default
    if subject_code == None:
        subject_id = exp_info.subjects_ids[0]
    # Select subject by index
    elif type(subject_code) == int:
        subject_id = exp_info.subjects_ids[subject_code]
    # Select subject by id
    elif type(subject_code) == str and (subject_code in exp_info.subjects_ids):
        subject_id = subject_code
    else:
        print('Subject not found.')

    # Preprocessing configuration
    ica_path = paths().ica_path()
    file_path = pathlib.Path(os.path.join(ica_path, subject_id, f'Subject_data.pkl'))
    try:
        f = open(file_path, 'rb')
        ica_subject = pickle.load(f)
        f.close()
    except:
        print(f'Directory: {os.listdir(pathlib.Path(os.path.join(ica_path, subject_id)))}')
        raise ValueError(f'Preprocessed data for subject {subject_id} not found in {file_path}')

    return ica_subject


class subject:
    """
    Class containing methods to get eeg,et and bh 
    data for a subject.

    Attributes
    -------
    _path: str
        Path to the EEG data.
    subjects_ids: list
        List of subject's id.

    subjects_groups: list
        List of subject's group
    """
    def __init__(self,exp_info,subject_code=None):
        self.experiment = exp_info.experiment
        
        if subject_code == None:
            subject_id = exp_info.subjects_ids[0]
        elif type(subject_code) == int:
            subject_id = exp_info.subjects_ids[subject_code]
        elif type(subject_code) == str and (subject_code in exp_info.subjects_ids):
            subject_id = subject_code
        else:
            subject_id = None
            print('Subject not found.')

        if subject_id is not None:    
            self.subject_id   = subject_id
            logger = logging.getLogger()
            logger.info("Class containing subject %s information was loaded",subject_id)
            logger.info("-----------------------------------------------------")
        

    def load_bh_csv(self):
        if self.experiment=='UBA':
            bh_csv_path = paths().eeg_raw_path() + self.subject_id + '/'
            files = os.listdir(bh_csv_path)
            # Filter the list to include only CSV files
            csv_files = [f for f in files if f.endswith('.csv') and not f.startswith('.')]
        elif self.experiment=='UON':
            bh_csv_path = paths(self.experiment).main_path +'Psychopy_experiment/data/' 
            files = os.listdir(bh_csv_path)
            # Filter the list to include only CSV files
            csv_files = [f for f in files if f.endswith('.csv') and f.startswith(str(self.subject_id))]
            print(files,bh_csv_path)

        if len(csv_files) == 1:
            # Read the CSV file into a DataFrame
            tprint(f'''\nLoading behavioural data.....
            from   {csv_files}\n''',font="fancy67")
            csv_path = os.path.join(bh_csv_path, csv_files[0])
            df = pd.read_csv(csv_path)
            logger = logging.getLogger()
            logger.info("Behaviour csv from subject %s was loaded",self.subject_id)
            logger.info("-----------------------------------------------------")

            return df
        else:
            print('Error: There is not exactly one CSV file in the folder.\n')
            print(csv_files)
            

    def load_analysis_eeg(self,reduced=False):
        if reduced==True:
            tprint(f'''\nLoading EEG data REDUCED VERSION.....
            subject   {self.subject_id}\n''',font="fancy67")
            # get subject path
            set_path =  paths(self.experiment).reduced_heads_path()
            set_file =  os.path.join(set_path,f'{self.subject_id}_analysis_64.fif')

            # Load sesions
            try:
                raw     =  mne.io.read_raw_fif(set_file, preload=True)
                return raw
            # Missing data
            except FileNotFoundError:
                print('No .fif files found in directory: {}'.format(set_path))
                
        elif reduced == False:           
            tprint(f'''\nLoading EEG data.....
            subject   {self.subject_id}\n''',font="fancy67")
            # get subject path
            set_path =  paths(self.experiment).eeg_analysis_path()
            set_file =  os.path.join(set_path,f'{self.subject_id}_analysis.set')

            # Load sesions
            try:
                raw     = mne.io.read_raw_eeglab( set_file, preload=True)
                raw = self.load_electrode_positions(raw) #uncomment this for UBA 128
                return raw
            # Missing data
            except FileNotFoundError:
                print('No .set files found in directory: {}'.format(set_path))

    def load_event_struct(self):
        tprint('''\nLoading events data.....
        from\n''',font="fancy67")        # get subject path
        #exp_paths = paths(self.experiment)
        evts_path =  paths(self.experiment).evts_path()
        evts_file =  os.path.join(evts_path,f'{self.subject_id}_events.csv')
        tprint(evts_file +'\n',font="fancy67")
        # Load sesions
        try:
            df = pd.read_csv(evts_file)
            logger = logging.getLogger()
            logger.info("Events csv from subject %s was loaded",self.subject_id)
            logger.info("-----------------------------------------------------")
            return df
        # Missing data
        except FileNotFoundError:
            print('No event files found in directory: {}'.format(evts_path))


    def get_et_data(self,raw,sample_limits,plot=0):
        '''
        sample_limits = [start_sample, stop_sample] list
        plot 1/0
        '''
        try:
            et_chans = exp_info().et_channel_names
            et       = raw[et_chans,sample_limits[0]:sample_limits[1]]
        except:
            et_chans = exp_info().et_channel_namesL
            et       = raw[et_chans,sample_limits[0]:sample_limits[1]]
        x = et[0].T[:,0]
        y = et[0].T[:,1]

        if plot:
            plt.plot(x,y)
            plt.show()
        return x , y 
    
    def load_electrode_positions(self,raw):
        if self.experiment== 'UBA':
            montage = mne.channels.make_standard_montage('biosemi128')
        elif self.experiment== 'UON':
            montage = mne.channels.make_standard_montage('biosemi64')
        raw.set_montage(montage, on_missing='ignore')
        return raw
    
    def load_metadata(self):
        tprint('''\nLoading events data.....
        from\n''',font="fancy67")        # get subject path
        sub_id = self.subject_id
        metadata_path =  paths(self.experiment).full_metadata_path()
        evts_file =  os.path.join(metadata_path,f'{sub_id}_full_metadata.csv')
        tprint(evts_file +'\n',font="fancy67")
        # Load sesions
        try:
            df = pd.read_csv(evts_file)
            logger = logging.getLogger()
            logger.info("Events csv from subject %s was loaded",self.subject_id)
            logger.info("-----------------------------------------------------")
            return df
        # Missing data
        except FileNotFoundError:
            print('No event files found in directory: {}'.format(evts_path))

def load_experiment(experiment):
    path = paths(experiment)
    info = exp_info(experiment)
    tprint(f'\nEEG data will be loaded from:...\n {path.eeg_analysis_path()}',font="fancy67")
    return path, info

if __name__== '__main__':
    sub = exp_info()
    pp=   subject(sub,0)        
    raw      = pp.load_analysis_eeg()
    samlim = [133,2000000]
    event=pp.get_et_data(raw,samlim,1)
    #print(event)