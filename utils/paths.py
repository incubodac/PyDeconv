import os

class paths:
    """
    Paths to participants data.

    Attributes
    -------
    name: str
        Runing computers name result of os.popen('whoami').read().
    """

    def __init__(self,experiment='UBA'):
        self.name = os.popen('whoami').read()
        self.experiment = experiment

        if self.name == 'dac\n':
            if experiment=='UBA':
                if os.path.exists('/media/dac/SSD-curie'):
                    self.main_path = '/media/dac/SSD-curie/Hybrid/'
                elif os.path.exists('/Volumes/DAC500/'):
                    self.main_path = '/Volumes/DAC500/Hybrid/'
                elif os.path.exists('/media/dac/Nuevo vol/EEGDATA/'):
                    self.main_path = '/media/dac/Nuevo vol/EEGDATA/'
                else:
                    print('HDD not connected\n')            
            elif experiment=='UON':
                if os.path.exists('/media/dac/SSD-curie'):
                    self.main_path = '/media/dac/SSD-curie/Hybrid/'
                elif os.path.exists('/Volumes/DAC500/'):
                    self.main_path = '/Volumes/DAC500/Hybrid/'
                elif os.path.exists('/media/dac/Nuevo vol/EEGDATA/HS_EEG_UoN/'):
                    self.main_path = '/media/dac/Nuevo vol/EEGDATA/HS_EEG_UoN/'
                else:
                    print('HDD not connected\n')



        elif self.name == '':
            self.main_path = ''


    def eeg_analysis_path(self):
        """       
        Paths to participants EEG data to analyze.

        Returns
        -------
        eeg_path: str
            Path in str format to the .set and .fdt EEG data.
        """
        if self.experiment == 'UBA':
            eeg_path = self.main_path + 'Hybrid_preana_out/8.data_analysis/'
        elif self.experiment == 'UON':
            eeg_path = self.main_path + 'preAnalysis/8.data_analysis/'


        return eeg_path

    def eeg_raw_path(self):
        """       
        Paths to participants raw EEG data.

        Returns
        -------
        eeg_path: str
            Path in str format to the  EEG bdf data and the 
            bh data tables.
        """
        if self.experiment == 'UBA':
            raw_path = self.main_path + 'HybridSearch/'
        elif self.experiment == 'UON':
            raw_path = self.main_path + 'HS_EEG_BDF/'

        return raw_path

    def evts_path(self):
        """
        Paths to participants event eeglab struct data
        
        (MATLAB).

        Returns
        -------
        et_path: str
            Path in str format to the ET folder containing the Eye-Tracker data.
        """
        if self.experiment == 'UBA':
            evts_path = self.main_path + 'Hybrid_preana_out/9.event_structs/'
        elif self.experiment == 'UON':
            evts_path = self.main_path + 'preAnalysis/9.event_structs/'

        return evts_path





    def results_path(self):
        """
        Paths to the results folder.

        Returns
        -------
        results_path: str
            Path in str format to the folder to store the results.
        """
        if self.experiment == 'UBA':
            results_path = self.main_path + 'Hybrid_ana_out/'
        elif self.experiment == 'UON':
            results_path = self.main_path + 'UON_ana_out/'

        # Create directory if it doesn't exist
        os.makedirs(results_path, exist_ok=True)

        return results_path
    
    def effects_path(self,effect_name):
        """
        Paths to the results folder.

        Returns
        -------
        results_path: str
            Path in str format to the folder to store the results.
        """

        effect_path = self.main_path + 'Hybrid_ana_out/effect/' + effect_name

        # Create directory if it doesn't exist
        os.makedirs(effect_path, exist_ok=True)

        return effect_path
    
    def model_path(self,model_name):
        """
        Paths to the results folder.

        Returns
        -------
        results_path: str
            Path in str format to the folder to store the results.
        """
        if self.experiment == 'UBA':
            model_path = self.main_path + 'Hybrid_ana_out/models/' + model_name
        elif self.experiment == 'UON':
            model_path = self.results_path()+ 'models/' + model_name

        # Create directory if it doesn't exist
        os.makedirs(model_path, exist_ok=True)

        return model_path
    
    def combined_model_path(self,model_name):
        """
        Paths to the results folder for both datasets.

        Returns
        -------
        results_path: str
            Path in str format to the folder to store the results.
        """
        
        model_path = self.main_path + 'combined_models/' + model_name


        # Create directory if it doesn't exist
        os.makedirs(model_path, exist_ok=True)

        return model_path

    def plots_path(self):
        """
        Paths to the plots folder.

        Returns
        -------
        plots_path: str
            Path in str format to the folder to store the results.
        """

        plots_path = self.main_path + 'Hybrid_ana_out/plots/'

        # Create directory if it doesn't exist
        os.makedirs(plots_path, exist_ok=True)

        return plots_path


    def item_pos_path(self):
        """
        Paths to the search items positions file.

        Returns
        -------
        item_pos_path: str
            Path in str format to the items positions file.
        """

        item_pos_path = self.main_path + 'HybridSearch/posiciones_estimulos_completa_2022.csv'

        return item_pos_path


    def experiment_path(self):
        """
        Paths to the Psychopy experiment directory.

        Returns
        -------
        exp_path: str
            Path in str format to the items positions file.
        """

        exp_path = self.main_path + 'HybridSearch/Psychopy experimento/'

        return exp_path
    
    def log_path(self):
        """
        Paths to the logs folder

        Returns
        -------
        log_path: str
            Path in str format to the folder to store the log files.
        """

        log_path = self.main_path + 'Hybrid_ana_out/logs/'

        # Create directory if it doesn't exist
        os.makedirs(log_path, exist_ok=True)
        
        return log_path
    
    def full_metadata_path(self):
        """
        Paths to the logs folder

        Returns
        -------
        log_path: str
            Path in str format to the folder to store the log files.
        """
        if self.experiment=='UBA':
            full_metadata_path = self.main_path + 'Hybrid_ana_out/metadata/'
        elif self.experiment=='UON':
            full_metadata_path = self.main_path + 'UON_ana_out/metadata/'

        # Create directory if it doesn't exist
        os.makedirs(full_metadata_path, exist_ok=True)
        
        return full_metadata_path
    
    def reduced_heads_path(self):
        """
        Paths to the reduced head subjects folder

        Returns
        -------
        log_path: str
            Path in str format to the folder to store the log files.
        """
        if self.experiment=='UBA':
            reduced_path = self.main_path + 'Hybrid_ana_out/reduced_heads/'
        elif self.experiment=='UON':
            reduced_path = self.main_path + 'UON_ana_out/reduced_heads/'

        # Create directory if it doesn't exist
        os.makedirs(reduced_path, exist_ok=True)
        
        return reduced_path





