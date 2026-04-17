import numpy as np 
from matplotlib import pyplot as plt
from scipy.stats import skewnorm, uniform
import pandas as pd

class EEGSimulator:
    def __init__(self, duration, sample_rate):
        """
        Initialize the EEGSimulator.

        Args:
            duration (float): Total duration of the simulated signal in seconds.
            sample_rate (int): The sampling rate (Hz).

        Attributes:
            duration (float): Duration of the signal (seconds).
            sample_rate (int): Sampling rate of the signal (Hz).
            samples (int): Number of samples in the time series.
            time (np.ndarray): Array representing the time axis.
            data (np.ndarray): The simulated EEG data array.
            evts (pd.DataFrame): DataFrame to store event information.
        """
        self.duration = duration
        self.sample_rate = sample_rate
        self.samples = int(duration * sample_rate)
        self.time = np.arange(0, duration, 1/sample_rate)
        self.data = np.zeros(self.time.shape)
        self.evts = pd.DataFrame(columns=['latency', 'type', 'categorical', 'continuous'])

        
    def data_stats(self):
        """
        Display and print basic statistics for the simulated EEG data.

        Prints:
            Mean, median, and variance of the EEG data.
        """
        median = np.median(self.data)
        mean = np.mean(self.data)
        variance = np.var(self.data)
        print(f'Mean:    {mean} \nMedian:    {median}\nVariance:  {variance}')

    def add_noise(self, scale=0.5, noise_type='brown'):
        """
        Add noise to the EEG data.

        Args:

            scale (float, optional): Amount by which to scale the generated noise (default=0.5).

        Notes:
            The generated brown noise is added in-place to self.data.
        """
        noise = np.random.randn(self.samples+1)
        freqs = np.fft.fftfreq(self.samples+1, 1 / self.sample_rate)

        if noise_type == 'brown':
            psd = 1 / np.abs(freqs[1:]) ** 2
        elif noise_type == 'white':
            psd = 1
        elif noise_type == 'pink':
            psd = 1 / np.abs(freqs[1:])
        else:
            raise ValueError(f"Unsupported noise type: {noise_type}")

        noise = np.fft.fft(noise[1:])
        noise = noise[:len(psd)]
        noise *= np.sqrt(psd)
        noise = np.real(np.fft.ifft(noise))
        noise *= scale

        self.data += noise

    def get_psd(self):
        """Return the Power Spectral Density (PSD) of the data."""
        N = self.samples
        X = np.fft.fft(self.data)/N
        psd = 10 * np.log10(2 * np.abs(X[:int(N/2)+1]) ** 2)
        freq = np.linspace(0, self.sample_rate/2, int(N/2)+1)
        return freq, psd

    def plot_datanpsd(self):
        """
        Plot the simulated data (timeseries) and its Power Spectral Density (PSD).

        Shows:
            - Upper panel: EEG timeseries with vertical lines indicating event onsets by condition.
            - Lower panel: PSD of the simulated data up to 100 Hz.
        """
        frec, pwr = self.get_psd()  # Get the power spectral density
        t = self.time  # Time array
        fig, axs = plt.subplots(2, 1, figsize=(8, 6), tight_layout=True)
        
        # Plot the data signal
        axs[0].plot(t, self.data, lw=0.8)
        
        # Define a color map or list for different conditions (responses)
        colors = ['r', 'b', 'g', 'm', 'c', 'y', 'k']  # Add more colors if needed
        used_conditions = set()  # To track the conditions already plotted in the legend
        
        # Plot vertical lines for each event based on its type (condition)
        for onset, condition in zip(self.onsets, self.conditions):
            # Ensure that the condition value has a corresponding color
            color_idx = condition % len(colors)  # Use modulus to loop through colors if more conditions than colors
            color = colors[color_idx]
            
            # Plot the vertical line for this event (convert sample index to seconds)
            axs[0].axvline(x=onset / self.sample_rate, color=color, linestyle='--', label=f'ERP {condition} (event {condition})')
            
            # Avoid duplicate labels in the legend by adding only once per condition
            if condition not in used_conditions:
                used_conditions.add(condition)
        
        # Optional: Add a legend (only one label per event type)
        handles, labels = axs[0].get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        axs[0].legend(by_label.values(), by_label.keys())
        
        # Set labels and titles for the first plot
        axs[0].set_xlabel('Time (s)')
        axs[0].set_ylabel('Amplitude')
        axs[0].set_title('Data signal')
        
        # Plot the power spectral density (PSD)
        idx = np.where(frec >= 100)[0][0]
        axs[1].plot(frec[:idx], pwr[:idx], lw=0.8)
        
        # Set labels and titles for the second plot
        axs[1].set_xlabel('Frequency (Hz)')
        axs[1].set_ylim([-100, 40])
        axs[1].set_ylabel('Power (dB)')
        axs[1].set_title('Spectral Power')
        
        plt.show()


    def simulate(self, noise='brown',erp_ker=None, isi = {'dist': 'uniform', 'lims': [100,400]} ,w_matrix = None ,add_linear_mod = False):
        """
        Simulates the EEG data. example of erp_kernel:
        kernels = {
            0: {'onsets': [0, 0.19, 0.25], 'amplitudes': [0.1, -0.05, 0.04], 'widths': [0.05, 0.05, 0.07], 'weight':0.3},
            1: {'onsets': [0, 0.19, 0.25], 'amplitudes': [0.1, -0.05, 0.04], 'widths': [0.05, 0.05, 0.07], 'weight':0.3},
            2: {'onsets': [0, 0.19, 0.25], 'amplitudes': [0.1, -0.07, 0.04], 'widths': [0.05, 0.05, 0.07], 'weight':0.3},
            3: {'onsets': [0, 0.19, 0.25], 'amplitudes': [0.1, -0.07, 0.04], 'widths': [0.05, 0.05, 0.07], 'weight':0.3},
            'modulation': {'ker_idx2mod': 1, 'mod': 'linear','dist': 'uniform', 'lims': [100, 600]}
        }
        each kernel index (0, 1, 2, 3) corresponds to a different ERP response, defined by its onsets, amplitudes, and widths.
        The 'weight' parameter can be used to determine the probability of transitioning to that kernel in the sequence of events
        in case the w matrix is not provided. 
        The 'modulation' entry is an example of how you might specify a linear modulation for one of the kernels (in this case, kernel index 1), with its own distribution for the modulation values.
        w_matrix is a transition matrix that defines the probabilities of transitioning from one kernel to another in the sequence of events.
        but in future implementation we may want to have a more flexible way of defining how events are chosen, for example by using a 
        more complex state machine or by allowing for different types of modulations that can affect the transition probabilities between kernels.
        The basic next idea would be to separate structure conditions and behavioural contions, where the first would be related to block structure of
        the experiment and the secont to event to event changes in the task. For example, in a visual search task, the structure conditions could be
        related to the type of search (feature vs conjunction) and the behavioural conditions could be related to the presence or absence of a target in the display.
        """
        self.data = np.zeros(self.samples)
        
        if noise is not None:
            self.add_noise(noise_type=noise)
        else:
            print('No noise added')

        # Retrieve all possible conditions from the ERP kernel dictionary
        if erp_ker is None:
            raise ValueError("erp_ker dictionary must be provided.")
        else:
            erp_conditions = list(erp_ker.keys())
        ker_erp_idx = [cond for cond in erp_conditions if isinstance(cond, (int, float))]
        #here I check the index for each kernel in the kernel dictionary 
        weights = [erp_ker[cond]['weight'] for cond in ker_erp_idx]
        # Normalize weights (in case they don't sum to 1)
        for i, row in enumerate(w_matrix):
            w_matrix[i] = np.array(row) / np.sum(row)

        # hardcoded padding
        left_padding = ker_erp_idx[0]

        # Create a conditions_array with random 1s and 0s
        sample_size = 1200

        # chosing the condition index for the train of responses being generated
        current_state = 0 

        # Array to hold the sequence of states
        states_sequence = [current_state]

        # Generate the sequence
        for _ in range(sample_size - 1):  # Since the initial state is already included
            # Select next state based on transition probabilities of current state
            next_state = np.random.choice(ker_erp_idx, p=w_matrix[current_state])
            # Append the new state to the sequence
            states_sequence.append(next_state)
            # Update current state
            current_state = next_state

        times = []
        samples = []
        
    # Generate ISI samples based on the sequence of states and their corresponding ISI parameters   
        for choice in states_sequence:
            isi_param = getattr(self, f'isi_{choice}', None)

            if isi_param:
                if isi_param['dist'] == 'skewed':
                    skew = isi_param['skew']
                    loc = isi_param['lims'][0]
                    scale = isi_param['scale']
                    samples.append(skewnorm.rvs(skew, loc=loc, scale=scale))
                elif isi_param['dist'] == 'uniform':
                    loc = isi_param['lims'][0]
                    scale = isi_param['lims'][1] - loc
                    samples.append(uniform.rvs(loc=loc, scale=scale))
                else:
                    raise ValueError(f"Unsupported distribution: {isi_param['dist']}")
            else:
                raise AttributeError(f"ISI parameters not set for condition {choice}.")
        
        samples.insert(0,left_padding)
        times = np.cumsum(samples) # latency for all events
        
        # add onsets to self.onsets
        self.onsets = times[:-1]*self.sample_rate
        self.evts['latency'] = times[:-1]*self.sample_rate
        # add conditions to self.onsets
        self.conditions = states_sequence
        self.evts['type'] = states_sequence
        # if add_linear_mod:
        #     # currently linear modulation is hardcoded to be a truncated gaussian
        #     # intended to match saccade amplitude values in the linear region (1-3 degs)
        #     # as signals are in arbitrary units this is just for the sake of 
        #     mod_feature_values =  
        
        self.add_neural_responses(times, states_sequence,erp_ker=erp_ker)

        return self.data

    def gaussian_response(self, onset, amp, width, short_time= False):
        """Creates a Gaussian response starting at onset."""
        times = self.time
        if short_time is not False:
            times = short_time
        mu = onset + 2 * width
        gaussian = 1 / (width * np.sqrt(2 * np.pi)) * np.exp(-0.5 * ((times - mu) / width) ** 2)
        gaussian *= amp
        return gaussian
    

    def add_neural_responses(self, erp_onsets, conditions_array, erp_ker=None):
        # implement how to  model collinerity 
        if erp_ker is None:
            self.erp_ker = {
                0: {'onsets': [0, 0.19, 0.25], 'amplitudes': [0.1, -0.05, 0.04], 'widths': [0.05, 0.05, 0.07]},
                1: {'onsets': [0, 0.19, 0.25], 'amplitudes': [0.1, -0.07, 0.04], 'widths': [0.05, 0.05, 0.07]}
            }
        else:
            self.erp_ker = erp_ker

        response = np.zeros(len(self.time))
        
        # Ensure conditions_array are integers
        conditions_array = [int(cond) for cond in conditions_array]

        for onset, cond in zip(erp_onsets, conditions_array):
            if cond in self.erp_ker:
                for resp in range(len(self.erp_ker[cond]['onsets'])):
                    resp_onset = self.erp_ker[cond]['onsets'][resp] + onset
                    resp_ampli = self.erp_ker[cond]['amplitudes'][resp]
                    resp_width = self.erp_ker[cond]['widths'][resp]
                    response += self.gaussian_response(resp_onset, resp_ampli, resp_width)

        self.data += response
        
        
    def get_neural_response(self, onset, kernel_idx,times):
        """Returns the neural response for a given onset and kernel index."""
        if kernel_idx not in self.erp_ker:
            raise ValueError(f"Kernel index {kernel_idx} not found in ERP kernel.")
        
        response = np.zeros(500) #It shoulden't be hardcoded
        
        # Loop through each component of the ERP kernel (e.g., multiple peaks)
        for resp in range(len(self.erp_ker[kernel_idx]['onsets'])):
            resp_onset = self.erp_ker[kernel_idx]['onsets'][resp] + onset
            resp_ampli = self.erp_ker[kernel_idx]['amplitudes'][resp]
            resp_width = self.erp_ker[kernel_idx]['widths'][resp]
            response += self.gaussian_response(resp_onset, resp_ampli, resp_width,short_time=times)
        
        return response

    
    def create_isi_pdf(self, kernel_idx, sample_size, lims=[.1, .6], dist_type='uniform', mode=.1, skew=0, scale=1):
        """Create and store ISI PDF parameters."""
        isi_params = {
            'dist': dist_type,
            'sample_size': sample_size,
            'lims': lims,
            'mode': mode,
            'skew': skew,
            'scale': scale
        }
        setattr(self, f'isi_{kernel_idx}', isi_params)  # Store ISI parameters dynamically

    def plot_isi_pdf(self, kernel_idx):
        """Plot the ISI PDF based on the kernel index."""
        if hasattr(self, f'isi_{kernel_idx}'):
            print(f"Plotting ISI PDF for kernel {kernel_idx}")

            # Retrieve ISI parameters
            isi_params = getattr(self, f'isi_{kernel_idx}')
            lims = isi_params['lims']
            sample_size = isi_params['sample_size']
            scale = isi_params['scale']
            mode = isi_params['mode']
            skew = isi_params['skew']
            dist_type = isi_params['dist']

            # Generate x values
            x = np.linspace(lims[0] - .1, lims[1] + .1, sample_size)

            # Generate PDF based on type
            if dist_type == 'uniform':
                pdf = uniform.pdf(x, loc=lims[0], scale=lims[1] - lims[0])
            elif dist_type == 'skewed':
                pdf = skewnorm.pdf(x, skew, loc=mode, scale=scale)
            else:
                raise ValueError(f"Unsupported distribution type: {dist_type}")

            # Plot the PDF
            plt.plot(x, pdf, label=f'ISI {kernel_idx}')
            plt.title(f'ISI PDF for kernel {kernel_idx}')
            plt.xlabel('ISI')
            plt.ylabel('Probability Density')
            plt.legend()
            plt.show()
        else:
            print(f"Simulation object does not have an ISI for kernel {kernel_idx}")
    
    def plot_response_idx(self, kernel_idx):
        """
        Plot example ERP response and ISI PDF for a given kernel/condition.

        Args:
            kernel_idx (int): Kernel index for which to show neural response and ISI PDF.

        Shows:
            Matplotlib figure with:
                - Upper panel: ERP response for the chosen kernel.
                - Lower panel: ISI distribution for that kernel.
        Raises:
            ValueError: If ISI distribution type is unsupported.
        """
        if not hasattr(self, f'isi_{kernel_idx}'):
            print(f"None ISI params for kernel {kernel_idx}")
            return
        
        # Retrieve ISI parameters
        isi_params = getattr(self, f'isi_{kernel_idx}')
        lims = isi_params['lims']
        sample_size = isi_params['sample_size']
        scale = isi_params['scale']
        mode = isi_params['mode']
        skew = isi_params['skew']
        dist_type = isi_params['dist']
        
        # Generate ISI PDF
        x = np.linspace(lims[0] - 0.1, lims[1] + 0.1, sample_size)
        if dist_type == 'uniform':
            pdf = uniform.pdf(x, loc=lims[0], scale=lims[1] - lims[0])
        elif dist_type == 'skewed':
            pdf = skewnorm.pdf(x, skew, loc=mode, scale=scale)
        else:
            raise ValueError(f"Unsupported distribution type: {dist_type}")
        
        # Create figure
        fig, axs = plt.subplots(2, 1, figsize=(6, 8), tight_layout=True)
        
        # Generate the neural response using get_neural_response method
        onset = 2 * self.erp_ker[kernel_idx]['widths'][0]
        times = np.linspace(-onset,-onset+1,500)
        neural_response = self.get_neural_response(0,kernel_idx,times)
        
        # Plot the data signal (ERP response)
        axs[0].plot(times, neural_response, lw=0.8)
        axs[0].set_xlabel('Time (s)')
        axs[0].set_ylabel('Amplitude')
        axs[0].set_title(f'ERP response for kernel {kernel_idx}')
        
        # Plot the ISI probability distribution
        axs[1].plot(x, pdf, lw=0.8)
        axs[1].set_xlabel('Time (s)')
        axs[1].set_ylabel('Probability')
        axs[1].set_title(f'ISI distribution for kernel {kernel_idx}')
        # plt.show(block=True)

        
    def plot_all_responses(self):
        print('not implemented')


# new suggestion 2026

class ExperimentDesign:
    """Defines the structure an experimenter would create."""
    
    def __init__(self, sfreq=500):
        self.sfreq = sfreq
        self.trial_types = {}    # condition definitions
        self.event_sequence = {} # events within each trial type
        self.kernels = {}        # ERP kernel per event label
        self.covariates = {}     # how covariates modulate kernels
        self.blocks = []         # block structure
    
    def add_trial_type(self, name, events, probability=None):
        """
        Define a trial type with its within-trial event sequence.
        
        E.g.: 
          design.add_trial_type('congruent', 
              events=[
                  {'label': 'fixation', 'onset': 0},
                  {'label': 'stimulus', 'onset': {'dist': 'uniform', 'lims': [0.5, 0.8]}},
                  {'label': 'response', 'onset': {'dist': 'exgaussian', 'mu': 0.4, 'sigma': 0.05, 'tau': 0.1}}
              ],
              probability=0.5)
        """
        self.trial_types[name] = {
            'events': events,
            'probability': probability
        }
    
    def add_block(self, trial_types, n_trials, iti):
        """
        Define a block of trials.
        
        E.g.:
          design.add_block(
              trial_types=['congruent', 'incongruent'],  
              n_trials=60,
              iti={'dist': 'uniform', 'lims': [0.8, 1.5]})
        """
        self.blocks.append({
            'trial_types': trial_types,
            'n_trials': n_trials,
            'iti': iti
        })
    
    def add_kernel(self, event_label, components):
        """
        Define the ERP kernel for an event label.
        
        E.g.:
          design.add_kernel('fixation', 
              components=[
                  {'onset': 0.0,  'amplitude': 0.08, 'width': 0.04},
                  {'onset': 0.17, 'amplitude': -0.05, 'width': 0.05}
              ])
        """
        self.kernels[event_label] = components
    
    def add_covariate(self, name, kind, affects, **kwargs):
        """
        Define a covariate that modulates an ERP.
        
        Categorical (from trial type):
          design.add_covariate('effect', kind='categorical',
              affects='stimulus', 
              mapping={'congruent': 0, 'incongruent': 1},
              amplitude_scale={'congruent': 1.0, 'incongruent': 1.4})
        
        Continuous (behavioral):
          design.add_covariate('rt', kind='continuous',
              affects='stimulus',
              source='response_onset',  # derived from response event timing
              modulation='linear', slope=0.3)
              
          design.add_covariate('sac_amplitude', kind='continuous',
              affects='fixation',
              dist={'dist': 'truncnorm', 'mean': 2.0, 'std': 1.0, 'lims': [0.5, 5.0]},
              modulation='linear', slope=0.5)
        """
        self.covariates[name] = {'kind': kind, 'affects': affects, **kwargs}


class EEGSimulator_v2:
    def __init__(self, design: ExperimentDesign, duration=None, noise='brown'):
        self.design = design
        self.sfreq = design.sfreq
        self.noise = noise
        self._duration = duration  # if None, derived from events

    def _sample_from_dist(self, dist_spec):
        """Sample a single value from a distribution specification dict."""
        if isinstance(dist_spec, (int, float)):
            return float(dist_spec)
        dist = dist_spec['dist']
        lims = dist_spec.get('lims', [0, 1])
        if dist == 'uniform':
            return uniform.rvs(loc=lims[0], scale=lims[1] - lims[0])
        elif dist == 'skewed':
            skew = dist_spec.get('skew', 0)
            scale = dist_spec.get('scale', 0.1)
            return skewnorm.rvs(skew, loc=lims[0], scale=scale)
        elif dist == 'truncnorm':
            from scipy.stats import truncnorm
            mean = dist_spec['mean']
            std = dist_spec['std']
            a = (lims[0] - mean) / std
            b = (lims[1] - mean) / std
            return truncnorm.rvs(a, b, loc=mean, scale=std)
        else:
            raise ValueError(f"Unsupported distribution: {dist}")

    def _generate_trial_sequence(self, block):
        """Generate a randomized sequence of trial type names for a block."""
        trial_types = block['trial_types']
        n_trials = block['n_trials']
        probs = []
        for tt in trial_types:
            p = self.design.trial_types[tt].get('probability')
            probs.append(p if p is not None else 1.0 / len(trial_types))
        probs = np.array(probs)
        probs = probs / probs.sum()
        return np.random.choice(trial_types, size=n_trials, p=probs).tolist()

    def _generate_trial(self, trial_type_name, trial_onset_time):
        """Generate events for one trial, returning list of event dicts."""
        trial_def = self.design.trial_types[trial_type_name]
        events = []
        for evt_def in trial_def['events']:
            onset_spec = evt_def['onset']
            if isinstance(onset_spec, (int, float)):
                relative_onset = float(onset_spec)
            else:
                relative_onset = self._sample_from_dist(onset_spec)
            evt_time = trial_onset_time + relative_onset
            events.append({
                'latency_s': evt_time,
                'latency': evt_time * self.sfreq,
                'type': evt_def['label'],
                'trial_type': trial_type_name,
            })
        return events

    def _sample_iti(self, iti_spec):
        """Sample an inter-trial interval."""
        return self._sample_from_dist(iti_spec)

    def _apply_covariates(self):
        """Add covariate columns to self.evts based on design specifications."""
        for cov_name, cov_spec in self.design.covariates.items():
            if cov_spec['kind'] == 'categorical':
                mapping = cov_spec['mapping']
                self.evts[cov_name] = self.evts['trial_type'].map(mapping)
            elif cov_spec['kind'] == 'continuous':
                affects = cov_spec['affects']
                mask = self.evts['type'] == affects
                n_affected = mask.sum()
                if 'dist' in cov_spec:
                    values = [self._sample_from_dist(cov_spec['dist']) for _ in range(n_affected)]
                    self.evts.loc[mask, cov_name] = values
                else:
                    self.evts[cov_name] = np.nan

    def _gaussian_response(self, time_array, onset, amplitude, width):
        """Generate a Gaussian bump on time_array centered at onset + 2*width."""
        mu = onset + 2 * width
        g = 1 / (width * np.sqrt(2 * np.pi)) * np.exp(-0.5 * ((time_array - mu) / width) ** 2)
        return g * amplitude

    def _generate_eeg(self):
        """Build the continuous EEG signal from kernels and events."""
        max_latency_s = self.evts['latency_s'].max() + 1.0
        if self._duration is not None:
            duration = max(self._duration, max_latency_s)
        else:
            duration = max_latency_s
        n_samples = int(duration * self.sfreq)
        self.time = np.arange(n_samples) / self.sfreq
        self.data = np.zeros(n_samples)
        self.samples = n_samples
        self.duration = duration

        if self.noise == 'brown':
            self._add_brown_noise()

        cov_scales = {}
        for cov_name, cov_spec in self.design.covariates.items():
            if cov_spec['kind'] == 'categorical' and 'amplitude_scale' in cov_spec:
                cov_scales[cov_name] = cov_spec['amplitude_scale']

        for _, evt in self.evts.iterrows():
            label = evt['type']
            if label not in self.design.kernels:
                continue
            onset_s = evt['latency_s']
            scale_factor = 1.0
            for cov_name, amp_scales in cov_scales.items():
                trial_type = evt['trial_type']
                if trial_type in amp_scales:
                    scale_factor *= amp_scales[trial_type]
            for cov_name, cov_spec in self.design.covariates.items():
                if cov_spec['kind'] == 'continuous' and cov_spec['affects'] == label:
                    if cov_name in evt and not pd.isna(evt[cov_name]):
                        slope = cov_spec.get('slope', 0)
                        scale_factor *= (1.0 + slope * evt[cov_name])
            for comp in self.design.kernels[label]:
                self.data += scale_factor * self._gaussian_response(
                    self.time, onset_s + comp['onset'], comp['amplitude'], comp['width'])

    def _add_brown_noise(self, scale=0.5):
        """Add 1/f^2 brown noise to self.data."""
        n = self.samples
        noise = np.random.randn(n + 1)
        freqs = np.fft.fftfreq(n + 1, 1 / self.sfreq)
        psd = 1 / np.sqrt(np.abs(freqs[1:]))
        noise_fft = np.fft.fft(noise[1:])
        noise_fft = noise_fft[:len(psd)]
        noise_fft *= psd
        noise = np.real(np.fft.ifft(noise_fft))
        self.data[:len(noise)] += noise * scale

    def simulate(self):
        """
        1. Generate trial sequence from blocks (randomized within block)
        2. For each trial, sample within-trial event onsets
        3. Build the events DataFrame with latency, type, and covariates
        4. Generate continuous EEG by summing kernels (modulated by covariates)
        """
        events_list = []
        current_time = 0.0

        for block in self.design.blocks:
            trial_sequence = self._generate_trial_sequence(block)
            for trial_type in trial_sequence:
                trial_events = self._generate_trial(trial_type, current_time)
                events_list.extend(trial_events)
                current_time = trial_events[-1]['latency_s'] + self._sample_iti(block['iti'])

        self.evts = pd.DataFrame(events_list)
        self._apply_covariates()
        self._generate_eeg()
        return self.data, self.evts

    def plot_datanpsd(self):
        """Plot the data and its power spectral density."""
        N = self.samples
        X = np.fft.fft(self.data) / N
        pwr = 10 * np.log10(2 * np.abs(X[:int(N/2)+1]) ** 2)
        frec = np.linspace(0, self.sfreq / 2, int(N/2)+1)

        fig, axs = plt.subplots(2, 1, figsize=(10, 6), tight_layout=True)

        axs[0].plot(self.time, self.data, lw=0.5)
        colors = {'fixation': 'r', 'stimulus': 'b', 'response': 'g'}
        used_labels = set()
        for _, evt in self.evts.iterrows():
            label = evt['type']
            c = colors.get(label, 'k')
            lbl = label if label not in used_labels else None
            axs[0].axvline(x=evt['latency_s'], color=c, linestyle='--', alpha=0.3, label=lbl)
            used_labels.add(label)
        axs[0].legend()
        axs[0].set_xlabel('Time (s)')
        axs[0].set_ylabel('Amplitude')
        axs[0].set_title('Data signal')

        idx = np.where(frec >= 100)[0]
        if len(idx) > 0:
            idx = idx[0]
        else:
            idx = len(frec)
        axs[1].plot(frec[:idx], pwr[:idx], lw=0.8)
        axs[1].set_xlabel('Frequency (Hz)')
        axs[1].set_ylabel('Power (dB)')
        axs[1].set_ylim([-100, 40])
        axs[1].set_title('Spectral Power')
        plt.show()

    def plot_kernel(self, event_label):
        """Plot the ERP kernel waveform and ITI distribution for a given event label."""
        if event_label not in self.design.kernels:
            print(f"No kernel defined for '{event_label}'")
            return

        components = self.design.kernels[event_label]
        # Build a 1-second time window starting before the first component
        first_width = components[0]['width']
        t_start = -2 * first_width
        t_end = t_start + 1.0
        times = np.linspace(t_start, t_end, 500)

        # Sum all kernel components
        waveform = np.zeros_like(times)
        for comp in components:
            waveform += self._gaussian_response(times, comp['onset'], comp['amplitude'], comp['width'])

        # Get ITI spec from the first block (they share the same ITI in most designs)
        iti_spec = self.design.blocks[0]['iti'] if self.design.blocks else None

        n_plots = 2 if iti_spec else 1
        fig, axs = plt.subplots(n_plots, 1, figsize=(6, 4 * n_plots), tight_layout=True)
        if n_plots == 1:
            axs = [axs]

        # Top: kernel waveform
        axs[0].plot(times, waveform, lw=0.8)
        axs[0].set_xlabel('Time (s)')
        axs[0].set_ylabel('Amplitude')
        axs[0].set_title(f'ERP kernel: {event_label}')

        # Bottom: ITI distribution
        if iti_spec:
            lims = iti_spec.get('lims', [0, 1])
            x = np.linspace(lims[0] - 0.1, lims[1] + 0.1, 500)
            dist = iti_spec.get('dist', 'uniform')
            if dist == 'uniform':
                pdf = uniform.pdf(x, loc=lims[0], scale=lims[1] - lims[0])
            elif dist == 'skewed':
                pdf = skewnorm.pdf(x, iti_spec.get('skew', 0),
                                   loc=lims[0], scale=iti_spec.get('scale', 0.1))
            else:
                pdf = np.zeros_like(x)
            axs[1].plot(x, pdf, lw=0.8)
            axs[1].set_xlabel('Time (s)')
            axs[1].set_ylabel('Probability')
            axs[1].set_title('ITI distribution')

        plt.show()

    def data_stats(self):
        """Display basic statistics for the data."""
        print(f'Mean:    {np.mean(self.data)}\nMedian:    {np.median(self.data)}\nVariance:  {np.var(self.data)}')


if __name__ == '__main__':
    print('Trying EEG Simulation...')
    sig = EEGSimulator(200, 500)
    # I am changing the 'weight' parameter for a more physiologicallty correlated way of thinking af event, 
    # which is like transitions between states. Now corresponding changes from one erp kernel to another
    # jwould be governed by a probability for that transstition to happen W_{12}
    # first attempt of using a transitions matrix to determine ISI's final check should be to study 
    # the 1st neightbourt distance distribution and try to reach similarty to empirical results.
    W_matrix = [[0, 0.45, 0.45, 0.1],[0.9, 0, 0 , .1],[0.9, 0, 0,.1], [.33,.33,.33,0]]
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


    sig.simulate(noise=None,erp_ker=kernels,w_matrix=W_matrix)
    sig.plot_response_idx(0)
    sig.plot_response_idx(1)
    sig.plot_response_idx(2)
    plt.show()
    sig.plot_datanpsd()
    sig.data_stats()
    sig.evts['latency'].diff().hist(bins=50)
    plt.show()
    print(sig.evts)


########
# TODO #
########
# 1.1 add linear modulation to one or several kernels, define how events will be chosen
# 1.2 
# 2. filter it as It was a real data
# 4. create ERP make a method to do so 
# 5. apply PyDeconv
# 6. compare scores on several conditions 
#     .Levels of noise 
#     . 
#     . 

# terminar ruido con features
# distribution of onsets and overlaping characterization
# add continuous modulation

###############
# Notes on SNR#
###############
# Shurui 2023 used four measures of SNR (Park(bootstrap), Maidhof, Hammer, M&P)
# Kristensen 2017 used a way to measure relative potencial relactions SNR(...), 
# SIR and SAR from "blind sources separation community" this is interesting 
# because she also measure the overlap with these measures
# Burns, Makeig 2013 used SNR and ROV (read more)
# Ozcam 2006 SNR_amp

# Bardy 2014 uses the condition number to estimate the error  from (Conte and Boor 1980)
# and then uses the intra calss correlation coeffitient ICC to asses the quality of the 
# convolution.
# Bardy also study how SOA affects condition number, in ralation to the jitter range 
########################################################################################