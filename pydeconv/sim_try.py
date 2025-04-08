import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import skewnorm, uniform
import pandas as pd
from scipy.signal import convolve , fftconvolve

# work in progress to attempt using convolution to simulate the data

class EEGSimulator:
    """
    Simulates EEG data with event-related potentials (ERPs) and noise.

    Attributes
    ----------
    duration : float
        Duration of the simulation in seconds.
    sample_rate : int
        Sampling rate in Hz.
    samples : int
        Number of samples in the simulation.
    time : numpy.ndarray
        Time array for the simulation.
    data : numpy.ndarray
        Simulated EEG data.
    evts : pandas.DataFrame
        DataFrame to store event-related information.
    onsets : numpy.ndarray or None
        Onsets of events in samples.
    conditions : list or None
        List of event conditions.
    erp_ker : dict or None
        Dictionary to store ERP kernel information.
    isi : dict or None
        Dictionary to store inter-stimulus interval (ISI) parameters.
        
    """

    def __init__(self, duration, sample_rate):
        """
        Initializes the EEGSimulator with the specified duration and sample rate.

        Parameters
        ----------
        duration : float
            Duration of the simulation in seconds.
        sample_rate : int
            Sampling rate in Hz.

        """
        self.duration = duration
        self.sample_rate = sample_rate
        self.samples = int(duration * sample_rate) 
        self.time = np.arange(0, duration, 1/sample_rate)
        self.data = np.zeros(self.time.shape)
        self.evts = pd.DataFrame(columns=['latency', 'type', 'categorical', 'continuous'])
        self.onsets = None
        self.conditions = None
        self.erp_ker = None
        self.isi = None

    def data_stats(self):
        """
        Prints basic statistics of the simulated EEG data.
        """
        median = np.median(self.data)
        mean = np.mean(self.data)
        variance = np.var(self.data)
        print(f'Mean:    {mean} \nMedian:    {median}\nVariance:  {variance}')
        
    def add_brown_noise(self, scale=0.5):
        """
        Adds brown noise to the simulated EEG data.

        Parameters
        ----------
        scale : float
            Scale factor for the noise.
        """
        noise = np.random.randn(self.samples+1)
        freqs = np.fft.fftfreq(self.samples+1, 1 / self.sample_rate)
        psd = 1 / np.sqrt(np.abs(freqs[1:]))
        noise = np.fft.fft(noise[1:])
        noise = noise[:len(psd)]
        noise *= psd
        noise = np.real(np.fft.ifft(noise))
        noise *= scale
        self.data += noise
        
    def get_psd(self):
        """
        Computes the power spectral density (PSD) of the simulated EEG data.
        
        Returns
        -------
        freq : numpy.ndarray
            Frequency array.
        psd : numpy.ndarray
            Power spectral density array.
        """
        N = self.samples
        X = np.fft.fft(self.data)/N
        psd = 10 * np.log10(2 * np.abs(X[:int(N/2)+1]) ** 2)
        freq = np.linspace(0, self.sample_rate/2, int(N/2)+1)
        return freq, psd
    
    def plot_datanpsd(self):
        """
        Plots the simulated EEG data and its power spectral density (PSD).
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
            
            # Plot the vertical line for this event
            axs[0].axvline(x=onset, color=color, linestyle='--', label=f'ERP {condition} (event {condition})')
            
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

    def gaussian_response(self, onset, amp, width, short_time=False):
        """
        Computes a Gaussian response function.

        Returns a Gaussian response starting at onset with a given amplitude and width.

        Parameters
        ----------
        onset : float
            Onset time of the response.
        amp : float
            Amplitude of the response.
        width : float
            Width of the Gaussian response.
        short_time : bool, optional
            If True, uses the short time array for the response. Default is False.

        Returns
        -------
        numpy.ndarray
            Gaussian response array.

        """
        # Check if short_time is provided
        # and use it if available
        # else use the full time array
        # to compute the response
        if short_time:
            times = short_time
        else:
            times = self.time
        
        # Compute the Gaussian response
        # using the formula: A * exp(-((x - mu) / sigma) ** 2)
        # where A is the amplitude, mu is the mean (onset),
        # and sigma is the standard deviation (width)
        # The width parameter is used to control the spread of the Gaussian
        # response.
        # The mean (mu) is set to the onset time plus 2 times the width
        # to ensure the Gaussian is centered around the onset.
        # The Gaussian response is normalized by dividing by the width
        # and multiplying by the amplitude.
        # The response is then scaled by the amplitude.
        mu = onset + 2 * width
        gaussian = 1 / (width * np.sqrt(2 * np.pi)) * np.exp(-0.5 * ((times - mu) / width) ** 2)
        gaussian *= amp
        return gaussian

    def simulate(self, noise='brown', erp_ker=None, isi={'dist': 'uniform', 'lims': [100, 400]}, w_matrix=None, add_linear_mod=False):
        """Simulates the EEG data."""
        self.data = np.zeros(self.samples)
        if noise == 'brown':
            self.add_brown_noise()
        else:
            print('No noise added')
        
        if erp_ker is None:
            raise ValueError("erp_ker dictionary must be provided.")
        else:
            erp_conditions = list(erp_ker.keys())
        
        ker_erp_idx = [cond for cond in erp_conditions if isinstance(cond, (int, float))]
        weights = [erp_ker[cond]['weight'] for cond in ker_erp_idx]
        
        for i, row in enumerate(w_matrix):
            w_matrix[i] = np.array(row) / np.sum(row)

        left_padding = ker_erp_idx[0]
        sample_size = self.samples//10 # hardcoded for now amount of events to model
        current_state = 0
        states_sequence = [current_state]

        for _ in range(sample_size - 1):
            next_state = np.random.choice(ker_erp_idx, p=w_matrix[current_state])
            states_sequence.append(next_state)
            current_state = next_state

        times = []
        samples = []
        for choice in states_sequence:
            isi_param = getattr(self, f'isi_{choice}', None)

            if isi_param:
                if isi_param['dist'] == 'skewed':
                    skew = isi_param['skew']
                    loc = isi_param['lims'][0]
                    scale = isi_param['scale']
                    sample = skewnorm.rvs(skew, loc=loc, scale=scale)
                    samples.append(max(0, sample))  # Ensure positive value
                elif isi_param['dist'] == 'uniform':
                    loc = isi_param['lims'][0]
                    scale = isi_param['lims'][1] - loc
                    samples.append(uniform.rvs(loc=loc, scale=scale))
                else:
                    raise ValueError(f"Unsupported distribution: {isi_param['dist']}")
            else:
                raise AttributeError(f"ISI parameters not set for condition {choice}.")
        
        samples.insert(0, left_padding)
        cumulative_time = np.cumsum(samples)
        times = cumulative_time[cumulative_time <= self.duration]
        states_sequence = states_sequence[:len(times)]
        self.evts['latency'] = times * self.sample_rate
        samples.insert(0, left_padding)
        times = np.cumsum(samples)

        # Ensure times does not exceed self.duration
        times = times[times <= self.duration]

        self.onsets = times[:-1] * self.sample_rate # onsets in samples
        self.conditions = states_sequence
        self.evts['type'] = states_sequence
        
        id_kers = [kid for kid in erp_ker.keys() if isinstance(kid, int)]
        for ker in id_kers:
            onset = self.onsets[np.array(states_sequence) == ker].astype(int)
            print(f"Kernel ID: {ker}")
            print(f"States Sequence: {states_sequence}")
            print(f"Onsets: {self.onsets}")
            print(f"Filtered Onsets for Kernel {ker}: {onset}")
            
            onset_sticks = np.zeros(len(self.time))
            if len(onset) > 0:
                valid_onsets = onset[onset < len(self.time)]  # Ensure onsets are within bounds
                onset_sticks[valid_onsets] = 1
            print(f"Onset Sticks for Kernel {ker}: {onset_sticks}")
            self.add_neural_responses_convolution(onset_sticks, states_sequence,erp_ker=erp_ker)


        return self.data

    def add_neural_responses_convolution(self, erp_onsets, states_sequence,erp_ker=None):
        if erp_ker is None:
            self.erp_ker = {
                0: {'onsets': [0, 0.19, 0.25], 'amplitudes': [0.1, -0.05, 0.04], 'widths': [0.05, 0.05, 0.07]},
                1: {'onsets': [0, 0.19, 0.25], 'amplitudes': [0.1, -0.07, 0.04], 'widths': [0.05, 0.05, 0.07]}
            }#this is a default kernel but is not properly implemented yet
        else:
            self.erp_ker = erp_ker


        for ker_id in [key for key in self.erp_ker.keys() if isinstance(key, int)]:
            kernel = self.get_neural_response(0, ker_id, self.time)
            kernel_onset = erp_onsets[states_sequence == ker_id]
            print(f"Kernel ID: {ker_id}")
            print(f"ERP Onsets: {erp_onsets}")
            print(f"Kernel Onset: {kernel_onset}")
            
            if len(kernel_onset) == 0:
                print(f"Warning: No onsets found for Kernel ID {ker_id}. Skipping convolution.")
                continue  # Skip this kernel if no onsets are found

            response = fftconvolve(kernel, kernel_onset, mode='same')
            self.data += response

    def get_neural_response(self, onset, kernel_idx, times):
        """Compute the neural response for a given onset and kernel index."""
        if kernel_idx not in self.erp_ker:
            raise ValueError(f"Kernel index {kernel_idx} not found in ERP kernel.")

        response = np.zeros_like(times)  # Ensure it has the correct shape
        
        kernel = self.erp_ker[kernel_idx]
        
        if len(kernel['onsets']) == 0:
            print(f"Warning: Kernel {kernel_idx} has no onsets. Returning zero response.")
            return response  # Ensure it has the correct shape

        for resp_onset, resp_ampli, resp_width in zip(kernel['onsets'], kernel['amplitudes'], kernel['widths']):
            shifted_onset = resp_onset + onset
            response += self.gaussian_response(shifted_onset, resp_ampli, resp_width, short_time=times)

        print(f"Generated response shape: {response.shape}")  # Debugging
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

            isi_params = getattr(self, f'isi_{kernel_idx}')
            lims = isi_params['lims']
            sample_size = isi_params['sample_size']
            scale = isi_params['scale']
            mode = isi_params['mode']
            skew = isi_params['skew']
            dist_type = isi_params['dist']

            x = np.linspace(lims[0] - .1, lims[1] + .1, sample_size)

            if dist_type == 'uniform':
                pdf = uniform.pdf(x, loc=lims[0], scale=lims[1] - lims[0])
            elif dist_type == 'skewed':
                pdf = skewnorm.pdf(x, skew, loc=mode, scale=scale)
            else:
                raise ValueError(f"Unsupported distribution type: {dist_type}")

            plt.plot(x, pdf, label=f'ISI {kernel_idx}')
            plt.title(f'ISI PDF for kernel {kernel_idx}')
            plt.xlabel('ISI')
            plt.ylabel('Probability Density')
            plt.legend()
            plt.show()
        else:
            print(f"Simulation object does not have an ISI for kernel {kernel_idx}")
    
    def plot_response_idx(self, kernel_idx):
        if not hasattr(self, f'isi_{kernel_idx}'):
            print(f"None ISI params for kernel {kernel_idx}")
            return
        
        isi_params = getattr(self, f'isi_{kernel_idx}')
        lims = isi_params['lims']
        sample_size = isi_params['sample_size']
        scale = isi_params['scale']
        mode = isi_params['mode']
        skew = isi_params['skew']
        dist_type = isi_params['dist']
        
        x = np.linspace(lims[0] - 0.1, lims[1] + 0.1, sample_size)
        if dist_type == 'uniform':
            pdf = uniform.pdf(x, loc=lims[0], scale=lims[1] - lims[0])
        elif dist_type == 'skewed':
            pdf = skewnorm.pdf(x, skew, loc=mode, scale=scale)
        else:
            raise ValueError(f"Unsupported distribution type: {dist_type}")
        
        fig, axs = plt.subplots(2, 1, figsize=(6, 8), tight_layout=True)
        
        onset = 2 * self.erp_ker[kernel_idx]['widths'][0]
        times = np.linspace(-onset, -onset + 1, 500)
        neural_response = self.get_neural_response(0, kernel_idx, times)
        
        axs[0].plot(times, neural_response, lw=0.8)
        axs[0].set_xlabel('Time (s)')
        axs[0].set_ylabel('Amplitude')
        axs[0].set_title(f'ERP response for kernel {kernel_idx}')
        
        axs[1].plot(x, pdf, lw=0.8)
        axs[1].set_xlabel('Time (s)')
        axs[1].set_ylabel('Probability')
        axs[1].set_title(f'ISI distribution for kernel {kernel_idx}')
        
        plt.show()

    def plot_all_responses(self):
        print('not implemented')

if __name__ == '__main__':
    print('Trying EEG Simulation...')
    sig = EEGSimulator(200, 500)
    # I am changing the 'weight' parameter for a more physiologically correlated way of thinking an event, 
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
    
    
    