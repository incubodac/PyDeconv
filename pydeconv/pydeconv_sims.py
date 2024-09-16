import numpy as np 
from matplotlib import pyplot as plt
from scipy.stats import skewnorm, uniform

class EEGSimulator:
    def __init__(self, duration, sample_rate):
        self.duration = duration
        self.sample_rate = sample_rate
        self.samples = int(duration * sample_rate)
        self.time = np.arange(0, duration, 1/sample_rate)
        self.data = np.zeros(self.time.shape)
        self._set_isi_count = 0
        # self. = 
        
    def data_stats(self):
        """Display basic statistics for the data."""
        median = np.median(self.data)
        mean = np.mean(self.data)
        variance = np.var(self.data)
        print(f'Mean:    {mean} \nMedian:    {median}\nVariance:  {variance}')
        
    def add_brown_noise(self, scale=0.5):
        """Adds brown noise to the simulated EEG data."""
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
        """Return the Power Spectral Density (PSD) of the data."""
        N = self.samples
        X = np.fft.fft(self.data)/N
        psd = 10 * np.log10(2 * np.abs(X[:int(N/2)+1]) ** 2)
        freq = np.linspace(0, self.sample_rate/2, int(N/2)+1)
        return freq, psd
        
    def plot_datanpsd(self):
        """Plot the data and its power spectral density."""
        frec, pwr = self.get_psd()
        t = self.time
        fig, axs = plt.subplots(2, 1, figsize=(8, 6), tight_layout=True)
        
        # Plot the data signal
        axs[0].plot(t, self.data, lw=0.8)
        
        # Plot vertical lines for each event (different color for each event type)
        for onset, condition in zip(self.onsets, self.conditions):
            if condition == 0:
                axs[0].axvline(x=onset, color='r', linestyle='--', label='ERP 0 (event 0)')
            elif condition == 1:
                axs[0].axvline(x=onset, color='b', linestyle='--', label='ERP 1 (event 1)')
        
        # Optional: add a legend (only one label per event type)
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


    def simulate(self, noise='brown',erp_ker=None, isi = {'dist': 'uniform', 'lims': [100,400]} ,add_linear_mod = False):
        """Simulates the EEG data."""
        self.data = np.zeros(self.samples)
        if noise == 'brown':
            self.add_brown_noise()
        else:
            print('No noise added')
        
        
        
        N = 10
        isis = []
        # create array for the number of responses for each condition
        if erp_ker is None:
            # default test
            n_responses = [5,5]
        if isinstance(erp_ker, dict):
            count = 0
            for key in erp_ker.keys():
                if isinstance(key, int):
                    count += 1
            if count >0:
                # Generate an array of n random values
                isis = np.empty([count,])
                random_values = np.random.rand(count)
                
                # Scale the array so that the sum is N
                scaled_values = random_values / random_values.sum() * N
                
                    
                
        # Generate all times, amplitudes, widths, and conditions holistically
        # times = np.random.uniform(0, self.duration, size=n_responses[0]).tolist()
        # amplitudes = np.random.uniform(0.5, 1.5, size=n_responses).tolist()
        # widths = np.random.uniform(0.05, 0.2, size=n_responses).tolist()

        # Create a conditions_array with random 1s and 0s
        sample_size = 10
        weight1= .5
        weight2= .5
        conditions_array = np.random.choice([0, 1],  size=sample_size, p=[weight1, weight2]).tolist()

        times = []
        samples = []
        for choice in conditions_array:
            if choice == 0:
                # Retrieve the parameters for ISI_0 (skewnorm)
                isi_param = getattr(self, f'isi_{0}', None)
                if isi_param:
                    skew = isi_param['skew']
                    mean_shift1 = isi_param['lims'][0]
                    scale1 = isi_param['scale']
                    samples.append(skewnorm.rvs(skew, loc=mean_shift1, scale=scale1))
                else:
                    raise AttributeError("ISI_0 parameters not set.")
            elif choice == 1:
                isi_param = getattr(self, f'isi_{1}', None)
                if isi_param:
                    loc2 = isi_param['lims'][0]
                    scale2 = isi_param['scale']
                    skew = isi_param['skew']
                    samples.append(uniform.rvs(loc=loc2, scale=scale2))
                else:
                    raise AttributeError("ISI_1 parameters not set.")

        times = np.cumsum(samples)

        
        # add onsets to self.onsets
        self.onsets = times
        # add conditions to self.onsets
        self.conditions = conditions_array
        # if add_linear_mod:
        #     # currently linear modulation is hardcoded to be a truncated gaussian
        #     # intended to match saccade amplitude values in the linear region (1-3 degs)
        #     # as signals are in arbitrary units this is just for the sake of 
        #     mod_feature_values =  
        
        self.add_neural_responses(times, conditions_array,erp_ker=erp_ker)

        return self.data

    def gaussian_response(self, onset, amp, width):
        """Creates a Gaussian response starting at onset."""
        times = self.time
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
    
    def create_isi_pdf(self, kernel_idx, sample_size, lims=[.1, .6], dist_type='uniform', mode=.1, skew=0, scale=1):
        """Create and store ISI PDF parameters."""
        isi_params = {
            'type': dist_type,
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
            dist_type = isi_params['type']

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

# Example usage:
    # def plot_isi():
    
    # # def overlap_score(self):
    # #     # not implemente yet    
    # #     self.onsets 
    
    # # def get_erp(self,condition = None,baseline = None):
    # #     if condition is None:
            
    # #     return erp_data
    
    

if __name__ == '__main__':
    print('Trying EEG Simulation...')
    sig = EEGSimulator(200, 500)
    kernels = {
                0: {'onsets': [0, 0.19, 0.25], 'amplitudes': [0.1, -0.05, 0.04], 'widths': [0.05, 0.05, 0.07]},
                1: {'onsets': [0, 0.19, 0.25], 'amplitudes': [0.1, -0.07, 0.04], 'widths': [0.05, 0.05, 0.07]},
                'modulation': {'ker_idx2mod': 1, 'mod': 'linear','dist': 'uniform', 'lims': [100, 600]}
            }
    sig.create_isi_pdf(0, sample_size=100, lims=[.1, .6], dist_type='skewed', mode=.3, skew=2, scale=.05)
    sig.create_isi_pdf(1, sample_size=100, lims=[.1, .6], dist_type='uniform')
    # sig.combine_isi_pdf
    sig.plot_isi_pdf(0)
    sig.plot_isi_pdf(1)
    sig.simulate(noise=None)
    sig.plot_datanpsd()
    sig.data_stats()

# terminar ruido con features
# distribution of onsets and overlaping characterization
# add continuous modulation
