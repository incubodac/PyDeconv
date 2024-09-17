import numpy as np 
from matplotlib import pyplot as plt
from scipy.stats import skewnorm, uniform
import pandas as pd

class EEGSimulator:
    def __init__(self, duration, sample_rate):
        self.duration = duration
        self.sample_rate = sample_rate
        self.samples = int(duration * sample_rate)
        self.time = np.arange(0, duration, 1/sample_rate)
        self.data = np.zeros(self.time.shape)
        self.evts = pd.DataFrame(columns=['latency', 'type', 'categorical', 'continuous'])

        
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



    def simulate(self, noise='brown',erp_ker=None, isi = {'dist': 'uniform', 'lims': [100,400]} ,add_linear_mod = False):
        """Simulates the EEG data."""
        self.data = np.zeros(self.samples)
        if noise == 'brown':
            self.add_brown_noise()
        else:
            print('No noise added')
        
    
        # Retrieve all possible conditions from the ERP kernel dictionary
            if erp_ker is None:
                raise ValueError("erp_ker dictionary must be provided.")
        erp_conditions = list(erp_ker.keys())
        ker_erp_idx = [cond for cond in erp_conditions if isinstance(cond, (int, float))]
        weights = [erp_ker[cond]['weight'] for cond in ker_erp_idx]
        # Normalize weights (in case they don't sum to 1)
        weights = np.array(weights) / np.sum(weights)

        # Create a conditions_array with random 1s and 0s
        sample_size = 50
        # weight1= .5
        # weight2= .5
        # conditions_array = np.random.choice([0, 1],  size=sample_size, p=[weight1, weight2]).tolist()
        conditions_array = np.random.choice(ker_erp_idx, size=sample_size,p= weights).tolist()

        times = []
        samples = []
        for choice in conditions_array:
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
        

        times = np.cumsum(samples) # latencies for all events

        
        # add onsets to self.onsets
        self.onsets = times
        self.evts['latencies'] = times
        # add conditions to self.onsets
        self.conditions = conditions_array
        self.evts['type'] = conditions_array
        # if add_linear_mod:
        #     # currently linear modulation is hardcoded to be a truncated gaussian
        #     # intended to match saccade amplitude values in the linear region (1-3 degs)
        #     # as signals are in arbitrary units this is just for the sake of 
        #     mod_feature_values =  
        
        self.add_neural_responses(times, conditions_array,erp_ker=erp_ker)

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
        

if __name__ == '__main__':
    print('Trying EEG Simulation...')
    sig = EEGSimulator(200, 500)
    kernels = {
                0: {'onsets': [0, 0.19, 0.25], 'amplitudes': [0.1, -0.05, 0.04], 'widths': [0.05, 0.05, 0.07], 'weight':0.3},
                1: {'onsets': [0, 0.19, 0.25], 'amplitudes': [0.1, -0.05, 0.04], 'widths': [0.05, 0.05, 0.07], 'weight':0.3},
                2: {'onsets': [0, 0.19, 0.25], 'amplitudes': [0.1, -0.07, 0.04], 'widths': [0.05, 0.05, 0.07], 'weight':0.3},
                'modulation': {'ker_idx2mod': 1, 'mod': 'linear','dist': 'uniform', 'lims': [100, 600]}
            }
    sig.create_isi_pdf(2, sample_size=100, lims=[.1, .6], dist_type='skewed', mode=.3, skew=2, scale=.05)
    sig.create_isi_pdf(0, sample_size=100, lims=[.1, .6], dist_type='skewed', mode=.3, skew=2, scale=.05)
    sig.create_isi_pdf(1, sample_size=100, lims=[.1, .6], dist_type='uniform')
    # sig.combine_isi_pdf
    # sig.plot_isi_pdf(0)
    # sig.plot_isi_pdf(1)


    sig.simulate(noise=None,erp_ker=kernels)
    sig.plot_response_idx(0)
    sig.plot_response_idx(1)
    sig.plot_response_idx(2)
    plt.show()
    sig.plot_datanpsd()
    sig.data_stats()
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

# Bardy 2014 uses the condition number to estimate the error  from (Conte and Boor 1980)
# and then uses the intra calss correlation coeffitient ICC to asses the quality of the 
# convolution.
# Bardy also study how SOA affects condition number, in ralation to the jitter range 
########################################################################################