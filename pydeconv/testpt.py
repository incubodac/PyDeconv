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
        
    def data_stats(self):
        """Display basic statistics for the data."""
        median = np.median(self.data)
        mean = np.mean(self.data)
        variance = np.var(self.data)
        print(f'Mean:    {mean} \nMedian:    {median}\nVariance:  {variance}')
        
    def add_brown_noise(self, scale=0.5):
        """Adds Brown noise to the simulated EEG data."""
        noise = np.random.randn(self.samples)
        freqs = np.fft.fftfreq(self.samples, 1 / self.sample_rate)
        
        # Handle zero frequency separately to avoid division by zero
        psd = np.zeros_like(freqs)
        psd[1:] = 1 / np.sqrt(np.abs(freqs[1:]))
        
        noise_fft = np.fft.fft(noise)
        noise_fft[1:] *= psd[1:]  # Apply PSD to noise
        noise = np.fft.ifft(noise_fft).real
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
        freq, pwr = self.get_psd()
        t = self.time
        fig, axs = plt.subplots(2, 1, figsize=(8, 6), tight_layout=True)
        axs[0].plot(t, self.data, lw=0.8)
        
        if hasattr(self, 'onsets'):
            for onset in self.onsets:
                axs[0].axvline(x=onset, color='r', linestyle='--', label='event')
            axs[0].legend()

        axs[0].set_xlabel('Time (s)')
        axs[0].set_ylabel('Amplitude')
        axs[0].set_title('EEG Data')
        
        idx = np.where(freq >= 100)[0][0]
        axs[1].plot(freq[:idx], pwr[:idx], lw=0.8)
        axs[1].set_xlabel('Frequency (Hz)')
        axs[1].set_ylim([-100, 40])
        axs[1].set_ylabel('Power (dB)')
        axs[1].set_title('Power Spectral Density')
        plt.show()

    def simulate(self, noise=None, erp_ker=None, isi={'dist': 'uniform', 'lims': [100, 400]}, add_linear_mod=False):
        """Simulate EEG data."""
        self.data = np.zeros(self.samples)
        
        if noise == 'brown':
            self.add_brown_noise()
        else:
            print('No noise added')

        # Simulating neural events
        N = 10
        isis = []
        
        if isinstance(erp_ker, dict):
            count = len([key for key in erp_ker if isinstance(key, int)])
            if count > 0:
                isis = np.empty([count])
                random_values = np.random.rand(count)
                scaled_values = random_values / random_values.sum() * N
        
        # Create conditions array with random 0s and 1s
        sample_size = 10
        weight1 = 0.5
        weight2 = 0.5
        conditions_array = np.random.choice([0, 1], size=sample_size, p=[weight1, weight2]).tolist()

        times, samples = [], []
        for choice in conditions_array:
            isi_param = getattr(self, f'isi_{choice}', None)
            if isi_param:
                if choice == 0:
                    skew = isi_param['skew']
                    mean_shift1 = isi_param['lims'][0]
                    scale1 = isi_param['scale']
                    samples.append(skewnorm.rvs(skew, loc=mean_shift1, scale=scale1))
                elif choice == 1:
                    loc2 = isi_param['lims'][0]
                    scale2 = isi_param['scale']
                    samples.append(uniform.rvs(loc=loc2, scale=scale2))
            else:
                raise AttributeError(f"ISI_{choice} parameters not set.")

        times = np.cumsum(samples)

        self.onsets = times
        self.conditions = conditions_array
        self.add_neural_responses(times, conditions_array, erp_ker=erp_ker)

        return self.data

    def gaussian_response(self, onset, amp, width):
        """Creates a Gaussian response starting at the specified onset."""
        mu = onset + 2 * width
        return amp * (1 / (width * np.sqrt(2 * np.pi)) * np.exp(-0.5 * ((self.time - mu) / width) ** 2))

    def add_neural_responses(self, erp_onsets, conditions_array, erp_ker=None):
        """Add neural responses to the EEG signal based on event-related potential (ERP) kernels."""
        if erp_ker is None:
            self.erp_ker = {
                0: {'onsets': [0, 0.19, 0.25], 'amplitudes': [0.1, -0.05, 0.04], 'widths': [0.05, 0.05, 0.07]},
                1: {'onsets': [0, 0.19, 0.25], 'amplitudes': [0.1, -0.07, 0.04], 'widths': [0.05, 0.05, 0.07]}
            }
        else:
            self.erp_ker = erp_ker

        response = np.zeros(len(self.time))
        conditions_array = [int(cond) for cond in conditions_array]

        for onset, cond in zip(erp_onsets, conditions_array):
            if cond in self.erp_ker:
                for resp in range(len(self.erp_ker[cond]['onsets'])):
                    resp_onset = self.erp_ker[cond]['onsets'][resp] + onset
                    resp_ampli = self.erp_ker[cond]['amplitudes'][resp]
                    resp_width = self.erp_ker[cond]['widths'][resp]
                    response += self.gaussian_response(resp_onset, resp_ampli, resp_width)

        self.data += response
    
    def create_isi_pdf(self, kernel_idx, sample_size, lims=[100, 600], dist_type='uniform', mode=100, skew=0, scale=1):
        """Create and store ISI PDF parameters."""
        isi_params = {
            'type': dist_type,
            'sample_size': sample_size,
            'lims': lims,
            'mode': mode,
            'skew': skew,
            'scale': scale
        }
        setattr(self, f'isi_{kernel_idx}', isi_params)

    def plot_isi_pdf(self, kernel_idx):
        """Plot the ISI PDF for a given kernel index."""
        if hasattr(self, f'isi_{kernel_idx}'):
            isi_params = getattr(self, f'isi_{kernel_idx}')
            lims, sample_size, scale, mode, skew, dist_type = isi_params.values()
            x = np.linspace(lims[0] - 200, lims[1] + 200, sample_size)

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
            print(f"ISI parameters for kernel {kernel_idx} not found.")

# Example usage:
if __name__ == '__main__':
    print('Simulating EEG data...')
    sim = EEGSimulator(8000, 500)
    kernels = {
        0: {'onsets': [0, 0.19, 0.25], 'amplitudes': [0.1, -0.05, 0.04], 'widths': [0.05, 0.05, 0.07]},
        1: {'onsets': [0, 0.19, 0.25], 'amplitudes': [0.1, -0.07, 0.04], 'widths': [0.05, 0.05, 0.07]}
    }
    sim.create_isi_pdf(0, 10, lims=[100, 400], dist_type='skewed', mode=250, skew=4, scale=40)
    sim.create_isi_pdf(1, 10, lims=[100, 400], dist_type='uniform', scale=40)
    sim.simulate(erp_ker=kernels)
    sim.plot_datanpsd()
    sim.plot_isi_pdf(0)
    sim.plot_isi_pdf(1)
    sim.data_stats()
