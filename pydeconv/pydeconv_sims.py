import numpy as np 
from matplotlib import pyplot as plt

class EEGSimulator:
    def __init__(self, duration, sample_rate):
        self.duration = duration
        self.sample_rate = sample_rate
        self.samples = int(duration * sample_rate)
        self.time = np.arange(0, duration, 1/sample_rate)
        self.data = np.zeros(self.time.shape)
        # self. = 
        
    def data_stats(self):
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
        N = self.samples
        X = np.fft.fft(self.data)/N
        psd = 10 * np.log10(2 * np.abs(X[:int(N/2)+1]) ** 2)
        freq = np.linspace(0, self.sample_rate/2, int(N/2)+1)
        return freq, psd
        
    def plot_datanpsd(self):
        frec, pwr = self.get_psd()
        t = self.time
        fig, axs = plt.subplots(2, 1, figsize=(8, 6), tight_layout=True)
        axs[0].plot(t, self.data, lw=0.8)
        for onset in self.onsets:
            axs[0].axvline(x=onset, color='r', linestyle='--', label='event')

        # Optional: add a legend
        axs[0].legend()
        axs[0].set_xlabel('Time (s)')
        axs[0].set_ylabel('Amplitude')
        axs[0].set_title('Data signal')
        
        idx = np.where(frec >= 100)[0][0]
        axs[1].plot(frec[:idx], pwr[:idx], lw=0.8)
        axs[1].set_xlabel('Frequency (Hz)')
        axs[1].set_ylim([-100,40])
        axs[1].set_ylabel('Power (dB)')
        axs[1].set_title('Spectral Power')
        plt.show()

    def simulate(self, noise='brown'):
        """Simulates the EEG data."""
        self.data = np.zeros(self.samples)
        if noise == 'brown':
            self.add_brown_noise()
        else:
            print('No noise added')
        
        
        n_responses = 5
        # Generate all times, amplitudes, widths, and conditions holistically
        times = np.random.uniform(0, self.duration, size=n_responses).tolist()
        # amplitudes = np.random.uniform(0.5, 1.5, size=n_responses).tolist()
        # widths = np.random.uniform(0.05, 0.2, size=n_responses).tolist()

        # Create a conditions_array with random 1s and 0s
        conditions_array = np.random.choice([0, 1], size=n_responses).tolist()

        
        # add onsets to self.onsets
        self.onsets = times
        # add conditions to self.onsets
        self.conditions = conditions_array

        
        self.add_neural_responses(times, conditions_array)

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
    
    def overlap_score(self):
        
        self.onsets 

if __name__ == '__main__':
    print('Trying EEG Simulation...')
    sig = EEGSimulator(80, 500)
    kernels = {
                0: {'onsets': [0, 0.19, 0.25], 'amplitudes': [0.1, -0.05, 0.04], 'widths': [0.05, 0.05, 0.07]},
                1: {'onsets': [0, 0.19, 0.25], 'amplitudes': [0.1, -0.07, 0.04], 'widths': [0.05, 0.05, 0.07]}
            }
    # sig.add_modulation()
    sig.simulate(noise=None)
    sig.data_stats()
    sig.plot_datanpsd()
