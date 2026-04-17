import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import fftconvolve, windows
import pandas as pd


class ERPKernel:
    """A single ERP component defined by peak latency, width, and shape.

    Parameters
    ----------
    sfreq : float
        Sampling frequency in Hz.
    peak_latency : float
        Peak latency in seconds relative to event onset.
    amplitude : float
        Peak amplitude (arbitrary units).
    width : float
        Width parameter in seconds (std for Gaussian, half-width for Hanning).
    shape : str
        Kernel shape: ``'gaussian'`` or ``'hanning'``.
    label : str or None
        Optional human-readable label (e.g. ``'P1'``, ``'N170'``).
    """

    def __init__(self, sfreq: float, peak_latency: float, amplitude: float,
                 width: float, shape: str = 'gaussian', label: str | None = None):
        self.sfreq = sfreq
        self.peak_latency = peak_latency
        self.amplitude = amplitude
        self.width = width
        self.shape = shape
        self.label = label or f"{shape}@{peak_latency:.3f}s"
        self._build()

    def _build(self):
        """Compute the discrete kernel waveform."""
        # Kernel spans from 0 to peak_latency + 4*width
        t_end = self.peak_latency + 4 * self.width
        n_samples = max(int(np.ceil(t_end * self.sfreq)), 1)
        self.time = np.arange(n_samples) / self.sfreq

        if self.shape == 'gaussian':
            self.waveform = self.amplitude * np.exp(
                -0.5 * ((self.time - self.peak_latency) / self.width) ** 2)
        elif self.shape == 'hanning':
            half_w = int(np.round(self.width * self.sfreq))
            win_len = 2 * half_w + 1
            han = windows.hann(win_len)
            center = int(np.round(self.peak_latency * self.sfreq))
            self.waveform = np.zeros(n_samples)
            start = max(center - half_w, 0)
            end = min(center + half_w + 1, n_samples)
            w_start = start - (center - half_w)
            w_end = w_start + (end - start)
            self.waveform[start:end] = self.amplitude * han[w_start:w_end]
        else:
            raise ValueError(f"Unsupported shape '{self.shape}'. Use 'gaussian' or 'hanning'.")

    def plot(self, ax=None):
        """Plot the kernel waveform.

        Parameters
        ----------
        ax : matplotlib.axes.Axes or None
            Axes to plot on. If None, a new figure is created.

        Returns
        -------
        ax : matplotlib.axes.Axes
        """
        if ax is None:
            _, ax = plt.subplots(figsize=(6, 3), tight_layout=True)
        ax.plot(self.time * 1e3, self.waveform, lw=1.2)
        ax.axvline(self.peak_latency * 1e3, color='grey', ls='--', lw=0.8, alpha=0.6)
        ax.set_xlabel('Time (ms)')
        ax.set_ylabel('Amplitude')
        ax.set_title(self.label)
        return ax


class CompoundKernel:
    """Multiple ERP components summed into one kernel.

    Parameters
    ----------
    sfreq : float
        Sampling frequency in Hz.
    """

    def __init__(self, sfreq: float):
        self.sfreq = sfreq
        self.components: list[ERPKernel] = []
        self.waveform: np.ndarray | None = None
        self.time: np.ndarray | None = None

    def add(self, peak_latency: float, amplitude: float, width: float,
            shape: str = 'gaussian', label: str | None = None) -> 'CompoundKernel':
        """Add a component to the compound kernel.

        Parameters
        ----------
        peak_latency : float
            Peak latency in seconds.
        amplitude : float
            Peak amplitude.
        width : float
            Width in seconds.
        shape : str
            ``'gaussian'`` or ``'hanning'``.
        label : str or None
            Component label.

        Returns
        -------
        self : CompoundKernel
            For method chaining.
        """
        self.components.append(
            ERPKernel(self.sfreq, peak_latency, amplitude, width, shape, label))
        self._rebuild()
        return self

    def _rebuild(self):
        """Recompute the summed waveform from all components."""
        max_len = max(len(k.waveform) for k in self.components)
        self.time = np.arange(max_len) / self.sfreq
        self.waveform = np.zeros(max_len)
        for k in self.components:
            self.waveform[:len(k.waveform)] += k.waveform

    def plot(self, show_components: bool = True):
        """Plot the compound kernel, optionally with individual components.

        Parameters
        ----------
        show_components : bool
            If True, overlay each component as a thin dashed line.
        """
        fig, ax = plt.subplots(figsize=(7, 3.5), tight_layout=True)
        ax.plot(self.time * 1e3, self.waveform, 'k', lw=1.5, label='sum')
        if show_components:
            for k in self.components:
                ax.plot(k.time * 1e3, k.waveform, ls='--', lw=0.8, alpha=0.6, label=k.label)
        ax.set_xlabel('Time (ms)')
        ax.set_ylabel('Amplitude')
        ax.set_title('Compound ERP Kernel')
        ax.legend(fontsize=8)
        plt.show()


class EEGSimulator:
    """Generate continuous EEG by convolving event stick functions with ERP kernels.

    Parameters
    ----------
    sfreq : float
        Sampling frequency in Hz.
    duration : float
        Total signal duration in seconds.
    """

    def __init__(self, sfreq: float, duration: float):
        self.sfreq = sfreq
        self.duration = duration
        self.n_samples = int(duration * sfreq)
        self.time = np.arange(self.n_samples) / sfreq
        self.data = np.zeros(self.n_samples)
        self.kernels: dict[str, CompoundKernel | ERPKernel] = {}
        self.event_sticks: dict[str, np.ndarray] = {}

    def add_kernel(self, name_or_dict, kernel=None):
        """Register one or more named kernels for convolution.

        Parameters
        ----------
        name_or_dict : str or dict
            Either a single event label (str) paired with ``kernel``, or a
            dict mapping event labels to kernels,
            e.g. ``{'fixation': k1, 'saccade': k2}``.
        kernel : CompoundKernel, ERPKernel, or None
            Required when ``name_or_dict`` is a str. Ignored when a dict is passed.
        """
        if isinstance(name_or_dict, dict):
            self.kernels.update(name_or_dict)
        else:
            if kernel is None:
                raise ValueError("A kernel must be provided when name is a string.")
            self.kernels[name_or_dict] = kernel

    def set_events(self, events: pd.DataFrame):
        """Build stick functions from an events DataFrame.

        Parameters
        ----------
        events : pandas.DataFrame
            Must contain ``'latency'`` (in samples) and ``'type'`` columns.
        """
        self.events = events.copy()
        self.event_sticks = {}
        for evt_type in events['type'].unique():
            stick = np.zeros(self.n_samples)
            lats = events.loc[events['type'] == evt_type, 'latency'].values.astype(int)
            valid = lats[(lats >= 0) & (lats < self.n_samples)]
            stick[valid] = 1.0
            self.event_sticks[evt_type] = stick

    def simulate(self) -> np.ndarray:
        """Convolve each stick function with its kernel and sum into self.data.

        Returns
        -------
        data : numpy.ndarray
            The simulated EEG signal (n_samples,).
        """
        self.data = np.zeros(self.n_samples)
        for name, kernel in self.kernels.items():
            if name not in self.event_sticks:
                continue
            stick = self.event_sticks[name]
            convolved = fftconvolve(stick, kernel.waveform, mode='full')[:self.n_samples]
            self.data += convolved
        return self.data

    # ------ noise methods ------

    def add_noise(self, colour: str = 'brown', scale: float = 0.5):
        """Add coloured noise to the signal.

        Parameters
        ----------
        colour : str
            Noise colour: ``'white'``, ``'pink'``, or ``'brown'``.
        scale : float
            Amplitude scaling factor (std of the resulting noise).
        """
        exponents = {'white': 0, 'pink': 0.5, 'brown': 1.0}
        exp = exponents.get(colour.lower())
        if exp is None:
            raise ValueError(f"Unknown noise colour '{colour}'. Use 'white', 'pink', or 'brown'.")

        if colour.lower() == 'white':
            self.data += np.random.randn(self.n_samples) * scale
            return

        freqs = np.fft.rfftfreq(self.n_samples, d=1 / self.sfreq)
        freqs[0] = 1  # avoid division by zero
        spectrum = np.random.randn(len(freqs)) + 1j * np.random.randn(len(freqs))
        spectrum /= freqs ** exp
        noise = np.fft.irfft(spectrum, n=self.n_samples)
        noise = noise / np.std(noise) * scale
        self.data += noise

    # ------ plotting ------

    def plot(self):
        """Plot the simulated signal with event markers and PSD."""
        N = self.n_samples
        X = np.fft.rfft(self.data) / N
        pwr = 10 * np.log10(2 * np.abs(X) ** 2 + 1e-30)
        frec = np.fft.rfftfreq(N, d=1 / self.sfreq)

        fig, axs = plt.subplots(2, 1, figsize=(10, 6), tight_layout=True)

        # Time-domain signal
        axs[0].plot(self.time, self.data, lw=0.5)
        colors = plt.cm.tab10.colors
        used = set()
        if hasattr(self, 'events'):
            for _, evt in self.events.iterrows():
                etype = evt['type']
                c = colors[hash(etype) % len(colors)]
                lbl = etype if etype not in used else None
                lat_s = evt['latency'] / self.sfreq
                axs[0].axvline(x=lat_s, color=c, ls='--', alpha=0.3, label=lbl)
                used.add(etype)
            axs[0].legend(fontsize=8)
        axs[0].set_xlabel('Time (s)')
        axs[0].set_ylabel('Amplitude')
        axs[0].set_title('Simulated EEG')

        # PSD (up to 100 Hz)
        idx = np.searchsorted(frec, 100)
        axs[1].plot(frec[:idx], pwr[:idx], lw=0.8)
        axs[1].set_xlabel('Frequency (Hz)')
        axs[1].set_ylabel('Power (dB)')
        axs[1].set_ylim([-100, 40])
        axs[1].set_title('Power Spectral Density')

        plt.show()
