import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import fftconvolve, windows
import pandas as pd
from typing import Any, Callable, Sequence
import inspect

ISISampler = Callable[[pd.Series], int]


def build_uniform_isi_sampler(
    width: int,
    offset: int = 0,
    *,
    rng: np.random.Generator | None = None,
) -> ISISampler:
    """Build an ISI sampler that draws uniformly from [offset, offset + width]."""
    rng = rng or np.random.default_rng()

    def _sample(_row: pd.Series) -> int:
        return int(rng.integers(offset, offset + width + 1))

    return _sample

def build_gamma_isi_sampler(mean: int, scale: int = 1, offset: int = 0) -> ISISampler:
    """
    Build an ISI sampler that draws from a gamma distribution offseted by offset.
    
    Parameters
    ----------
    mean : int
        mean of the gamma distribution.
    scale : int
        scale of the gamma distribution. Equivalent to a rate value of 1/scale.
    offset : int
        offset in .samples used to shift the resulting value.
    """
    rng = rng or np.random.default_rng()

    def _sample(_row: pd.Series) -> int:
        return int(offset + rng.gamma(mean, scale))

    return _sample

def build_constant_isi_sampler(value: int = 0) -> ISISampler:
    """
    Build an ISI sampler that always returns value for an ISI.
    
    Parameters
    ----------
    value : int
        the ISI value it'll always return.
    """
    rng = rng or np.random.default_rng()

    def _sample(_row: pd.Series) -> int:
        return int(value)

    return _sample


def assign_event_latencies(events: pd.DataFrame, sampler: ISISampler) -> pd.DataFrame:
    """
    Assign cumulative latencies by sampling an ISI offset per row.
    
    Parameters
    ----------
    events : pd.DataFrame
        A dataframe having one row per event
    sampler : ISISampler
        A function that generates an ISI value in samples between each event
    """
    out = events.copy()
    latencies: list[int] = []
    for _, row in out.iterrows():
        delta = sampler(row)
        latencies.append(delta if not latencies else latencies[-1] + delta)
    out["latency"] = latencies
    return out


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
    modifier : Callable[..., float]
        Function called to modify the amplitude of the kernel.
    shape : str
        Kernel shape: ``'gaussian'`` or ``'hanning'``.
    label : str or None
        Optional human-readable label (e.g. ``'P1'``, ``'N170'``).
    """

    def __init__(self, sfreq: float, peak_latency: float, amplitude: float,
                 width: float, modifier: Callable[..., float] = lambda: 0.0, shape: str = 'gaussian', label: str | None = None):
        self.sfreq = sfreq
        self.peak_latency = peak_latency
        self.amplitude = amplitude
        self.width = width
        self.shape = shape
        self.label = label or f"{shape}@{peak_latency:.3f}s"

        self.modifying_variables = set()
        for name, _ in inspect.signature(modifier).parameters.items():
            self.modifying_variables.add(name)

        self.modifier = modifier
        if 'kwargs' not in self.modifying_variables:
            def wrapper(**kwargs):
                filtered = {k: v for k, v in kwargs.items() if k in self.modifying_variables}
                return modifier(**filtered)
            
            self.modifier = wrapper

        self._build()

    def _build(self, **kwargs):
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
        
        if all([v in kwargs for v in self.modifying_variables]):
            self.waveform *= (1 + self.modifier(**kwargs))

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

    def __init__(self, name: str, sfreq: float):
        self.name = name
        self.sfreq = sfreq
        self.components: list[ERPKernel] = []
        self.waveform: np.ndarray | None = None
        self.time: np.ndarray | None = None

    def add(self, peak_latency: float, amplitude: float, width: float, modifier: Callable[..., float] = lambda : 0.0,
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
        modifier : Callable[..., float]
            Function called to modify the amplitude of the kernel.
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
            ERPKernel(self.sfreq, peak_latency, amplitude, width, modifier, shape, label))
        self._rebuild()
        return self

    def _rebuild(self, **kwargs):
        """Recompute the summed waveform from all components."""
        for k in self.components:
            k._build(**kwargs)

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

    def plot_interactive(self, show_components: bool = True, **kwargs):
        """Plot the compound kernel, optionally with individual components.

        Parameters
        ----------
        show_components : bool
            If True, overlay each component as a thin dashed line.
        """
        from matplotlib.widgets import Button, Slider

        # Make horizontal sliders for parameters
        vars = set()
        for k in self.components:
            for x in k.modifying_variables:
                vars.add(x)

        args = {k: kwargs[k] for c in self.components for k in c.modifying_variables}
        sliders = []

        fig, ax = plt.subplots()
        ax.plot(self.time * 1e3, self.waveform, 'k', lw=1.5, label='sum')
        
        fig.subplots_adjust(bottom=0.15 + 0.1 * len(vars))

        bottom = 0.1 * len(vars) 
        for x in vars:
            axv = fig.add_axes([0.25, bottom, 0.65, 0.03])
            slider = Slider(
                ax=axv,
                label=x,
                valmin=0,
                valmax=30,
                valinit=kwargs[x]
            )

            def update(value):
                args[x] = value
                self._rebuild(**args)

                ax.cla()
                ax.plot(self.time * 1e3, self.waveform, 'k', lw=1.5, label='sum')
                if show_components:
                    for k in self.components:
                        ax.plot(k.time * 1e3, k.waveform, ls='--', lw=0.8, alpha=0.6, label=k.label)

                fig.canvas.draw_idle()

            slider.on_changed(update)
            sliders.append(slider)
            bottom -= 0.1

        if show_components:
            for k in self.components:
                ax.plot(k.time * 1e3, k.waveform, ls='--', lw=0.8, alpha=0.6, label=k.label)

        ax.set_xlabel('Time (ms)')
        ax.set_ylabel('Amplitude')
        ax.set_title('Compound ERP Kernel')
        ax.legend(fontsize=8)
        plt.show()
        return sliders


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
        self.kernels: list[CompoundKernel] = []
        self.event_sticks: dict[str, np.ndarray] = {}

    def add_kernel(self, activation_function : Callable[..., bool], kernel=None):
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
        self.kernels.append((kernel, activation_function))

    def set_events(self, events: pd.DataFrame):
        """Build stick functions from an events DataFrame.

        Parameters
        ----------
        events : pandas.DataFrame
            Must contain ``'latency'`` (in samples) and ``'type'`` columns.
        """
        self.events = events.copy()
        self.event_sticks = {}
        self.component_sticks = {}

        for (comp_kernel, act_function) in self.kernels:
            for kernel in comp_kernel.components:
                stick = np.zeros(self.n_samples)
                for _row_idx, event in events.iterrows():
                    if not act_function(event):
                        continue

                    # TODO: do we need to define a sort of onset instead?
                    latency_offset = int(event['latency']) + int(kernel.peak_latency * kernel.sfreq)
                    stick[latency_offset] = 1.0 + kernel.modifier(**event)

                self.component_sticks[(comp_kernel.name, kernel.label)] = stick


        # for evt_type in events['type'].unique():
        #     stick = np.zeros(self.n_samples)
        #     lats = events.loc[events['type'] == evt_type, 'latency'].values.astype(int)
        #     valid = lats[(lats >= 0) & (lats < self.n_samples)]
        #     stick[valid] = 1.0
        #     self.event_sticks[evt_type] = stick

    def simulate(self) -> np.ndarray:
        """Convolve each stick function with its kernel and sum into self.data.

        Returns
        -------
        data : numpy.ndarray
            The simulated EEG signal (n_samples,).
        """
        self.data = np.zeros(self.n_samples)
        for comp_kernel, _act_function in self.kernels:
            for kernel in comp_kernel.components:
                if (comp_kernel.name, kernel.label) not in self.component_sticks:
                    continue

                stick = self.component_sticks[(comp_kernel.name, kernel.label)]
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

    def plot(self, event_category: Callable[..., str] = lambda evt: evt.get('type', "NoType")):
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
                etype = event_category(evt)
                c = colors[hash(etype) % len(colors)]
                lbl = etype if etype not in used else None
                axs[0].axvline(x=evt['latency'], color=c, ls='--', alpha=0.3, label=lbl)
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


class TrialVariable:
    """Specification for a variable attached to events.

    Parameters
    ----------
    name
        Column name to add to the events DataFrame.
    generator
        Function called to generate values. It will be called as
        ``generator(n, rng)`` and must return a length-n sequence.
        (Extra args are optional; see notes.)
    static_accross_trial
        If True, generate one value per *trial* and broadcast to all events in
        that trial. If False, generate one value per *event*.

    Notes
    -----
    For convenience, generators may also accept fewer parameters:
    - ``generator(n)``
    - ``generator(n, rng)``
    """

    name: str
    generator: Callable[..., Sequence[Any]]
    static_across_trial: bool = False

    def __init__(self, name: str, generator: Callable[..., Sequence[Any]], static_accross_trial: bool = False):
        self.name = name
        self.generator = generator
        self.static_across_trial = static_accross_trial


class TrialStructure:
    """Defines per-trial variables and can generate an events DataFrame.

    This is a lightweight building block intended to sit “above” the low-level
    convolution simulator (`EEGSimulator`). The output events DataFrame is
    compatible with `EEGSimulator.set_events()` (must contain `latency`, `type`).
    """

    def __init__(self, sfreq: float, variables: list[TrialVariable] = [], seed: int | None = None):
        self.sfreq = sfreq
        self.variables = variables
        self.rng = np.random.default_rng(seed)

    def add_variable(self, name: str, generator: Callable[..., Any], *, static: bool = False):
        self.variables.append(TrialVariable(name=name, generator=generator, static_across_trial=static))
        return self

    def generate_events_df(
        self,
        n_events: int,
    ) -> pd.DataFrame:
        """Generate an events DataFrame for one trial.

        Parameters
        ----------
        n_events
            Number of events in the trial.

        Returns
        -------
        events : pandas.DataFrame
            Columns: any variables defined in the trial structure.
        """
        events = pd.DataFrame(index=range(n_events))

        for var in self.variables:
            if var.static_across_trial:
                value = var.generator(1, self.rng)
                if len(value) != 1:
                    raise ValueError(f"Static variable '{var.name}' generator must return length 1.")
                events[var.name] = value[0]
            else:
                values = var.generator(n_events, self.rng)
                events[var.name] = values

        return events
