"""Simulation utilities for EEG/MEG deconvolution testing.

This module provides classes for generating synthetic continuous EEG data
with known event-related kernels. The simulated signal can then be fed
through the PyDeconv deconvolution pipeline to verify kernel recovery.

Classes
-------
ERPKernel
    A single ERP component (Gaussian or Hanning window).
CompoundKernel
    Multiple ``ERPKernel`` components summed into one named kernel.
EEGSimulator
    Generates continuous EEG by convolving event stick functions with kernels.
TrialVariable
    Specification for a covariate column attached to events.
TrialStructure
    Generates an events DataFrame from ``TrialVariable`` specs.
ExperimentDesign
    High-level convenience wrapper that combines ``TrialStructure`` with
    ISI sampling to produce a complete events DataFrame ready for
    ``EEGSimulator``.

Functions
---------
build_uniform_isi_sampler
    Factory for uniform inter-stimulus-interval samplers.
assign_event_latencies
    Assign cumulative latencies to an events DataFrame via an ISI sampler.
"""

import inspect
from typing import Any, Callable, Sequence

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.signal import fftconvolve, windows

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

ISISampler = Callable[[pd.Series], int]
"""Callable that receives an event row and returns an inter-stimulus interval
(in samples)."""

# ---------------------------------------------------------------------------
# ISI helpers
# ---------------------------------------------------------------------------


def build_uniform_isi_sampler(
    width: int,
    offset: int = 0,
    *,
    rng: np.random.Generator | None = None,
) -> ISISampler:
    """Build an ISI sampler that draws uniformly from ``[offset, offset+width]``.

    Parameters
    ----------
    width : int
        Range of the uniform distribution (in samples).
    offset : int
        Minimum ISI value (in samples).
    rng : numpy.random.Generator or None
        Random number generator. If None, a new default generator is created.

    Returns
    -------
    sampler : ISISampler
        A callable ``(row) -> int``.
    """
    rng = rng or np.random.default_rng()

    def _sample(_row: pd.Series) -> int:
        return int(rng.integers(offset, offset + width + 1))

    return _sample


def assign_event_latencies(
    events: pd.DataFrame,
    sampler: ISISampler,
) -> pd.DataFrame:
    """Assign cumulative latencies by sampling an ISI offset per row.

    Parameters
    ----------
    events : pandas.DataFrame
        Events table (rows are individual events).
    sampler : ISISampler
        Callable ``(row) -> int`` returning an ISI in samples.

    Returns
    -------
    events : pandas.DataFrame
        Copy of *events* with a ``latency`` column (cumulative, in samples).
    """
    out = events.copy()
    latencies: list[int] = []
    for _, row in out.iterrows():
        delta = sampler(row)
        latencies.append(delta if not latencies else latencies[-1] + delta)
    out["latency"] = latencies
    return out


# ---------------------------------------------------------------------------
# ERPKernel
# ---------------------------------------------------------------------------


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
    modifier : callable, optional
        Function that receives event-row keyword arguments and returns an
        additive amplitude offset.  For example
        ``lambda contrast: 2.0 * contrast`` would scale the kernel by the
        event's ``contrast`` column.  Only parameters whose names match
        columns in the events DataFrame are forwarded.
    shape : str
        Kernel shape: ``'gaussian'`` or ``'hanning'``.
    label : str or None
        Human-readable label (e.g. ``'P1'``, ``'N170'``).
    """

    def __init__(
        self,
        sfreq: float,
        peak_latency: float,
        amplitude: float,
        width: float,
        modifier: Callable[..., float] | None = None,
        shape: str = "gaussian",
        label: str | None = None,
    ):
        self.sfreq = sfreq
        self.peak_latency = peak_latency
        self.amplitude = amplitude
        self.width = width
        self.shape = shape
        self.label = label or f"{shape}@{peak_latency:.3f}s"

        # --- modifier wiring ---
        self.modifying_variables: set[str] = set()
        if modifier is None:
            self.modifier: Callable[..., float] = lambda **_kw: 0.0
        else:
            for name in inspect.signature(modifier).parameters:
                self.modifying_variables.add(name)

            if "kwargs" not in self.modifying_variables:
                _orig = modifier

                def _wrapper(**kwargs: Any) -> float:
                    filtered = {
                        k: v for k, v in kwargs.items()
                        if k in self.modifying_variables
                    }
                    return _orig(**filtered)

                self.modifier = _wrapper
            else:
                self.modifier = modifier

        self._build()

    # ----- internal -----

    def _build(self) -> None:
        """Compute the discrete kernel waveform."""
        t_end = self.peak_latency + 4 * self.width
        n_samples = max(int(np.ceil(t_end * self.sfreq)), 1)
        self.time = np.arange(n_samples) / self.sfreq

        if self.shape == "gaussian":
            self.waveform = self.amplitude * np.exp(
                -0.5 * ((self.time - self.peak_latency) / self.width) ** 2
            )
        elif self.shape == "hanning":
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
            raise ValueError(
                f"Unsupported shape '{self.shape}'. Use 'gaussian' or 'hanning'."
            )

    # ----- plotting -----

    def plot(self, ax: plt.Axes | None = None) -> plt.Axes:
        """Plot the kernel waveform.

        Parameters
        ----------
        ax : matplotlib.axes.Axes or None
            Axes to plot on.  If None a new figure is created.

        Returns
        -------
        ax : matplotlib.axes.Axes
        """
        if ax is None:
            _, ax = plt.subplots(figsize=(6, 3), tight_layout=True)
        ax.plot(self.time * 1e3, self.waveform, lw=1.2)
        ax.axvline(
            self.peak_latency * 1e3, color="grey", ls="--", lw=0.8, alpha=0.6
        )
        ax.set_xlabel("Time (ms)")
        ax.set_ylabel("Amplitude")
        ax.set_title(self.label)
        return ax


# ---------------------------------------------------------------------------
# CompoundKernel
# ---------------------------------------------------------------------------


class CompoundKernel:
    """Multiple ``ERPKernel`` components summed into one named kernel.

    Parameters
    ----------
    name : str
        Identifier for this kernel (used as key in ``EEGSimulator``).
    sfreq : float
        Sampling frequency in Hz.
    """

    def __init__(self, name: str, sfreq: float):
        self.name = name
        self.sfreq = sfreq
        self.components: list[ERPKernel] = []
        self.waveform: np.ndarray | None = None
        self.time: np.ndarray | None = None

    def add(
        self,
        peak_latency: float,
        amplitude: float,
        width: float,
        modifier: Callable[..., float] | None = None,
        shape: str = "gaussian",
        label: str | None = None,
    ) -> "CompoundKernel":
        """Add a component to the compound kernel.

        Parameters
        ----------
        peak_latency : float
            Peak latency in seconds.
        amplitude : float
            Peak amplitude.
        width : float
            Width in seconds.
        modifier : callable or None
            Amplitude modifier (see ``ERPKernel``).
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
            ERPKernel(
                self.sfreq, peak_latency, amplitude, width,
                modifier, shape, label,
            )
        )
        self._rebuild()
        return self

    def _rebuild(self) -> None:
        """Recompute the summed waveform from all components."""
        max_len = max(len(k.waveform) for k in self.components)
        self.time = np.arange(max_len) / self.sfreq
        self.waveform = np.zeros(max_len)
        for k in self.components:
            self.waveform[: len(k.waveform)] += k.waveform

    def plot(self, show_components: bool = True) -> None:
        """Plot the compound kernel, optionally with individual components.

        Parameters
        ----------
        show_components : bool
            If True, overlay each component as a thin dashed line.
        """
        _, ax = plt.subplots(figsize=(7, 3.5), tight_layout=True)
        ax.plot(self.time * 1e3, self.waveform, "k", lw=1.5, label="sum")
        if show_components:
            for k in self.components:
                ax.plot(
                    k.time * 1e3, k.waveform,
                    ls="--", lw=0.8, alpha=0.6, label=k.label,
                )
        ax.set_xlabel("Time (ms)")
        ax.set_ylabel("Amplitude")
        ax.set_title(f"Compound ERP Kernel — {self.name}")
        ax.legend(fontsize=8)
        plt.show()


# ---------------------------------------------------------------------------
# EEGSimulator
# ---------------------------------------------------------------------------


class EEGSimulator:
    """Generate continuous single-channel EEG by convolving stick functions
    with ERP kernels.

    Kernels are registered together with an *activation function* that decides,
    for each event row, whether that kernel fires.  Each ``ERPKernel`` inside a
    ``CompoundKernel`` builds its own stick function whose amplitude is
    ``1.0 + modifier(**event_row)``, allowing covariate-driven amplitude
    modulation.

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
        self.kernels: list[tuple[CompoundKernel, Callable[..., bool]]] = []
        self.component_sticks: dict[tuple[str, str], np.ndarray] = {}
        self.events: pd.DataFrame | None = None

    # ----- kernel registration -----

    def add_kernel(
        self,
        kernel: CompoundKernel,
        activation: Callable[..., bool] | None = None,
    ) -> None:
        """Register a ``CompoundKernel`` with an activation function.

        Parameters
        ----------
        kernel : CompoundKernel
            The compound kernel to register.
        activation : callable or None
            A function ``(event_row: pd.Series) -> bool`` that decides whether
            this kernel fires for a given event.  If None, the kernel fires for
            every event.
        """
        if activation is None:
            activation = lambda _row: True  # noqa: E731
        self.kernels.append((kernel, activation))

    # ----- events -----

    def set_events(self, events: pd.DataFrame) -> None:
        """Build per-component stick functions from an events DataFrame.

        For each registered ``CompoundKernel`` and each of its ``ERPKernel``
        components, a stick function is created.  The stick value at each
        qualifying event equals ``1.0 + modifier(**event_row)``.

        Parameters
        ----------
        events : pandas.DataFrame
            Must contain a ``'latency'`` column (in samples).
        """
        self.events = events.copy()
        self.component_sticks = {}

        for comp_kernel, act_fn in self.kernels:
            for component in comp_kernel.components:
                stick = np.zeros(self.n_samples)
                for _, event in events.iterrows():
                    if not act_fn(event):
                        continue
                    lat = int(event["latency"])
                    if lat < 0 or lat >= self.n_samples:
                        continue
                    stick[lat] = 1.0 + component.modifier(
                        **event.to_dict()
                    )
                self.component_sticks[
                    (comp_kernel.name, component.label)
                ] = stick

    # ----- simulation -----

    def simulate(self) -> np.ndarray:
        """Convolve each component stick with its kernel waveform and sum.

        Returns
        -------
        data : numpy.ndarray, shape ``(n_samples,)``
            The simulated EEG signal.
        """
        self.data = np.zeros(self.n_samples)
        for comp_kernel, _ in self.kernels:
            for component in comp_kernel.components:
                key = (comp_kernel.name, component.label)
                if key not in self.component_sticks:
                    continue
                stick = self.component_sticks[key]
                convolved = fftconvolve(
                    stick, component.waveform, mode="full"
                )[: self.n_samples]
                self.data += convolved
        return self.data

    # ----- noise -----

    def add_noise(
        self,
        colour: str = "brown",
        scale: float = 0.5,
        rng: np.random.Generator | None = None,
    ) -> None:
        """Add coloured noise to the signal.

        Parameters
        ----------
        colour : str
            Noise colour: ``'white'``, ``'pink'``, or ``'brown'``.
        scale : float
            Standard deviation of the resulting noise.
        rng : numpy.random.Generator or None
            Random number generator for reproducibility.
        """
        rng = rng or np.random.default_rng()
        exponents = {"white": 0.0, "pink": 0.5, "brown": 1.0}
        exp = exponents.get(colour.lower())
        if exp is None:
            raise ValueError(
                f"Unknown noise colour '{colour}'. "
                "Use 'white', 'pink', or 'brown'."
            )

        if colour.lower() == "white":
            self.data += rng.standard_normal(self.n_samples) * scale
            return

        freqs = np.fft.rfftfreq(self.n_samples, d=1.0 / self.sfreq)
        freqs[0] = 1.0  # avoid division by zero
        spectrum = (
            rng.standard_normal(len(freqs))
            + 1j * rng.standard_normal(len(freqs))
        )
        spectrum /= freqs ** exp
        noise = np.fft.irfft(spectrum, n=self.n_samples)
        noise = noise / np.std(noise) * scale
        self.data += noise

    # ----- convenience -----

    def generate(
        self,
        events: pd.DataFrame,
        noise_colour: str = "brown",
        noise_scale: float = 0.5,
        rng: np.random.Generator | None = None,
    ) -> np.ndarray:
        """One-shot helper: set events, simulate, add noise.

        Parameters
        ----------
        events : pandas.DataFrame
            Events table with ``'latency'`` column (in samples).
        noise_colour : str
            Noise colour (see ``add_noise``).
        noise_scale : float
            Noise amplitude.
        rng : numpy.random.Generator or None
            Random number generator for noise.

        Returns
        -------
        data : numpy.ndarray, shape ``(n_samples,)``
            Simulated EEG signal with noise.
        """
        self.set_events(events)
        self.simulate()
        self.add_noise(colour=noise_colour, scale=noise_scale, rng=rng)
        return self.data

    # ----- plotting -----

    def plot(
        self,
        event_category: Callable[..., str] | None = None,
    ) -> None:
        """Plot the simulated signal with event markers and PSD.

        Parameters
        ----------
        event_category : callable or None
            Function ``(event_row: pd.Series) -> str`` returning a category
            label used for colour-coding event markers.  Defaults to using
            the ``'type'`` column if present, otherwise ``'event'``.
        """
        if event_category is None:
            event_category = lambda evt: evt.get("type", "event")  # noqa: E731

        n = self.n_samples
        spectrum = np.fft.rfft(self.data) / n
        pwr = 10.0 * np.log10(2.0 * np.abs(spectrum) ** 2 + 1e-30)
        freqs = np.fft.rfftfreq(n, d=1.0 / self.sfreq)

        fig, axs = plt.subplots(2, 1, figsize=(10, 6), tight_layout=True)

        # ---- time-domain ----
        axs[0].plot(self.time, self.data, lw=0.5)
        colors = plt.cm.tab10.colors
        used: set[str] = set()
        if self.events is not None:
            for _, evt in self.events.iterrows():
                etype = event_category(evt)
                c = colors[hash(etype) % len(colors)]
                lbl = etype if etype not in used else None
                lat_s = evt["latency"] / self.sfreq
                axs[0].axvline(
                    x=lat_s, color=c, ls="--", alpha=0.3, label=lbl
                )
                used.add(etype)
            axs[0].legend(fontsize=8)
        axs[0].set_xlabel("Time (s)")
        axs[0].set_ylabel("Amplitude")
        axs[0].set_title("Simulated EEG")

        # ---- PSD (up to 100 Hz) ----
        idx = np.searchsorted(freqs, 100)
        axs[1].plot(freqs[:idx], pwr[:idx], lw=0.8)
        axs[1].set_xlabel("Frequency (Hz)")
        axs[1].set_ylabel("Power (dB)")
        axs[1].set_ylim([-100, 40])
        axs[1].set_title("Power Spectral Density")

        plt.show()


# ---------------------------------------------------------------------------
# TrialVariable & TrialStructure
# ---------------------------------------------------------------------------


class TrialVariable:
    """Specification for a covariate column attached to events.

    Parameters
    ----------
    name : str
        Column name to add to the events DataFrame.
    generator : callable
        Called as ``generator(n, rng)`` and must return a length-*n* sequence.
    static_across_trial : bool
        If True, generate one value per *trial* and broadcast to all events
        in that trial.  If False, generate one value per *event*.
    """

    def __init__(
        self,
        name: str,
        generator: Callable[..., Sequence[Any]],
        static_across_trial: bool = False,
    ):
        self.name = name
        self.generator = generator
        self.static_across_trial = static_across_trial


class TrialStructure:
    """Generates an events DataFrame from ``TrialVariable`` specifications.

    The output is compatible with ``EEGSimulator.set_events()`` once
    latencies are assigned (e.g. via ``assign_event_latencies``).

    Parameters
    ----------
    sfreq : float
        Sampling frequency in Hz.
    variables : list of TrialVariable
        Variable specifications.
    seed : int or None
        Seed for the random number generator.
    """

    def __init__(
        self,
        sfreq: float,
        variables: list[TrialVariable] | None = None,
        seed: int | None = None,
    ):
        self.sfreq = sfreq
        self.variables: list[TrialVariable] = variables or []
        self.rng = np.random.default_rng(seed)

    def add_variable(
        self,
        name: str,
        generator: Callable[..., Any],
        *,
        static: bool = False,
    ) -> "TrialStructure":
        """Add a variable specification.

        Parameters
        ----------
        name : str
            Column name.
        generator : callable
            ``(n, rng) -> Sequence``.
        static : bool
            Broadcast a single value to all events in the trial.

        Returns
        -------
        self : TrialStructure
            For method chaining.
        """
        self.variables.append(
            TrialVariable(
                name=name,
                generator=generator,
                static_across_trial=static,
            )
        )
        return self

    def generate_events_df(self, n_events: int) -> pd.DataFrame:
        """Generate an events DataFrame for one trial.

        Parameters
        ----------
        n_events : int
            Number of events to generate.

        Returns
        -------
        events : pandas.DataFrame
            One row per event, columns from the registered variables.
        """
        events = pd.DataFrame(index=range(n_events))

        for var in self.variables:
            if var.static_across_trial:
                value = var.generator(1, self.rng)
                if len(value) != 1:
                    raise ValueError(
                        f"Static variable '{var.name}' generator must "
                        "return length 1."
                    )
                events[var.name] = value[0]
            else:
                values = var.generator(n_events, self.rng)
                events[var.name] = values

        return events


# ---------------------------------------------------------------------------
# ExperimentDesign  (convenience layer)
# ---------------------------------------------------------------------------


class ExperimentDesign:
    """High-level helper for generating a complete events DataFrame.

    Wraps ``TrialStructure`` and an ISI sampler to produce events with
    ``latency`` (samples), ``latency_s`` (seconds), and ``type`` columns,
    plus any user-defined covariates.

    Parameters
    ----------
    event_types : list of str
        Labels cycled across events (e.g. ``['stimulus', 'response']``).
    n_events : int
        Total number of events to generate.
    sfreq : float
        Sampling frequency in Hz.
    duration_s : float
        Maximum signal duration in seconds.  Events whose cumulative latency
        exceeds ``duration_s`` are dropped.
    isi_range : tuple of int
        ``(offset, width)`` passed to ``build_uniform_isi_sampler``.
    variables : list of TrialVariable or None
        Additional covariates.
    seed : int or None
        RNG seed for reproducibility.
    """

    def __init__(
        self,
        event_types: list[str],
        n_events: int,
        sfreq: float,
        duration_s: float,
        isi_range: tuple[int, int] = (50, 200),
        variables: list[TrialVariable] | None = None,
        seed: int | None = None,
    ):
        self.event_types = event_types
        self.n_events = n_events
        self.sfreq = sfreq
        self.duration_s = duration_s
        self.isi_range = isi_range
        self.seed = seed

        rng = np.random.default_rng(seed)

        # Build the type column as a cycled sequence
        type_gen = lambda n, _rng: [  # noqa: E731
            event_types[i % len(event_types)] for i in range(n)
        ]

        all_vars = [TrialVariable("type", type_gen)]
        if variables:
            all_vars.extend(variables)

        self._structure = TrialStructure(
            sfreq=sfreq, variables=all_vars, seed=seed,
        )
        self._sampler = build_uniform_isi_sampler(
            width=isi_range[1], offset=isi_range[0], rng=rng,
        )

    def generate_events(self) -> pd.DataFrame:
        """Generate the full events DataFrame.

        Returns
        -------
        events : pandas.DataFrame
            Columns include ``type``, ``latency`` (samples),
            ``latency_s`` (seconds), plus any user-defined variables.
            Events exceeding ``duration_s`` are dropped.
        """
        max_sample = int(self.duration_s * self.sfreq)

        events = self._structure.generate_events_df(self.n_events)
        events = assign_event_latencies(events, self._sampler)
        events["latency_s"] = events["latency"] / self.sfreq

        # Drop events that exceed duration
        events = events[events["latency"] < max_sample].reset_index(drop=True)
        return events
