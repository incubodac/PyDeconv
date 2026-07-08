# Plotting methods (coefficients/kernels, design matrices, metrics, and waveforms)

import matplotlib.pyplot as plt
import numpy as np


def plot_simulation_kernels(simulator, figsize=None):
    """Plot simulated EEG data with event-related kernels, ISI info, PSD, and event occurrences.

    Parameters
    ----------
    simulator : pydeconv.simulation.EEGSimulator
        The simulator object containing events, data, and registered kernels.
    figsize : tuple, optional
        The figure size.
    
    Returns
    -------
    fig : matplotlib.figure.Figure

    """
    from scipy.signal import welch

    events = getattr(simulator, "events", None)
    y = getattr(simulator, "data", None)
    sfreq = getattr(simulator, "sfreq", 256)

    if events is None or y is None:
        raise ValueError("Simulator must have 'events' and 'data' (ensure you called simulate()).")

    # Extract ground truth from registered kernels
    ground_truth = {k.name: k for k, _ in getattr(simulator, "kernels", [])}

    if not ground_truth:
        raise ValueError("Simulator has no kernels registered to plot.")

    n_features = len(ground_truth)

    if figsize is None:
        figsize = (15, 3 * n_features + 8)

    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(n_features + 2, 3, height_ratios=[1]*n_features + [1.5, 1.2])

    # ── 1. Ground Truth Kernels (Simulations) ──
    for i, (feature_name, gt_kernel) in enumerate(ground_truth.items()):
        ax = fig.add_subplot(gs[i, :])

        times = gt_kernel.time
        gt_waveform = gt_kernel.waveform

        ax.plot(times, gt_waveform, color="tab:red", ls="-", lw=2, label="Ground Truth (Simulated)")

        ax.set_title(f"Kernel: {feature_name}", fontweight="bold")
        ax.set_ylabel("Amplitude")
        ax.axhline(0, color="k", lw=0.5, ls="--")
        ax.axvline(0, color="k", lw=0.5, ls="-")
        ax.legend(loc="upper right")

        if i == n_features - 1:
            ax.set_xlabel("Time (s)")

    # ── 2. EEG Snippet + Event Rugplot ──
    ax_eeg = fig.add_subplot(gs[n_features, :])
    snippet_len_s = min(10.0, len(y) / sfreq)
    snippet_len_samples = int(snippet_len_s * sfreq)
    t_eeg = np.arange(snippet_len_samples) / sfreq

    if y.ndim > 1:
        y_plot = y[:snippet_len_samples, 0]
        y_lbl = "EEG (Channel 0)"
    else:
        y_plot = y[:snippet_len_samples]
        y_lbl = "EEG"

    ax_eeg.plot(t_eeg, y_plot, color="black", lw=1, alpha=0.8, label=y_lbl)

    latency_s = events["latency"] / sfreq
    events_snippet = events[latency_s <= snippet_len_s]
    unique_types = events["type"].unique()
    colors = plt.cm.tab10(np.linspace(0, 1, max(10, len(unique_types))))
    color_map = {t: c for t, c in zip(unique_types, colors)}

    for ev_type in unique_types:
        mask = (events_snippet["type"] == ev_type)
        ev_times = events_snippet.loc[mask, "latency"] / sfreq
        if len(ev_times) > 0:
            ax_eeg.vlines(ev_times, ymin=y_plot.min(), ymax=y_plot.max(), color=color_map[ev_type], alpha=0.5, ls="--", label=f"Event: {ev_type}")

    ax_eeg.set_title(f"Simulated EEG Snippet (First {snippet_len_s:.1f}s)", fontweight="bold")
    ax_eeg.set_xlabel("Time (s)")
    ax_eeg.set_ylabel("Amplitude")
    ax_eeg.legend(loc="upper right", bbox_to_anchor=(1.15, 1))

    # ── 3. Event Occurrences (Raster) ──
    ax_raster = fig.add_subplot(gs[n_features + 1, 0])
    for i, ev_type in enumerate(unique_types):
        mask = (events["type"] == ev_type)
        ev_times = events.loc[mask, "latency"] / sfreq
        ax_raster.scatter(ev_times, np.full_like(ev_times, i), color=color_map[ev_type], s=10, label=ev_type)

    ax_raster.set_yticks(range(len(unique_types)))
    ax_raster.set_yticklabels(unique_types)
    ax_raster.set_title("Event Occurrences", fontweight="bold")
    ax_raster.set_xlabel("Time (s)")

    # ── 4. ISI Histogram ──
    ax_isi = fig.add_subplot(gs[n_features + 1, 1])
    isi = np.diff(np.sort(events["latency"] / sfreq))
    ax_isi.hist(isi, bins=30, color="grey", edgecolor="black", alpha=0.7)
    ax_isi.set_title("Inter-Stimulus Intervals", fontweight="bold")
    ax_isi.set_xlabel("ISI (s)")
    ax_isi.set_ylabel("Count")

    # ── 5. PSD ──
    ax_psd = fig.add_subplot(gs[n_features + 1, 2])
    if y.ndim > 1:
        y_psd = y[:, 0]
    else:
        y_psd = y
    freqs, psd = welch(y_psd, fs=sfreq, nperseg=min(len(y_psd), int(2 * sfreq)))
    psd = np.maximum(psd, 1e-12)
    ax_psd.plot(freqs, 10 * np.log10(psd), color="tab:purple")
    ax_psd.set_title("Power Spectral Density", fontweight="bold")
    ax_psd.set_xlabel("Frequency (Hz)")
    ax_psd.set_ylabel("Power (dB/Hz)")
    ax_psd.set_xlim(0, min(100, sfreq / 2))

    plt.tight_layout()
    return fig


def plot_trfs(model, info=None, features=None, top_topos=True, figsize=(15, 8)):
    """Plot the fitted Temporal Response Functions (TRFs) using MNE.

    This uses a horizontal layout inspired by the legacy PyDeconv plots,
    where each feature gets its own column consisting of a butterfly plot
    and (optionally) topomaps at peak times.

    Parameters
    ----------
    model : pydeconv.core.DeconvolutionModel
        A fitted DeconvolutionModel containing ``coef_``.
    info : mne.Info, optional
        The MNE Info object corresponding to the channels used to fit the model.
        If None, a dummy Info object is created automatically.
    features : list of str, optional
        A list of feature names to plot. If None, plots all features
        (except the intercept, if present).
    top_topos : bool, default True
        If True, plots joint time-series and topomaps (mne.Evoked.plot_joint).
        If False, only plots the butterfly time-series.
    figsize : tuple, default (15, 8)
        The overall figure size.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The resulting figure.

    """
    import mne

    if getattr(model, "coef_", None) is None:
        raise ValueError("Model is not fitted. Cannot plot TRFs.")

    if features is None:
        features = [f for f in model.feature_names_ if f != "intercept"]

    n_delays = len(model.delays_)
    times = getattr(model, "times_", np.arange(n_delays))

    def _feature_delay_mask(feat_name: str) -> np.ndarray:
        """Return per-feature delay mask from model.analysis_windows."""
        if not hasattr(model, "analysis_windows") or ":" not in feat_name:
            return np.ones(n_delays, dtype=bool)

        event_type = feat_name.split(":", 1)[0]
        win = model.analysis_windows.get(event_type)
        if win is None:
            return np.ones(n_delays, dtype=bool)

        delay_min = int(np.round(win[0] * model.sfreq))
        delay_max = int(np.round(win[1] * model.sfreq))
        mask = (model.delays_ >= delay_min) & (model.delays_ <= delay_max)
        if not np.any(mask):
            return np.ones(n_delays, dtype=bool)
        return mask

    coef = model.coef_
    if coef.ndim == 1:
        coef = coef[np.newaxis, :]

    if info is None:
        n_channels = coef.shape[0]
        ch_names = [f"ch_{i}" for i in range(n_channels)]
        info = mne.create_info(ch_names=ch_names, sfreq=model.sfreq, ch_types=["eeg"] * n_channels)
        top_topos = False  # Dummy info has no sensor coordinates for topomaps

    fig = plt.figure(figsize=figsize)

    # Layout constants from legacy code
    top_slide = 0.02
    horizontal_jump = 0.8 / len(features)  # dynamically space out based on n features

    for jump, feat_name in enumerate(features):
        try:
            n_coeff = model.feature_names_.index(feat_name)
        except ValueError:
            print(f"Warning: Feature '{feat_name}' not found in model. Skipping.")
            continue

        # Extract data for this TRF: shape (n_channels, n_delays)
        start_idx = n_coeff * n_delays
        end_idx = (n_coeff + 1) * n_delays
        data_full = coef[:, start_idx:end_idx]

        # Event-specific features can have narrower analysis windows.
        keep_mask = _feature_delay_mask(feat_name)
        data = data_full[:, keep_mask]
        times_feat = times[keep_mask]
        x_lims = (times_feat[0], times_feat[-1])

        # Create an Evoked object
        grand_avg = mne.EvokedArray(data, info, tmin=times_feat[0], verbose=False)
        grand_avg.nave = None

        # Determine global max for symmetric colormap
        vmax = np.max(np.abs(data))
        vlim = (-vmax, vmax)

        # Calculate horizontal position dynamically
        x0 = 0.05 + jump * horizontal_jump
        width = horizontal_jump - 0.05

        ax_frp = fig.add_axes((x0, 0.47, width, 0.2))

        if top_topos:
            # We place 3 topomaps directly above the line plot
            topo_w = width * 0.25
            gap = width * 0.05
            ax_topo1 = fig.add_axes((x0, 0.75, topo_w, 0.15))
            ax_topo2 = fig.add_axes((x0 + topo_w + gap, 0.75, topo_w, 0.15))
            ax_topo3 = fig.add_axes((x0 + 2*(topo_w + gap), 0.75, topo_w, 0.15))
            ax_topo_cb = fig.add_axes((x0 + 3*(topo_w + gap), 0.75, width * 0.02, 0.15))
            axs_topos = [ax_topo1, ax_topo2, ax_topo3, ax_topo_cb]

            grand_avg.plot_joint(
                title="",
                ts_args={'xlim': x_lims, 'axes': ax_frp, 'titles': dict(eeg=''), 'window_title': ''},
                topomap_args={'vlim': vlim, 'contours': 2, 'axes': axs_topos, 'size': 0.8},
                show=False
            )

            # Format topomap colorbar
            ax_cb = axs_topos[-1]
            ax_cb.set_title(r'$\mu V$', fontsize=10)
            for top in axs_topos:
                top.title.set_fontsize(10)

        else:
            grand_avg.plot(
                axes=ax_frp,
                titles=dict(eeg=''),
                window_title='',
                xlim=x_lims,
                show=False
            )

        # Clean up axes
        ax_frp.set_xlabel("Time (s)")
        ax_frp.set_title(f"{feat_name}", fontweight="bold", pad=15)
        if jump > 0:
            ax_frp.set_ylabel("")
            ax_frp.set_yticklabels([])

        # Remove any unwanted text like '(64 channels)'
        for c in ax_frp.get_children():
            if isinstance(c, plt.Text) and 'channels' in c.get_text():
                c.remove()

    fig.legends = []
    return fig
