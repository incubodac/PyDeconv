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
            ax_eeg.vlines(
                ev_times,
                ymin=y_plot.min(),
                ymax=y_plot.max(),
                color=color_map[ev_type],
                alpha=0.5,
                ls="--",
                label=f"Event: {ev_type}"
            )

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


# ---------------------------------------------------------------------------
# Feature grouping helper
# ---------------------------------------------------------------------------


def _group_features(feature_names):
    """Group feature names, collapsing B-spline bases and their intercepts.

    Spline columns named ``"prefix_sp_0"``, ``"prefix_sp_1"``, … are
    merged under the key ``"prefix"``.

    When a spline group belongs to an event (e.g. ``saccade:sac_amp``),
    the corresponding event intercept (``saccade:intercept``) is
    automatically absorbed into that group so the plotted TRF shows the
    total response (intercept + spline modulation).

    Parameters
    ----------
    feature_names : list of str
        The ``model.feature_names_`` list.

    Returns
    -------
    groups : dict[str, list[str]]
        Ordered mapping from group name to the list of feature names
        belonging to that group.  Intercepts that have been absorbed
        into a spline group are **not** listed as standalone groups.

    """
    groups: dict[str, list[str]] = {}
    for name in feature_names:
        base = name
        if "_sp_" in name:
            base = name.rsplit("_sp_", 1)[0]
        groups.setdefault(base, []).append(name)

    # Absorb event intercepts into their spline groups.
    # For each spline group (has _sp_ members), find its event prefix
    # and check if "prefix:intercept" exists as a standalone group.
    absorbed_intercepts: set[str] = set()
    spline_groups = [
        gname for gname, members in groups.items()
        if any("_sp_" in m for m in members)
    ]
    for gname in spline_groups:
        if ":" in gname:
            prefix = gname.split(":")[0]
            intercept_key = f"{prefix}:intercept"
        else:
            intercept_key = "intercept"

        if intercept_key in groups and intercept_key not in absorbed_intercepts:
            # Add the intercept feature name(s) into the spline group
            groups[gname] = groups[intercept_key] + groups[gname]
            absorbed_intercepts.add(intercept_key)

    # Remove absorbed intercept groups so they are not plotted standalone
    for key in absorbed_intercepts:
        del groups[key]

    return groups


def _feature_delay_mask(model, group_name):
    """Return a boolean mask over delays for an event-scoped group."""
    n_delays = len(model.delays_)
    times = model.times_
    event_type = group_name.split(":")[0] if ":" in group_name else None
    if event_type and hasattr(model, "analysis_windows"):
        win = model.analysis_windows.get(event_type)
        if win is not None:
            mask = (times >= win[0]) & (times <= win[1])
            if np.any(mask):
                return mask
    return np.ones(n_delays, dtype=bool)


# ---------------------------------------------------------------------------
# Lightweight TRF plot (no MNE dependency)
# ---------------------------------------------------------------------------


def plot_trfs_butterfly(model, features=None, figsize=None, baseline=None):
    """Plot TRFs as mean ± SEM across channels (pure matplotlib).

    Spline bases belonging to the same feature are grouped on one
    subplot so their individual contributions are easy to compare.

    Parameters
    ----------
    model : pydeconv.core.DeconvolutionModel
        A fitted model with ``coef_`` available.
    features : list of str, optional
        Group names (or individual feature names) to include.
        If ``None``, all features are plotted.
    figsize : tuple, optional
        Figure size. Defaults to ``(5 * n_groups, 4)``.
    baseline : tuple of float or None, optional
        The time interval (a, b) in seconds to use for baseline correction.
        If a is None, it defaults to the start of the time window.
        If b is None, it defaults to the end of the time window.
        Baseline correction subtracts the mean of the baseline period
        for each channel. If None, no correction is applied.

    Returns
    -------
    fig : matplotlib.figure.Figure

    """
    if getattr(model, "coef_", None) is None:
        raise ValueError("Model is not fitted. Cannot plot TRFs.")

    coef = model.coef_
    if coef.ndim == 1:
        coef = coef.reshape(1, -1)

    n_delays = len(model.delays_)
    times = model.times_
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    groups = _group_features(model.feature_names_)

    if features is not None:
        groups = {k: v for k, v in groups.items() if k in features}

    if not groups:
        raise ValueError("No matching feature groups found to plot.")

    n_groups = len(groups)
    if figsize is None:
        figsize = (5 * n_groups, 4)

    fig, axes = plt.subplots(1, n_groups, figsize=figsize, squeeze=False)

    for ax_idx, (group_name, feat_names) in enumerate(groups.items()):
        ax = axes[0, ax_idx]
        delay_mask = _feature_delay_mask(model, group_name)
        t_plot = times[delay_mask]

        # Sum up coefficients for all features in the group.
        # _group_features already absorbs the event intercept into spline
        # groups, so a simple sum over all members gives the total TRF.
        trf_group = np.zeros((coef.shape[0], n_delays))
        for feat_name in feat_names:
            feat_idx = model.feature_names_.index(feat_name)
            start = feat_idx * n_delays
            end = start + n_delays
            trf_group += coef[:, start:end]

        # Apply delay mask
        trf = trf_group[:, delay_mask].copy()

        if baseline is not None:
            bmin, bmax = baseline
            bmin = bmin if bmin is not None else t_plot[0]
            bmax = bmax if bmax is not None else t_plot[-1]
            base_mask = (t_plot >= bmin) & (t_plot <= bmax)
            if np.any(base_mask):
                base_mean = trf[:, base_mask].mean(axis=1, keepdims=True)
                trf = trf - base_mean

        label = group_name.split(":")[-1] if ":" in group_name else group_name
        # Get the color for this feature from the default color cycle
        group_idx = list(groups.keys()).index(group_name)
        color = colors[group_idx % len(colors)]
        for ch_idx in range(trf.shape[0]):
            lbl = label if ch_idx == 0 else None
            ax.plot(t_plot, trf[ch_idx], lw=0.7, alpha=0.4, color=color, label=lbl)

        ax.axhline(0, color="k", lw=0.5, ls="--")
        ax.axvline(0, color="k", lw=0.5, ls=":")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Coefficient (a.u.)")
        title = group_name.replace(":", " → ")
        ax.set_title(title, fontweight="bold")
        ax.legend(fontsize=7, loc="best")

    fig.suptitle(
        "Temporal Response Functions (Butterfly Plot)",
        fontweight="bold", y=1.02,
    )
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Design matrix heatmap
# ---------------------------------------------------------------------------


def plot_design_matrix(
    model,
    X,
    sfreq=None,
    snippet_start=0,
    snippet_duration_s=5.0,
    figsize=(12, 4),
):
    """Plot a heatmap of the design matrix at delay = 0.

    Shows one column per feature (at the zero-delay slice) over a short
    time window for visual inspection.

    Parameters
    ----------
    model : pydeconv.core.DeconvolutionModel
        A model whose ``feature_names_`` and ``delays_`` are populated
        (i.e. ``build_design_matrix`` has been called).
    X : numpy.ndarray, shape ``(n_samples, n_features * n_delays)``
        The full design matrix.
    sfreq : float, optional
        Sampling frequency for the time axis. Defaults to ``model.sfreq``.
    snippet_start : int
        First sample index to display.
    snippet_duration_s : float
        Duration of the snippet to display, in seconds.
    figsize : tuple
        Figure size.

    Returns
    -------
    fig : matplotlib.figure.Figure

    """
    if sfreq is None:
        sfreq = model.sfreq

    n_delays = len(model.delays_)
    snippet_end = min(
        snippet_start + int(snippet_duration_s * sfreq), X.shape[0],
    )

    zero_delay_idx = int(np.argmin(np.abs(model.delays_)))
    col_indices = [
        i * n_delays + zero_delay_idx for i in range(len(model.feature_names_))
    ]
    X_plot = X[snippet_start:snippet_end, :][:, col_indices]
    if hasattr(X_plot, "toarray"):
        X_plot = X_plot.toarray()

    t_snippet = np.arange(snippet_start, snippet_end) / sfreq

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.pcolormesh(
        t_snippet, np.arange(X_plot.shape[1]),
        X_plot.T, cmap="RdBu_r", shading="auto",
    )
    ax.set_yticks(np.arange(len(model.feature_names_)))
    ax.set_yticklabels(model.feature_names_, fontsize=7)
    ax.set_xlabel("Time (s)")
    ax.set_title("Design Matrix (delay = 0 slice)", fontweight="bold")
    plt.colorbar(im, ax=ax, label="Value")
    plt.tight_layout()
    return fig


def plot_trfs(model, info=None, features=None, top_topos=True, figsize=(15, 8), baseline=None):
    """Plot the fitted Temporal Response Functions (TRFs) using MNE.

    This uses a horizontal layout inspired by the legacy PyDeconv plots,
    where each feature-group gets its own column consisting of a butterfly
    plot and (optionally) topomaps at peak times.

    Spline bases belonging to the same feature are summed into a single
    TRF, and the corresponding event intercept is automatically added
    so the plot shows the total response.

    Parameters
    ----------
    model : pydeconv.core.DeconvolutionModel
        A fitted DeconvolutionModel containing ``coef_``.
    info : mne.Info, optional
        The MNE Info object corresponding to the channels used to fit the model.
        If None, a dummy Info object is created automatically.
    features : list of str, optional
        A list of group names to plot.  If ``None``, all groups
        (except standalone intercepts already absorbed into spline
        groups) are plotted.
    top_topos : bool, default True
        If True, plots joint time-series and topomaps (mne.Evoked.plot_joint).
        If False, only plots the butterfly time-series.
    figsize : tuple, default (15, 8)
        The overall figure size.
    baseline : tuple of float or None, optional
        The time interval (a, b) in seconds to use for baseline correction.
        If a is None, it defaults to the start of the time window.
        If b is None, it defaults to the end of the time window.
        If None, no baseline correction is applied.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The resulting figure.

    """
    import mne

    if getattr(model, "coef_", None) is None:
        raise ValueError("Model is not fitted. Cannot plot TRFs.")

    n_delays = len(model.delays_)
    times = getattr(model, "times_", np.arange(n_delays))

    coef = model.coef_
    if coef.ndim == 1:
        coef = coef[np.newaxis, :]

    # Use feature grouping (splines collapsed, intercepts absorbed)
    groups = _group_features(model.feature_names_)

    if features is not None:
        groups = {k: v for k, v in groups.items() if k in features}
    else:
        # Exclude any remaining standalone "intercept" (global)
        groups = {k: v for k, v in groups.items() if k != "intercept"}

    if not groups:
        raise ValueError("No matching feature groups found to plot.")

    if info is None:
        n_channels = coef.shape[0]
        ch_names = [f"ch_{i}" for i in range(n_channels)]
        info = mne.create_info(
            ch_names=ch_names, sfreq=model.sfreq,
            ch_types=["eeg"] * n_channels,
        )
        top_topos = False  # Dummy info has no sensor coordinates

    n_groups = len(groups)
    fig = plt.figure(figsize=figsize)

    horizontal_jump = 0.8 / n_groups

    for jump, (group_name, feat_names) in enumerate(groups.items()):
        delay_mask = _feature_delay_mask(model, group_name)

        # Sum coefficients for every member of the group
        trf_group = np.zeros((coef.shape[0], n_delays))
        for feat_name in feat_names:
            feat_idx = model.feature_names_.index(feat_name)
            start = feat_idx * n_delays
            end = start + n_delays
            trf_group += coef[:, start:end]

        data = trf_group[:, delay_mask].copy()
        times_feat = times[delay_mask]
        x_lims = (times_feat[0], times_feat[-1])

        # Create an Evoked object
        grand_avg = mne.EvokedArray(
            data, info, tmin=times_feat[0], verbose=False,
        )
        grand_avg.nave = None
        if baseline is not None:
            grand_avg.apply_baseline(baseline, verbose=False)
            data = grand_avg.data

        # Determine global max for symmetric colormap
        vmax = np.max(np.abs(data))
        vlim = (-vmax, vmax)

        # Calculate horizontal position dynamically
        x0 = 0.05 + jump * horizontal_jump
        width = horizontal_jump - 0.05

        ax_frp = fig.add_axes((x0, 0.47, width, 0.2))

        if top_topos:
            topo_w = width * 0.25
            gap = width * 0.05
            ax_topo1 = fig.add_axes((x0, 0.75, topo_w, 0.15))
            ax_topo2 = fig.add_axes((x0 + topo_w + gap, 0.75, topo_w, 0.15))
            ax_topo3 = fig.add_axes(
                (x0 + 2 * (topo_w + gap), 0.75, topo_w, 0.15),
            )
            ax_topo_cb = fig.add_axes(
                (x0 + 3 * (topo_w + gap), 0.75, width * 0.02, 0.15),
            )
            axs_topos = [ax_topo1, ax_topo2, ax_topo3, ax_topo_cb]

            grand_avg.plot_joint(
                title="",
                ts_args={
                    'xlim': x_lims, 'axes': ax_frp,
                    'titles': dict(eeg=''), 'window_title': '',
                },
                topomap_args={
                    'vlim': vlim, 'contours': 2,
                    'axes': axs_topos, 'size': 0.8,
                },
                show=False,
            )

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
                show=False,
            )

        # Clean up axes
        ax_frp.set_xlabel("Time (s)")
        title = group_name.replace(":", " → ")
        ax_frp.set_title(title, fontweight="bold", pad=15)
        if jump > 0:
            ax_frp.set_ylabel("")
            ax_frp.set_yticklabels([])

        for c in ax_frp.get_children():
            if isinstance(c, plt.Text) and 'channels' in c.get_text():
                c.remove()

    fig.legends = []
    return fig
