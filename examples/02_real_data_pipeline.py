"""02 — Real Data Pipeline
=======================
Load a real EEG dataset (EEGLAB .set format), pair it with a behavioural
events table, build a design matrix with spline features, fit a deconvolution
model, and visualise the results.

Steps
-----
1. Load EEG data from a .set file via MNE.
2. Load events from a CSV (latency, type, covariates).
3. Inspect event statistics.
4. Build the design matrix (intercept, additive features, splines).
5. Fit the model (Ridge regression).
6. Evaluate fit quality (R² per channel).
7. Plot recovered TRFs.
"""

import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for scripts
from pathlib import Path
from sklearn.linear_model import Ridge

# ── Ensure pydeconv is importable when running from /examples ────────
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pydeconv.core import DeconvolutionModel

# ── Paths ────────────────────────────────────────────────────────────
DATA_DIR = Path(__file__).resolve().parent.parent / "examples" / "example_data"
SET_FILE = DATA_DIR / "629959_analysis.set"
EVENTS_CSV = DATA_DIR / "629959_full_metadata.csv"
OUTPUT_DIR = Path(__file__).resolve().parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

# ── 1. Load EEG data ────────────────────────────────────────────────
import mne

print("Loading EEG data …")
raw = mne.io.read_raw_eeglab(str(SET_FILE), preload=True, verbose=False)

# Keep only EEG channels (drop EXG misc and eye-tracking channels)
eeg_picks = mne.pick_types(raw.info, eeg=True, misc=False, exclude="bads")
raw.pick(eeg_picks)

sfreq = raw.info["sfreq"]
n_channels = len(raw.ch_names)
n_samples = raw.n_times
print(f"  sfreq={sfreq} Hz | {n_channels} EEG channels | "
      f"{n_samples} samples ({n_samples / sfreq:.1f} s)")

# EEG data as (n_samples, n_channels)
y = raw.get_data().T  # shape: (n_samples, n_channels)

# ── 2. Load events ──────────────────────────────────────────────────
print("Loading events …")
events_df = pd.read_csv(EVENTS_CSV)

# Ensure latency is integer (samples)
events_df["latency"] = events_df["latency"].astype(int)

# Filter out events beyond recording length
events_df = events_df[events_df["latency"] < n_samples].copy()
events_df = events_df[events_df["latency"] >= 0].copy()

print(f"  {len(events_df)} events total")
print(f"  Event types: {dict(events_df['type'].value_counts())}")
print(f"  Columns available: {list(events_df.columns)}")

# ── 3. Event statistics ─────────────────────────────────────────────
print("\n── Event Statistics ──")
for etype in events_df["type"].unique():
    sub = events_df[events_df["type"] == etype]
    isi = np.diff(np.sort(sub["latency"].values)) / sfreq
    print(f"  {etype}: n={len(sub)}, "
          f"median ISI={np.median(isi):.3f} s, "
          f"min ISI={isi.min():.3f} s")

# ── 4. Define the deconvolution model ───────────────────────────────
# Events should be present in the events dataframe at the "type" column.
#
# The model includes:
#   - fixation intercept: one kernel per fixation event
#   - saccade intercept: one kernel per saccade event
#   - ontarget: a binary covariate (was fixation on target?) for fixation events
#   - sac_amplitude splines: B-spline expansion of saccade amplitude (5 bases)
#   - A shorter analysis window for saccade events (-0.1 to 0.2 s)

print("\n── Model Definition ──")
model = (
    DeconvolutionModel(
        tmin=-0.1,
        tmax=0.5,
        sfreq=sfreq,
        estimator=Ridge(alpha=100.0),
    )
    # Event-specific intercepts (name == from_event → intercept)
    .add_feature("fixation", from_event="fixation")
    .add_feature("saccade", from_event="saccade")
    # Continuous covariate: "ontarget" for fixation events
    .add_feature("ontarget", column="ontarget", from_event="fixation")
    # B-spline expansion of saccade amplitude
    .add_feature_splines(
        "sac_amp", column="sac_amplitude",
        from_event="saccade", n_splines=5,
        knot_method="quantile", intercept=False,
    )
    # Narrower analysis window for saccade events
    .add_new_analysis_window("saccade", tmin=-0.1, tmax=0.3)
)

print(f"  {model}")

# ── 5. Build design matrix ──────────────────────────────────────────
print("\n── Building Design Matrix ──")
X = model.build_design_matrix(events_df, n_samples=n_samples, use_gpu=False)
print(f"  X shape: {X.shape}")
print(f"  Feature names: {model.feature_names_}")
print(f"  Delays: {len(model.delays_)} "
      f"({model.delays_[0]}..{model.delays_[-1]} samples, "
      f"{model.times_[0]:.3f}..{model.times_[-1]:.3f} s)")

# ── 6. Fit model ────────────────────────────────────────────────────
print("\n── Fitting Model ──")
model.fit(X, y, standardize=True)
print("  Model fitted successfully.")

# ── 7. Evaluate ─────────────────────────────────────────────────────
print("\n── Evaluation ──")
scores = model.score(X, y)
if np.ndim(scores) == 0:
    print(f"  R² = {scores:.4f}")
else:
    print(f"  R² per channel: mean={np.mean(scores):.4f}, "
          f"median={np.median(scores):.4f}, "
          f"max={np.max(scores):.4f}")
    # Show top 5 channels
    top5 = np.argsort(scores)[-5:][::-1]
    for idx in top5:
        print(f"    {raw.ch_names[idx]}: R²={scores[idx]:.4f}")

# ── 8. Plot TRFs ────────────────────────────────────────────────────
print("\n── Plotting TRFs ──")
from pydeconv.utils.plotting import plot_trfs_butterfly, plot_design_matrix

fig_trf = plot_trfs_butterfly(model)
trf_path = OUTPUT_DIR / "02_trfs.png"
fig_trf.savefig(trf_path, dpi=150, bbox_inches="tight")
print(f"  TRF plot saved to {trf_path}")

# ── 9. Plot design matrix snippet ───────────────────────────────────
print("\n── Plotting Design Matrix Snippet ──")
snippet_start = max(0, int(events_df["latency"].iloc[0] - 0.5 * sfreq))
fig_dm = plot_design_matrix(
    model, X, snippet_start=snippet_start, snippet_duration_s=5.0,
)
dm_path = OUTPUT_DIR / "02_design_matrix.png"
fig_dm.savefig(dm_path, dpi=150, bbox_inches="tight")
print(f"  Design matrix plot saved to {dm_path}")

# ── 10. Save coefficients ───────────────────────────────────────────
coef_path = OUTPUT_DIR / "02_coefficients.npz"
np.savez(
    coef_path,
    coef=model.coef_,
    feature_names=model.feature_names_,
    times=model.times_,
    delays=model.delays_,
    sfreq=sfreq,
    ch_names=raw.ch_names,
)
print(f"\n  Coefficients saved to {coef_path}")
print("\n✅ Pipeline complete!")
