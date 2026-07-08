"""01 — Simulation Round-Trip
==========================
Simulate continuous EEG data with known event-related kernels, then run
the full PyDeconv pipeline to recover them. This script serves as a
sanity check: if the recovered coefficients match the ground truth,
the pipeline works end to end.

Steps
-----
1. Define an experiment design (event types, timing, covariates).
2. Create ground-truth kernels (Gaussian bumps at known latencies).
3. Simulate continuous EEG by convolving events with kernels + noise.
4. Build a design matrix from the simulated event structure.
5. Fit a regularised regression model (e.g. Ridge).
6. Extract and plot recovered kernels against the ground truth.
7. Compute goodness-of-fit metrics (R², Pearson r, AIC).
"""

import os
import sys

# Ensure the parent directory is in the Python path so 'pydeconv' can be imported
# when running this script directly.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from pydeconv.core import DeconvolutionModel
from pydeconv.simulation.simulation import (
    ExperimentDesign,
    EEGSimulator,
    CompoundKernel,
)
# from pydeconv.utils.metrics import calculate_pearson_r, calculate_aic
from pydeconv.utils.plotting import plot_simulation_kernels, plot_trfs
from pydeconv.estimators import Tridge
# ── 1. Experiment design ─────────────────────────────────────────────
design = ExperimentDesign(
    n_events=200,
    event_types=["stimulus", "response"],
    sfreq=256,
    duration_s=600,
)
events_df = design.generate_events()
events_df.head()

# ── 2. Ground-truth kernels ──────────────────────────────────────────
# Define compound kernels for stimulus and response
stim_kernel = CompoundKernel("stimulus", sfreq=256)
stim_kernel.add(peak_latency=0.10, amplitude=5.0, width=0.03)
stim_kernel.add(peak_latency=0.20, amplitude=-3.0, width=0.05)

resp_kernel = CompoundKernel("response", sfreq=256)
resp_kernel.add(peak_latency=0.05, amplitude=4.0, width=0.02)

# ── 3. Simulate EEG ─────────────────────────────────────────────────
simulator = EEGSimulator(sfreq=256, duration=600)
# Add kernels to the simulator, specifying which events they respond to
simulator.add_kernel(stim_kernel, activation=lambda row: row["type"] == "stimulus")
simulator.add_kernel(resp_kernel, activation=lambda row: row["type"] == "response")

# Generate the single-channel data
simulator.set_events(events_df)
y = simulator.simulate()
simulator.add_noise(colour="pink", scale=0.5)
y_noisy = simulator.data

# ── 4. create a model and design matrix ───────────────────────────────────────────
#scikit learn stimator
# model = (
#     DeconvolutionModel(tmin=-0.1, tmax=0.5, sfreq=256)
#     # name == from_event registers an event-specific intercept.
#     .add_feature("stimulus", from_event="stimulus")
#     .add_feature("response", from_event="response")
# )
#custom ridge estimator
model = (
    DeconvolutionModel(tmin=-0.1, tmax=0.5, sfreq=256, estimator=Tridge(alpha=1.0, use_gpu=False))
    # name == from_event registers an event-specific intercept.
    .add_feature("stimulus", from_event="stimulus")
    .add_feature("response", from_event="response")
    .add_new_analysis_window("response",tmin=-0.1, tmax=0.2)

)

# alternative would be to use the more efficient Gram matrix approach, but for now we stick with the default design matrix approach.
# X = model.build_gram_matrix(events_df, n_samples=len(y_noisy), use_gpu=False)
X = model.build_design_matrix(events_df, n_samples=len(y_noisy), use_gpu=False)
print("Design matrix shape:", X.shape)

# ── 5. Fit model ─────────────────────────────────────────────────────
model.fit(X, y_noisy, standardize=True)
print("Model fitted successfully.")
print("Coefficients shape:", model.coef_.shape)

# ── 6. Compare recovered vs ground truth ─────────────────────────────

plot_simulation_kernels(simulator)
plot_trfs(model, features=["stimulus:intercept", "response:intercept"])

# ── 7. Metrics ───────────────────────────────────────────────────────
score = model.score(X, y_noisy)
print(f"R² score: {score:.4f}")

import matplotlib.pyplot as plt
plt.show()
