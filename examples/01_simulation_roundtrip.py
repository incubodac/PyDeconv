"""
01 — Simulation Round-Trip
==========================
Simulate continuous EEG data with known event-related kernels, then run
the full PyDeconv pipeline to recover them. This script serves as a
sanity check: if the recovered coefficients match the ground truth,
the pipeline works end to end.

Steps
-----
1. Define an experiment design (event types, timing, covariates).
2. Create ground-truth kernels (Gaussian bumps at known latencies).
3. Simulate multi-channel EEG by convolving events with kernels + noise.
4. Build a design matrix from the simulated event structure.
5. Fit a regularised regression model (e.g. Ridge).
6. Extract and plot recovered kernels against the ground truth.
7. Compute goodness-of-fit metrics (R², Pearson r, AIC).
"""

# TODO: import from pydeconv once modules are implemented
# from pydeconv import PyDeconv
# from pydeconv.simulation import ExperimentDesign, EEGSimulator
# from pydeconv.utils.design_matrix import create_design_matrix, shifted_matrix
# from pydeconv.utils.metrics import calculate_pearson_r, calculate_aic
# from pydeconv.utils.plotting import plot_coefficients, plot_design_matrix

# ── 1. Experiment design ─────────────────────────────────────────────
# design = ExperimentDesign(
#     n_events=200,
#     event_types=["stimulus", "response"],
#     sfreq=256,
#     duration_s=600,
# )

# ── 2. Ground-truth kernels ──────────────────────────────────────────
# kernels = {
#     "stimulus": [{"onset": 0.10, "amplitude": 5.0, "width": 0.03},
#                  {"onset": 0.20, "amplitude": -3.0, "width": 0.05}],
#     "response": [{"onset": 0.05, "amplitude": 4.0, "width": 0.02}],
# }

# ── 3. Simulate EEG ─────────────────────────────────────────────────
# simulator = EEGSimulator(design, kernels, n_channels=64, noise_std=0.5)
# raw, events_df = simulator.generate()

# ── 4. Build design matrix ───────────────────────────────────────────
# X = create_design_matrix(raw, events_df, tmin=-0.1, tmax=0.5)

# ── 5. Fit model ─────────────────────────────────────────────────────
# model = PyDeconv(estimator="ridge", tmin=-0.1, tmax=0.5, sfreq=256)
# model.fit(X, raw.get_data().T)

# ── 6. Compare recovered vs ground truth ─────────────────────────────
# plot_coefficients(model, ground_truth=kernels)

# ── 7. Metrics ───────────────────────────────────────────────────────
# print("R² per channel:", model.score(X, raw.get_data().T))
