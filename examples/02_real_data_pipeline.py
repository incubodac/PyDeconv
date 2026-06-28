"""
02 — Real Data Pipeline
=======================
Load a real EEG dataset (EEGLAB .set format), pair it with a behavioural
events table, build a design matrix with spline features, fit a deconvolution
model, and visualise the results.

Steps
-----
1. Load EEG data from a .set file via MNE.
2. Load events from a CSV / DataFrame (latency, type, covariates).
3. Inspect event statistics (counts per condition, overlap analysis).
4. Optionally apply window-based artifact rejection.
5. Build the design matrix (intercept, additive features, interactions, splines).
6. Check collinearity with VIF.
7. Fit the model (Ridge regression with cross-validated alpha).
8. Evaluate fit quality: R², Pearson r, AIC per channel.
9. Plot recovered kernels, topographies, and design matrix heatmap.
10. Save model coefficients for group analysis.
"""

# TODO: import from pydeconv once modules are implemented
# from pydeconv import PyDeconv
# from pydeconv.utils.io import load_set_file
# from pydeconv.utils.design_matrix import create_design_matrix
# from pydeconv.utils.metrics import calculate_vif, calculate_aic, calculate_pearson_r
# from pydeconv.utils.event_stats import compute_event_stats
# from pydeconv.utils.window_rejection import cont_ArtifactDetect
# from pydeconv.utils.plotting import (
#     plot_coefficients,
#     plot_design_matrix,
#     plot_event_stats,
# )

# ── 1. Load data ─────────────────────────────────────────────────────
# raw = load_set_file("path/to/subject01.set")
# events_df = pd.read_csv("path/to/subject01_events.csv")

# ── 2. Event statistics ──────────────────────────────────────────────
# stats = compute_event_stats(events_df, sfreq=raw.info["sfreq"])
# plot_event_stats(stats)
# print(stats.summary())

# ── 3. Artifact rejection ────────────────────────────────────────────
# bad_windows = cont_ArtifactDetect(
#     raw, amplitudeThreshold=150, windowsize=2000, stepsize=100
# )

# ── 4. Design matrix ─────────────────────────────────────────────────
# X = create_design_matrix(
#     raw, events_df,
#     tmin=-0.1, tmax=0.6,
#     features=["condition", "rt"],
#     interactions=[("condition", "rt")],
#     use_splines={"rt": 5},
#     bad_windows=bad_windows,
# )

# ── 5. Collinearity check ────────────────────────────────────────────
# vif_scores = calculate_vif(X)
# print("VIF per predictor:", vif_scores)

# ── 6. Fit model ─────────────────────────────────────────────────────
# model = PyDeconv(estimator="ridge", tmin=-0.1, tmax=0.6, sfreq=raw.info["sfreq"])
# model.fit(X, raw.get_data(picks="eeg").T)

# ── 7. Evaluate ──────────────────────────────────────────────────────
# scores = model.score(X, raw.get_data(picks="eeg").T)
# aic = calculate_aic(raw.get_data(picks="eeg").T, model.predict(X), X)
# pearson = calculate_pearson_r(raw.get_data(picks="eeg").T, model.predict(X))
# print(f"Mean R²: {scores.mean():.4f}  |  Mean AIC: {aic.mean():.1f}")

# ── 8. Plot ──────────────────────────────────────────────────────────
# plot_coefficients(model)
# plot_design_matrix(X)

# ── 9. Save ──────────────────────────────────────────────────────────
# model.save_coefficients("output/subject01_coeffs.npz")
