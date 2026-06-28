"""
03 — Group Analysis with TFCE
=============================
Run threshold-free cluster enhancement (TFCE) across multiple subjects
to identify statistically significant spatio-temporal clusters in the
recovered deconvolution coefficients.

Steps
-----
1. Load pre-computed model coefficients for each subject.
2. Stack coefficients into a group observations array (subjects × time × channels).
3. Compute channel adjacency from montage information.
4. Run TFCE permutation cluster test (wraps MNE's implementation).
5. Visualise significant clusters as topographic maps over time.
"""

# TODO: import from pydeconv once modules are implemented
# import numpy as np
# from pydeconv.utils.tfce import tfce, get_channel_adjacency
# from pydeconv.utils.io import load_set_file
# from pydeconv.utils.plotting import plot_tfce_clusters

# ── 1. Load coefficients per subject ─────────────────────────────────
# subject_ids = ["sub01", "sub02", "sub03", "sub04", "sub05"]
# coefficients = []
# for sid in subject_ids:
#     data = np.load(f"output/{sid}_coeffs.npz")
#     coefficients.append(data["kernel_stimulus"])  # shape: (n_times, n_channels)
# observations = np.stack(coefficients, axis=0)  # shape: (n_subjects, n_times, n_channels)

# ── 2. Channel adjacency ────────────────────────────────────────────
# # Use the info from any subject (all share the same montage)
# raw = load_set_file(f"data/{subject_ids[0]}.set")
# ch_adjacency = get_channel_adjacency(raw.info)

# ── 3. Run TFCE ──────────────────────────────────────────────────────
# clusters_mask, pval_threshold = tfce(
#     observations,
#     ch_adjacency,
#     n_permutations=1024,
#     alpha=0.05,
# )
# print(f"Significant voxels at α={pval_threshold}: {clusters_mask.sum()}")

# ── 4. Visualise ─────────────────────────────────────────────────────
# plot_tfce_clusters(clusters_mask, raw.info, times=np.linspace(-0.1, 0.5, observations.shape[1]))
