"""
04 — GUI Demo
=============
Launch the PyDeconv graphical interface for interactive deconvolution
analysis. The GUI allows loading data, configuring the design matrix,
fitting models, and inspecting results — all without writing code.

Requirements
------------
Install the GUI extras first:
    pip install pydeconv[gui]

Usage
-----
Run this script or use the command line:
    python -m pydeconv.gui
"""

# TODO: import from pydeconv.gui once implemented
# from pydeconv.gui import launch
#
# launch()
#
# The GUI will provide panels for:
#   - Data loading:   Select .set / .fdt files or MNE Raw objects
#   - Events:         Load CSV events, inspect per-condition statistics
#   - Design matrix:  Choose features, interactions, splines; preview heatmap
#   - Model config:   Select solver (Ridge, Lasso), set tmin/tmax, alpha range
#   - Fitting:        Run fit with progress bar, view cross-validation curves
#   - Results:        Plot recovered kernels, topographies, metrics (R², AIC, VIF)
#   - Group:          Load multiple subjects, run TFCE, visualise clusters
