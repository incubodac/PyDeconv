# PyDeconv – Copilot Instructions

## Project Overview

PyDeconv is a scientific Python package for EEG/MEG deconvolution analysis. It builds linear regression models from experimental event features (including interactions and B-splines) and fits them to continuous neural data using regularized regression.

Key classes: `PyDeconv` (main deconvolution pipeline), `ExperimentDesign` + `EEGSimulator_v2` (simulation module).

## Code Style

- **PEP 8** for all Python code. Use 4-space indentation, snake_case for functions/variables, PascalCase for classes.
- Keep lines under 100 characters where practical.
- Use f-strings for string formatting.
- Use type hints for public method signatures (parameters and return types).
- Prefer NumPy vectorized operations over Python loops for numerical work.
- Do not leave dead code, debug prints, or commented-out blocks in production code. Clean up before committing.

## Docstrings

- Use NumPy-style docstrings for all public classes and methods.
- Include `Parameters`, `Returns`, and a brief one-line summary at minimum.
- Private/internal methods (`_prefixed`) need only a one-line summary.

## Architecture

```
pydeconv/
  __init__.py          # Public API exports
  pydeconv.py          # Core deconvolution: PyDeconv class
  pydeconv_sims.py     # Simulation: ExperimentDesign, EEGSimulator_v2
  utils/
    functions.py       # Design matrix construction helpers
    analysis_functions.py
    plot_general.py    # Plotting utilities (plot_model_results, etc.)
    load.py            # Data loading (EEGLAB .set files)
    TFCE.py            # Threshold-free cluster enhancement
    winrej.py          # Window rejection
config.py              # Real-data analysis config
config4test.py         # Simulation test config
```

- Keep simulation code (`ExperimentDesign`, `EEGSimulator_v2`) in `pydeconv_sims.py`.
- Keep plotting utilities in `pydeconv/utils/plot_general.py`.
- Config files at the repo root define model parameters (formula, tmin/tmax, solver, etc.).

## Conventions

- EEG data flows as MNE `Raw` objects; events as `pandas.DataFrame` with a `latency` column (in samples) and a `type` column.
- `EEGSimulator_v2` stores latencies in both samples (`latency`) and seconds (`latency_s`). Always use `latency_s` for plotting against time axes.
- Kernels are defined as lists of Gaussian components: `{'onset', 'amplitude', 'width'}`.
- When sklearn models return `coef_`, always handle the case where it may be 1-D (single channel) by reshaping to 2-D before indexing.
- Place all class/function definitions **before** `if __name__ == '__main__':` blocks.

## Testing

- Primary testing is done via Jupyter notebooks (`test_and_sims.ipynb`, `test.ipynb`).
- Use `%autoreload 2` in notebooks. After structural changes to modules, restart the kernel.
- When adding or modifying simulation methods, verify by running the simulation cell and visually inspecting plots (data signal + PSD, kernel waveforms).

## Dependencies

Python ≥ 3.9, NumPy, SciPy, Pandas, MNE, scikit-learn, Matplotlib. Do not introduce new dependencies without discussion.
