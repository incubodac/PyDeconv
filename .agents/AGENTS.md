# PyDeconv – Project Rules

> These rules are automatically loaded on every inference in this workspace.

## General idea
Now I watn to start a tidy package of python for EEG or MEG deconvolution itll be fed with mne raw or numpy array  (chans x time) data and also a pandas/polar dataframe or numpy array. then one may use the main functions like create_design_matrix(data, events) from command line and use THE  plotting methods , from the package we are developing, or create a group and then call the TFCE from mne to make a group analysis.


## Project Overview

PyDeconv is a scientific Python package for EEG/MEG deconvolution analysis. It builds linear regression models from experimental event features (including interactions and B-splines) and fits them to continuous neural data using regularized regression.

The repository is undergoing a rewrite. All legacy code is archived under `old/`. New code should be written from scratch at the repo root.

## Code Style

- **PEP 8** for all Python code. Use 4-space indentation, `snake_case` for functions/variables, `PascalCase` for classes.
- Keep lines under 100 characters where practical.
- Use f-strings for string formatting.
- Use type hints for all public method signatures (parameters and return types).
- Prefer NumPy vectorized operations over Python loops for numerical work.
- Do not leave dead code, debug prints, or commented-out blocks in production code.

## Docstrings

- Use NumPy-style docstrings for all public classes and methods.
- Include `Parameters`, `Returns`, and a brief one-line summary at minimum.
- Private/internal methods (`_prefixed`) need only a one-line summary.

## Architecture (Target)

```
pydeconv/
  __init__.py              # Public API exports
  core.py                  # Core deconvolution pipeline
  simulation.py            # Simulation: experiment design, EEG simulation
  utils/
    design_matrix.py       # Design matrix construction helpers
    analysis.py            # Analysis and scoring functions
    plotting.py            # Plotting utilities
    io.py                  # Data loading (EEGLAB .set files, etc.)
    tfce.py                # Threshold-free cluster enhancement
    window_rejection.py    # Window rejection utilities
```

- Keep simulation code separate from the core deconvolution pipeline.
- Keep plotting utilities in `pydeconv/utils/plotting.py`.
- Config files at the repo root define model parameters (formula, tmin/tmax, solver, etc.).

## Conventions

- EEG data flows as MNE `Raw` objects; events as `pandas.DataFrame` with a `latency` column (in samples) and a `type` column.
- Simulation latencies should be stored in both samples (`latency`) and seconds (`latency_s`). Always use `latency_s` for plotting against time axes.
- Kernels are defined as lists of Gaussian components: `{'onset', 'amplitude', 'width'}`.
- When sklearn models return `coef_`, always handle the case where it may be 1-D (single channel) by reshaping to 2-D before indexing.
- Place all class/function definitions **before** `if __name__ == '__main__':` blocks.

## Dependencies

Python ≥ 3.9, NumPy, SciPy, Pandas, MNE, scikit-learn, Matplotlib. Do not introduce new dependencies without discussion.

## Testing

- Write unit tests using `pytest` for new modules.
- Notebooks in `source_examples/` serve as integration/usage examples, not as the primary test suite.
- After structural changes, verify by running relevant tests and visually inspecting plots where applicable.

## Legacy Code

- All previous code is archived in the `old/` directory for reference only.
- Do not import from or modify files under `old/`.
- When reimplementing functionality, use `old/` as a reference but write clean, well-documented code from scratch.
