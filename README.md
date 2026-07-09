# PyDeconv

PyDeconv is an open-source Python package for EEG/MEG **deconvolution analysis** — the
technique of separating overlapping event-related brain responses from continuous
neural recordings.

Classical ERP approaches assume events are well-separated in time; in practice they
overlap (fixations, audio onsets, button presses, etc. all co-occur).  PyDeconv handles
this by building a **time-expanded design matrix** that accounts for every event
simultaneously and solves a single regularised linear regression, yielding one temporal
response function (TRF / rERP) per predictor — even for overlapping event streams.

## Features

- Deconvolution of overlapping EEG/MEG events via time-expanded design matrices
- Fluent **builder API** for defining additive features, interactions, and per-event
  analysis windows
- **B-spline basis expansion** for non-linear modelling of continuous covariates
- Collinearity diagnostics (VIF)
- Regularised solvers: Ridge, Tridge (Ridge regression with Torch compatibility), and any
  scikit-learn-compatible estimator
- Group-level statistics via TFCE permutation tests (wraps MNE)
- Interactive **GUI** built with PySide6

## Inputs

| Argument | Accepted formats |
|---|---|
| EEG data | `mne.io.Raw` object **or** NumPy array `(channels × samples)` |
| Events / features | `pandas.DataFrame` (requires a `latency` column in samples) **or** NumPy array |

## Quick start

```python
import numpy as np
from pydeconv.core import DeconvolutionModel

model = (
    DeconvolutionModel(tmin=-0.1, tmax=0.6, sfreq=256)

    # Intercept (mean TRF) for each event type
    .add_feature("fixation", from_event="fixation")
    .add_feature("audio",    from_event="audio")
    .add_feature("button",   from_event="button")

    # Additive covariate with optional transform
    .add_feature("log_rt", column="reaction_time",
                 from_event="button", transform=np.log)

    # Interaction term
    .add_interaction("log_rt", "condition", event_type="button")

    # Event-specific analysis window
    .add_new_analysis_window("button", tmin=-0.05, tmax=0.3)
)

# Build design matrix and fit
X = model.build_design_matrix(events_df, n_samples=raw.n_times)
model.fit(X, raw.get_data().T)
```

See [`examples/`](examples/) for complete runnable scripts.

## Installation

```bash
pip install -e .          # editable install from the repo root
```

For the GUI, also install PySide6:

```bash
pip install PySide6
python -m pydeconv.gui
```

## Dependencies

- [Python](https://www.python.org) ≥ 3.9
- [MNE](https://mne.tools/stable/index.html) ≥ 1.3.1
- [NumPy](https://numpy.org) ≥ 1.24.2
- [SciPy](https://scipy.org) ≥ 1.10.1
- [Matplotlib](https://matplotlib.org) ≥ 3.6
- [Pandas](https://pandas.pydata.org) ≥ 2.1.0
- [scikit-learn](https://scikit-learn.org) ≥ 1.2.2
- [PySide6](https://www.qt.io/development/qt-framework/python-bindings) *(GUI only)*

## Documentation

Find detailed tutorials and examples in the [documentation](#).