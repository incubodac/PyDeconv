# PyDeconv — Examples
## What is PyDeconv?

PyDeconv is a Python package for EEG/MEG **deconvolution analysis** — the
technique of separating overlapping event-related brain responses from
continuous neural recordings.

Classical ERP approaches assume events are well-separated in time; in practice
they overlap.  PyDeconv handles this by building a **time-expanded design
matrix** that accounts for every event simultaneously and solves a single
regularised linear regression, yielding one temporal response function (TRF /
rERP) per predictor.

### Inputs

| Argument | Accepted formats |
|---|---|
| EEG data | `mne.io.Raw` object **or** NumPy array `(channels × samples)` |
| Events / features | `pandas.DataFrame` (requires a `latency` column in samples) **or** NumPy array |

### Model definition — fluent builder

Models are defined by chaining calls that register predictors.  Each predictor
is scoped to a particular **event type** (e.g. `'button'`, `'fixation'`,
`'audio'`) so that the response to overlapping events can be estimated
independently:

```python
from pydeconv.core import DeconvolutionModel

model = (
    DeconvolutionModel(tmin=-0.1, tmax=0.6, sfreq=256)

    # Intercept (mean TRF) for each event type
    .add_feature("fixation", from_event="fixation")
    .add_feature("audio",    from_event="audio")
    .add_feature("button",   from_event="button")

    # Additive covariate — reaction time (log-transformed)
    .add_feature("log_rt", column="reaction_time",
                 from_event="button", transform=np.log)

    # Interaction between two features for the same event type
    .add_interaction("log_rt", "condition", event_type="button")

    # Event-specific analysis window (narrower epoch around button press)
    .add_new_analysis_window("button", tmin=-0.05, tmax=0.3)
)
```

Continuous features can be expanded into **B-spline bases** by passing a
`spline_config` argument, allowing non-linear modelling of covariates.

---

## Examples in this directory
This directory contains runnable scripts that demonstrate the PyDeconv
workflow from end to end: simulation, real-data analysis,
and the interactive GUI.

---

## Scripts

### `01_simulation_roundtrip.py` ✅ working
Simulate continuous EEG from known event-related kernels and verify that the
pipeline recovers them.  Good starting point to understand the API.

Key steps:
1. Generate a synthetic experiment with `ExperimentDesign`.
2. Define ground-truth kernels with `CompoundKernel` (Gaussian bumps).
3. Simulate single-channel EEG with `EEGSimulator` and add pink noise.
4. Build a `DeconvolutionModel` using the **fluent builder API**:

   ```python
   model = (
       DeconvolutionModel(tmin=-0.1, tmax=0.5, sfreq=256, estimator=Tridge(alpha=1.0))
       .add_feature("stimulus", from_event="stimulus")   # event-specific intercept
       .add_feature("response", from_event="response")
       .add_new_analysis_window("response", tmin=-0.1, tmax=0.2)  # narrower window
   )
   ```

5. Build the design matrix, fit, and plot recovered kernels vs. ground truth.
6. Report R² goodness-of-fit.

---

### `02_real_data_pipeline.py` 🚧 stub
Full pipeline for real EEG data loaded from an EEGLAB `.set` file.

Planned steps (imports are commented out until modules stabilise):
- Load raw data with `load_set_file`.
- Pair with a behavioural events CSV (requires a `latency` column in samples).
- Inspect event statistics and detect artefact windows.
- Build the design matrix — additive features, interactions, and B-spline
  expansions can all be registered via the builder:

  ```python
  model = (
      DeconvolutionModel(tmin=-0.1, tmax=0.6, sfreq=sfreq)
      .add_feature("condition", from_event="target")
      .add_feature("rt", column="reaction_time", from_event="target")
      .add_interaction("condition", "rt", event_type="target")
  )
  ```

- Check collinearity (VIF), fit with cross-validated Ridge, evaluate, and plot.
- Save per-subject coefficients for group analysis.

---

### `03_group_analysis_tfce.py` 🚧 stub
Run TFCE permutation tests across subjects.

Planned steps:
- Stack per-subject coefficient arrays `(n_subjects × n_times × n_channels)`.
- Compute channel adjacency from the shared montage.
- Call `tfce(observations, adjacency, n_permutations=1024, alpha=0.05)`.
- Visualise significant spatio-temporal clusters as topographic maps.

---

### `04_gui_demo.py` ✅ working
Launch the PyDeconv graphical interface.

```python
from pydeconv.gui import launch
launch()
```

Or from the terminal:

```bash
python -m pydeconv.gui
# or
python examples/04_gui_demo.py
```

**Requires PySide6** (`pip install PySide6`).

The GUI is under active development.  Planned interactive controls will mirror
the builder API directly:

| GUI action | Programmatic equivalent |
|---|---|
| *Add feature* panel | `model.add_feature(name, column, from_event=...)` |
| *Add interaction* panel | `model.add_interaction(feature_a, feature_b, event_type=...)` |
| *New analysis window* panel | `model.add_new_analysis_window(event_type, tmin, tmax)` |

---

## Running the examples

```bash
# from the repo root
pip install -e .                    # install pydeconv in editable mode
python examples/01_simulation_roundtrip.py
python examples/04_gui_demo.py
```

## Dependencies

Python ≥ 3.9, NumPy, SciPy, Pandas, MNE, scikit-learn, Matplotlib.
PySide6 is required only for the GUI example.
