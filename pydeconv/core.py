# Core deconvolution pipeline
# Contains main class definitions (DeconvolutionModel, Feature, SplineConfig)

from __future__ import annotations

import dataclasses
from typing import Callable, NamedTuple

import numpy as np
import pandas as pd
from scipy.interpolate import BSpline
from sklearn.base import BaseEstimator, clone, is_regressor
from sklearn.linear_model import Ridge

# NOTE: shifted_matrix (from .utils.design_matrix) is imported lazily
# inside build_design_matrix() to avoid requiring torch at import time.


# ---------------------------------------------------------------------------
# Feature specification
# ---------------------------------------------------------------------------


class Feature(NamedTuple):
    """Specification for a single additive predictor.

    Parameters
    ----------
    name : str
        Human-readable label used in ``feature_names_`` and coefficient
        indexing (e.g. ``'log_contrast'``).
    column : str
        Column name in the events ``DataFrame`` from which raw values are
        read.
    transform : callable or None
        Optional function applied element-wise to the raw column values
        before they enter the design matrix.  Can be any callable that
        accepts and returns an array-like (e.g. ``np.log``,
        ``lambda x: x ** 2``).

    """

    name: str
    column: str
    transform: Callable[[np.ndarray], np.ndarray] | None = None


# ---------------------------------------------------------------------------
# Spline configuration
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class SplineConfig:
    """B-spline expansion settings for a single feature.

    Parameters
    ----------
    n_splines : int
        Number of spline basis functions (interior knots = ``n_splines - 2``
        for a cubic B-spline with degree 3).
    knot_method : str
        Strategy for placing interior knots: ``'quantile'`` (default) places
        knots at evenly-spaced quantiles of the observed data;
        ``'uniform'`` places knots at evenly-spaced values across the data
        range.
    knots : numpy.ndarray or None
        Explicit interior knot vector.  When provided, ``knot_method`` is
        ignored.
    degree : int
        Polynomial degree of the B-spline.  Default is 3 (cubic).

    """

    n_splines: int = 5
    knot_method: str = "quantile"
    knots: np.ndarray | None = None
    degree: int = 3


def _parse_spline_config(
    spec: bool | int | dict | SplineConfig | None,
) -> dict[str, SplineConfig] | None:
    """Normalise user-facing spline specification into a per-feature dict.

    Accepted formats
    ----------------
    * ``None`` or ``False`` → no splines.
    * ``True`` or ``int`` → global ``SplineConfig`` applied to **every**
      continuous additive feature.
    * ``SplineConfig`` instance → same as above.
    * ``dict`` mapping feature **names** to any of:
      - ``int`` → ``SplineConfig(n_splines=int)``
      - ``[int, str]`` → ``SplineConfig(n_splines=int, knot_method=str)``
      - ``SplineConfig`` instance

    Returns
    -------
    dict[str, SplineConfig] or None
        ``None`` when splines are disabled; otherwise a dict keyed by
        feature name.

    """
    if spec is None or spec is False:
        return None

    if spec is True:
        return {"__global__": SplineConfig()}

    if isinstance(spec, int):
        return {"__global__": SplineConfig(n_splines=spec)}

    if isinstance(spec, SplineConfig):
        return {"__global__": spec}

    if isinstance(spec, dict):
        parsed: dict[str, SplineConfig] = {}
        for key, value in spec.items():
            if isinstance(value, int):
                parsed[key] = SplineConfig(n_splines=value)
            elif isinstance(value, (list, tuple)):
                n = value[0]
                method = value[1] if len(value) > 1 else "quantile"
                parsed[key] = SplineConfig(n_splines=n, knot_method=method)
            elif isinstance(value, SplineConfig):
                parsed[key] = value
            else:
                raise TypeError(
                    f"Unsupported spline spec for '{key}': {type(value)}"
                )
        return parsed

    raise TypeError(f"Unsupported spline_config type: {type(spec)}")


# ---------------------------------------------------------------------------
# B-spline basis expansion
# ---------------------------------------------------------------------------


def _compute_knots(
    values: np.ndarray,
    config: SplineConfig,
) -> np.ndarray:
    """Compute the full knot vector (including boundary knots).

    Parameters
    ----------
    values : numpy.ndarray
        Observed feature values used for knot placement.
    config : SplineConfig
        Spline configuration.

    Returns
    -------
    knots : numpy.ndarray
        Full knot vector of length ``n_interior + 2 * (degree + 1)``.

    """
    degree = config.degree
    if config.knots is not None:
        interior = np.sort(config.knots)
    else:
        n_interior = max(config.n_splines - (degree + 1), 1)
        if config.knot_method == "quantile":
            quantiles = np.linspace(0, 1, n_interior + 2)[1:-1]
            interior = np.quantile(values, quantiles)
        elif config.knot_method == "uniform":
            interior = np.linspace(
                np.min(values), np.max(values), n_interior + 2
            )[1:-1]
        else:
            raise ValueError(
                f"Unknown knot_method '{config.knot_method}'. "
                "Use 'quantile' or 'uniform'."
            )

    lo, hi = np.min(values), np.max(values)
    knots = np.concatenate([
        np.repeat(lo, degree + 1),
        interior,
        np.repeat(hi, degree + 1),
    ])
    return knots


def _bspline_basis(
    values: np.ndarray,
    config: SplineConfig,
) -> np.ndarray:
    """Expand a 1-D feature vector into a B-spline basis matrix.

    Parameters
    ----------
    values : numpy.ndarray, shape ``(n_events,)``
        Raw (or transformed) feature values.
    config : SplineConfig
        Spline settings.

    Returns
    -------
    basis : numpy.ndarray, shape ``(n_events, n_splines)``
        Each column is one B-spline basis function evaluated at *values*.

    """
    knots = _compute_knots(values, config)
    degree = config.degree
    n_basis = len(knots) - degree - 1

    basis = np.column_stack([
        BSpline.basis_element(
            knots[i: i + degree + 2], extrapolate=False
        )(values)
        for i in range(n_basis)
    ])
    # Replace NaN from extrapolation with 0
    np.nan_to_num(basis, nan=0.0, copy=False)
    return basis


# ---------------------------------------------------------------------------
# DeconvolutionModel
# ---------------------------------------------------------------------------

class DeconvolutionModel(BaseEstimator):
    """Linear deconvolution model for continuous EEG / MEG data.

    Builds a time-shifted design matrix from event features (with optional
    transformations, interactions, and B-spline expansions) and fits a
    regularised linear regression to recover temporal response functions.

    Parameters
    ----------
    tmin : float
        Start of the kernel window in seconds relative to event onset.
    tmax : float
        End of the kernel window in seconds.
    Event-specific windows can be registered with
    ``add_new_analysis_window(event_type=..., tmin=..., tmax=...)``.
    These are stored in the model state and overwrite prior windows for
    the same event type, while global ``tmin``/``tmax`` remain defaults
    for other events.
    sfreq : float
        Sampling frequency of the continuous data in Hz.
    Intercepts are event-specific and are added by registering a feature
    where ``name == from_event`` (or ``event_type`` / ``type`` alias).
    additive_features : list of Feature or None
        Main-effect predictors.  Each ``Feature`` specifies a column name,
        a human-readable label, and an optional transform.
    interactions : list of tuple[str, str] or None
        Pairs of feature **names** (matching ``Feature.name``) whose
        element-wise product is added as an interaction column.
    spline_config : None, bool, int, dict, or SplineConfig
        Controls B-spline expansion of continuous features.

        * ``None`` / ``False`` — no spline expansion.
        * ``True`` — expand every additive feature with default
          ``SplineConfig(n_splines=5, knot_method='quantile')``.
        * ``int`` — like ``True`` but with the given number of splines.
        * ``SplineConfig`` — global config applied to every feature.
        * ``dict`` — per-feature control, e.g.
    estimator : sklearn regressor or None
        Scikit-learn regressor instance.  Defaults to ``Ridge()``.
    scoring : str
        Scoring metric name (currently ``'r2'``).

    """

    def __init__(
        self,
        tmin: float = -0.2,
        tmax: float = 0.6,
        sfreq: float = 256.0,
        event_column: str = "type",
        additive_features: list[Feature] | dict[str, list[Feature]] | None = None,
        interactions: list[tuple[str, str]] | dict[str, list[tuple[str, str]]] | None = None,
        spline_config: (
            None | bool | int | dict[str, int | list | SplineConfig]
            | SplineConfig
        ) = None,
        estimator: BaseEstimator | None = None,
        scoring: str = "r2",
    ):
        self.tmin = tmin
        self.tmax = tmax
        self.sfreq = sfreq
        self.event_column = event_column

        if additive_features is None:
            self.additive_features = {}
        elif isinstance(additive_features, dict):
            self.additive_features = {k: list(v) for k, v in additive_features.items()}
        else:
            features_list = list(additive_features)
            self.additive_features = {"__global__": features_list} if features_list else {}

        if interactions is None:
            self.interactions = {}
        elif isinstance(interactions, dict):
            self.interactions = {k: list(v) for k, v in interactions.items()}
        else:
            interactions_list = list(interactions)
            self.interactions = {"__global__": interactions_list} if interactions_list else {}

        self.spline_config = spline_config
        self.estimator = estimator if estimator is not None else Ridge()
        self.scoring = scoring
        self.global_window: tuple[float, float] = (self.tmin, self.tmax)
        self.analysis_windows: dict[str, tuple[float, float]] = {}

        # ----- derived / fitted state -----
        self._spline_map: dict[str, SplineConfig] | None = (
            _parse_spline_config(spline_config)
        )
        self.feature_names_: list[str] = []
        self.coef_: np.ndarray | None = None
        self.is_fitted: bool = False
        self._feature_mean_: np.ndarray | None = None
        self._feature_std_: np.ndarray | None = None
        self.event_intercepts: set[str] = set()
        self._spline_features: dict[str, list[tuple[str, str, SplineConfig, bool]]] = {}
        self._last_event_scope: str | None = None

    # ----- builder helpers -----

    def add_feature(
        self,
        name: str,
        column: str | None = None,
        event_type: str | None = None,
        from_event: str | None = None,
        type: str | None = None,
        transform: Callable[[np.ndarray], np.ndarray] | None = None,
    ) -> "DeconvolutionModel":
        """Append an additive feature to the model.

        Parameters
        ----------
        name : str
            Human-readable label for this predictor.
        column : str or None
            Column name in the events DataFrame. If None, defaults to
            ``self.event_column``.
        event_type : str or None
            The specific value in ``events[self.event_column]`` this feature
            applies to. If None, applies globally to all events.
        from_event : str or None
            Alias for ``event_type`` with clearer semantics.
        type : str or None
            Alias for ``event_type`` for a compact builder style.
            Cannot be used together with ``event_type`` or ``from_event``.
        transform : callable or None
            Element-wise transformation applied before design-matrix
            construction.

        Returns
        -------
        self : DeconvolutionModel
            For method chaining.

        Notes
        -----
        If ``event_type`` (or one of its aliases) is provided and
        ``name == event_type``, this call is treated as a request for an
        event-specific intercept. In that case, the event type is recorded
        in ``event_intercepts`` and no additive feature column is added.

        """
        provided = [
            event_type is not None,
            from_event is not None,
            type is not None,
        ]
        if sum(provided) > 1:
            raise ValueError(
                "Use only one of 'event_type', 'from_event', or 'type'."
            )

        if event_type is None:
            event_type = from_event if from_event is not None else type

        if event_type is not None:
            self._last_event_scope = event_type

        if column is None:
            column = self.event_column

        if event_type is not None and name == event_type:
            self.event_intercepts.add(event_type)
            return self

        key = event_type if event_type is not None else "__global__"
        if key not in self.additive_features:
            self.additive_features[key] = []
        self.additive_features[key].append(Feature(name, column, transform))
        return self

    def add_feature_splines(
        self,
        name: str,
        column: str | None = None,
        from_event: str | None = None,
        n_splines: int = 5,
        knot_method: str = "quantile",
        degree: int = 3,
        intercept: bool = True,
    ) -> "DeconvolutionModel":
        """Register a B-spline expansion of a continuous covariate.

        This method expands a single numeric column into ``n_splines``
        basis functions, producing design-matrix columns named
        ``"{name}_sp_0"``, ``"{name}_sp_1"``, etc.

        Parameters
        ----------
        name : str
            Label prefix for the spline basis columns.
        column : str or None
            Column in the events DataFrame to read values from.
            Defaults to *name* when ``None``.
        from_event : str or None
            Restrict this spline feature to rows whose event column
            matches *from_event*.
        n_splines : int
            Number of B-spline basis functions.
        knot_method : str
            ``'quantile'`` (default) or ``'equidistant'`` / ``'uniform'``.
        degree : int
            Polynomial degree of the B-spline (default 3 = cubic).
        intercept : bool
            If ``False``, the first (constant-like) basis function is
            dropped to avoid collinearity with an event intercept.

        Returns
        -------
        self : DeconvolutionModel
            For method chaining.

        """
        if column is None:
            column = name

        # Normalise alias
        if knot_method == "equidistant":
            knot_method = "uniform"

        cfg = SplineConfig(
            n_splines=n_splines,
            knot_method=knot_method,
            degree=degree,
        )

        event_type = from_event
        if event_type is not None:
            self._last_event_scope = event_type

        key = event_type if event_type is not None else "__global__"
        if key not in self._spline_features:
            self._spline_features[key] = []
        self._spline_features[key].append((name, column, cfg, intercept))
        return self

    def add_interaction(
        self,
        feature_a: str,
        feature_b: str,
        event_type: str | None = None,
    ) -> "DeconvolutionModel":
        """Register an interaction term between two features.

        Parameters
        ----------
        feature_a : str
            Name of the first feature (must match a ``Feature.name``).
        feature_b : str
            Name of the second feature.
        event_type : str or None
            The specific event type this interaction applies to.

        Returns
        -------
        self : DeconvolutionModel
            For method chaining.

        """
        key = event_type if event_type is not None else "__global__"
        if key not in self.interactions:
            self.interactions[key] = []
        self.interactions[key].append((feature_a, feature_b))
        if event_type is not None:
            self._last_event_scope = event_type
        return self

    def add_new_analysis_window(
        self,
        event_type: str | None = None,
        tmin: float | None = None,
        tmax: float | None = None,
    ) -> "DeconvolutionModel":
        """Register or overwrite an event-specific analysis window.

        Parameters
        ----------
        tmin : float
            Window start in seconds relative to event onset.
        tmax : float
            Window end in seconds relative to event onset.
        event_type : str or None
            Value from ``events[self.event_column]`` to which this window
            applies. If None, reuse the last explicit event scope added via
            ``add_feature(..., from_event=...)`` / ``add_interaction(..., event_type=...)``.

        Returns
        -------
        self : DeconvolutionModel
            For method chaining.

        Notes
        -----
        Event-specific windows are applied during ``build_design_matrix`` by
        masking delays outside the configured interval for columns associated
        with that event type.

        """
        if tmin is None or tmax is None:
            raise ValueError("Both tmin and tmax must be provided.")

        if event_type is None:
            event_type = self._last_event_scope

        if not isinstance(event_type, str) or not event_type:
            raise ValueError("event_type must be a non-empty string.")
        if tmin > tmax:
            raise ValueError(f"tmin ({tmin}) must be <= tmax ({tmax}).")

        self.analysis_windows[event_type] = (float(tmin), float(tmax))
        return self

    def _window_delay_mask(self, event_type: str) -> np.ndarray:
        """Return a boolean mask selecting valid delays for an event type."""
        win = self.analysis_windows.get(event_type)
        if win is None:
            return np.ones(len(self.delays_), dtype=bool)

        win_min = int(np.round(win[0] * self.sfreq))
        win_max = int(np.round(win[1] * self.sfreq))
        return (self.delays_ >= win_min) & (self.delays_ <= win_max)

    # ----- design matrix -----

    def _resolve_spline_config(self, feature_name: str) -> SplineConfig | None:
        """Return the SplineConfig for a feature, or None."""
        if self._spline_map is None:
            return None
        if feature_name in self._spline_map:
            return self._spline_map[feature_name]
        if "__global__" in self._spline_map:
            return self._spline_map["__global__"]
        return None

    def build_design_matrix(
        self,
        events: pd.DataFrame,
        n_samples: int,
        use_gpu: bool = True,
    ) -> np.ndarray:
        """Construct the full time-shifted design matrix.

        The method operates in clearly separated passes:

        1. **Feature extraction pass** — identifies rows belonging to each
           configured event type, extracts feature values, and applies
           transforms. Missing event types are padded with zeros.
        2. **Column-building pass**:
              a. Creates event-specific intercept columns when registered via
                  ``add_feature(name=..., from_event=...)`` with ``name == from_event``.
           b. Expands additive features into B-spline bases where configured.
           c. Computes interaction columns from pre-transformed values.
        3. Places feature values at event latencies to build a
           ``(n_samples, n_features)`` stick-function matrix.
        4. Time-shifts the stick matrix via ``shifted_matrix``.

        Parameters
        ----------
        events : pandas.DataFrame
            Must contain a ``'latency'`` column (in samples) and the configured
            ``event_column`` if event-specific features are used.
        n_samples : int
            Length of the continuous recording in samples.
        use_gpu : bool
            Passed through to ``shifted_matrix``.

        Returns
        -------
        X : np.ndarray, shape ``(n_samples, n_columns * n_delays)``
            The complete design matrix.

        """
        if "latency" not in events.columns:
            raise ValueError("events DataFrame must contain a 'latency' column")

        self.delays_ = np.arange(
            int(np.round(self.tmin * self.sfreq)),
            int(np.round(self.tmax * self.sfreq)) + 1,
        )
        self.times_ = self.delays_ / self.sfreq

        n_events = len(events)
        columns: list[np.ndarray] = []
        names: list[str] = []
        column_event_types: list[str] = []

        all_event_types = (
            set(self.additive_features.keys())
            | set(self.interactions.keys())
            | set(self.event_intercepts)
            | set(self._spline_features.keys())
        )

        for ev_type in sorted(all_event_types):
            if ev_type == "__global__":
                mask = np.ones(n_events, dtype=bool)
                prefix = ""
            else:
                if self.event_column not in events.columns:
                    raise ValueError(f"Event column '{self.event_column}' not found in events DataFrame")
                mask = (events[self.event_column] == ev_type).values
                prefix = f"{ev_type}:"

            n_sub = mask.sum()
            if n_sub == 0:
                continue

            sub_events = events.iloc[mask]
            feature_values: dict[str, np.ndarray] = {}

            # --- Intercept ---
            add_intercept = ev_type in self.event_intercepts
            if add_intercept:
                col = np.zeros(n_events)
                col[mask] = 1.0
                columns.append(col)
                names.append(f"{prefix}intercept")
                column_event_types.append(ev_type)

            # --- Additive Features ---
            feats = self.additive_features.get(ev_type, [])
            for feat in feats:
                raw = sub_events[feat.column].values
                vals = feat.transform(raw) if feat.transform is not None else raw
                vals = vals.astype(float)
                feature_values[feat.name] = vals

                spline_cfg = self._resolve_spline_config(feat.name)
                if spline_cfg is not None:
                    basis = _bspline_basis(vals, spline_cfg)
                    start_col = 1 if add_intercept else 0
                    for i in range(start_col, basis.shape[1]):
                        col = np.zeros(n_events)
                        col[mask] = basis[:, i]
                        columns.append(col)
                        names.append(f"{prefix}{feat.name}_spl{i}")
                        column_event_types.append(ev_type)
                else:
                    col = np.zeros(n_events)
                    col[mask] = vals
                    columns.append(col)
                    names.append(f"{prefix}{feat.name}")
                    column_event_types.append(ev_type)

            # --- Spline Features (from add_feature_splines) ---
            spline_feats = self._spline_features.get(ev_type, [])
            for sp_name, sp_col, sp_cfg, sp_intercept in spline_feats:
                if sp_col not in sub_events.columns:
                    raise ValueError(
                        f"Spline feature '{sp_name}' references column "
                        f"'{sp_col}' not found in events DataFrame."
                    )
                sp_vals = sub_events[sp_col].values.astype(float)
                basis = _bspline_basis(sp_vals, sp_cfg)
                start_idx = 0 if sp_intercept else 1
                for i in range(start_idx, basis.shape[1]):
                    col = np.zeros(n_events)
                    col[mask] = basis[:, i]
                    columns.append(col)
                    names.append(f"{prefix}{sp_name}_sp_{i}")
                    column_event_types.append(ev_type)

            # --- Interactions ---
            inters = self.interactions.get(ev_type, [])
            for feat_a, feat_b in inters:
                if feat_a not in feature_values or feat_b not in feature_values:
                    raise ValueError(
                        f"Interaction '{feat_a}:{feat_b}' references a feature "
                        f"not present in the additive_features for event type "
                        f"'{ev_type}'."
                    )
                interaction_vals = feature_values[feat_a] * feature_values[feat_b]
                col = np.zeros(n_events)
                col[mask] = interaction_vals
                columns.append(col)
                names.append(f"{prefix}{feat_a}:{feat_b}")
                column_event_types.append(ev_type)

        if not columns:
            raise ValueError("No features were extracted to build the design matrix.")

        self.feature_names_ = names

        # ----- 2. Place into a (n_samples, n_columns) stick matrix -----
        n_columns = len(columns)
        feature_matrix = np.zeros((n_samples, n_columns), dtype=np.float64)

        latencies = events["latency"].values.astype(int)
        valid = (latencies >= 0) & (latencies < n_samples)

        for col_idx in range(n_columns):
            feature_matrix[latencies[valid], col_idx] = columns[col_idx][valid]

        # ----- 3. Time-shift -----
        from .utils.design_matrix import shifted_matrix

        X = shifted_matrix(
            feature_matrix,
            delays=self.delays_.tolist(),
            use_gpu=use_gpu,
        )

        # Apply event-specific analysis windows by masking delay bins per
        # feature block while preserving the global matrix shape.
        n_delays = len(self.delays_)
        for col_idx, ev_type in enumerate(column_event_types):
            if ev_type == "__global__":
                continue
            keep_mask = self._window_delay_mask(ev_type)
            if np.all(keep_mask):
                continue
            start = col_idx * n_delays
            end = start + n_delays
            X[:, start:end] *= keep_mask[np.newaxis, :]

        return X

    # ----- standardisation (private, called from fit) -----

    def _standardize(self, X: np.ndarray) -> np.ndarray:
        """Z-score each column of the design matrix.

        Parameters
        ----------
        X : numpy.ndarray, shape ``(n_rows, n_cols)``
            Design matrix (typically after removing zero-only rows).

        Returns
        -------
        X_std : numpy.ndarray
            Standardised copy of *X*.

        """
        self._feature_mean_ = X.mean(axis=0)
        self._feature_std_ = X.std(axis=0)
        # Avoid division by zero for constant columns (e.g. intercept)
        self._feature_std_[self._feature_std_ == 0] = 1.0
        return (X - self._feature_mean_) / self._feature_std_

    # ----- fit / predict / score -----

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        standardize: bool = True,
    ) -> "DeconvolutionModel":
        """Fit the deconvolution model.

        Parameters
        ----------
        X : numpy.ndarray, shape ``(n_rows, n_features * n_delays)``
            Design matrix (output of ``build_design_matrix``).
        y : numpy.ndarray, shape ``(n_rows, n_channels)`` or ``(n_rows,)``
            Continuous neural data aligned to the design matrix rows.
        standardize : bool
            If ``True``, z-score *X* before fitting and store the
            normalisation parameters for later use.

        Returns
        -------
        self : DeconvolutionModel

        """
        if not is_regressor(self.estimator):
            raise TypeError(
                f"estimator must be a scikit-learn regressor, "
                f"got {type(self.estimator)}"
            )

        est = clone(self.estimator)

        if standardize:
            X = self._standardize(X)

        est.fit(X, y)

        # Extract coefficients — handle 1-D edge case
        coef = est.coef_
        if coef.ndim == 1:
            coef = coef.reshape(1, -1)
        self.coef_ = coef
        self.estimator_ = est
        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Generate predictions from the fitted model.

        Parameters
        ----------
        X : numpy.ndarray, shape ``(n_rows, n_features * n_delays)``
            Design matrix.

        Returns
        -------
        y_pred : numpy.ndarray

        """
        if not self.is_fitted:
            raise RuntimeError("Model has not been fitted yet.")

        # Apply the same standardisation used during fit
        if self._feature_mean_ is not None:
            X = (X - self._feature_mean_) / self._feature_std_

        return self.estimator_.predict(X)

    def score(
        self, X: np.ndarray, y: np.ndarray
    ) -> np.ndarray | float:
        """Score predictions against observed data.

        Parameters
        ----------
        X : numpy.ndarray
            Design matrix.
        y : numpy.ndarray
            Observed neural data.

        Returns
        -------
        scores : float or numpy.ndarray
            R² score(s).  A single float when *y* is 1-D, otherwise an
            array of per-channel scores.

        """
        from sklearn.metrics import r2_score

        y_pred = self.predict(X)
        if y.ndim == 1:
            return r2_score(y, y_pred)
        return r2_score(y, y_pred, multioutput="raw_values")

    # ----- repr -----

    def __repr__(self) -> str:
        """Compact one-line summary."""
        parts = [
            f"tmin={self.tmin:.3f}",
            f"tmax={self.tmax:.3f}",
            f"sfreq={self.sfreq}",
            f"event_intercepts={len(self.event_intercepts)}",
            f"event_windows={len(self.analysis_windows)}",
            f"n_features={len(self.additive_features)}",
            f"n_interactions={len(self.interactions)}",
            f"splines={'on' if self._spline_map else 'off'}",
            f"estimator={type(self.estimator).__name__}",
            f"fitted={self.is_fitted}",
        ]
        return f"<DeconvolutionModel | {', '.join(parts)}>"
