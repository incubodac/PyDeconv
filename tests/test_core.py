# Tests for core deconvolution module

import numpy as np
import pandas as pd
import pytest

# Check optional dependency availability
try:
    import torch # noqa: F401
    _has_torch = True
except ImportError:
    _has_torch = False

from pydeconv.core import (
    DeconvolutionModel,
    Feature,
    SplineConfig,
    _bspline_basis,
    _compute_knots,
    _parse_spline_config,
)


# ---------------------------------------------------------------------------
# Feature
# ---------------------------------------------------------------------------


class TestFeature:
    """Tests for the Feature NamedTuple."""

    def test_basic_creation(self):
        f = Feature(name="contrast", column="contrast_col")
        assert f.name == "contrast"
        assert f.column == "contrast_col"
        assert f.transform is None

    def test_with_transform(self):
        f = Feature(name="log_contrast", column="contrast", transform=np.log1p)
        vals = np.array([0.0, 1.0, 9.0])
        result = f.transform(vals)
        np.testing.assert_allclose(result, np.log1p(vals))

    def test_lambda_transform(self):
        f = Feature(name="sq", column="x", transform=lambda x: x ** 2)
        np.testing.assert_array_equal(f.transform(np.array([2, 3])), [4, 9])


# ---------------------------------------------------------------------------
# SplineConfig & parsing
# ---------------------------------------------------------------------------


class TestSplineConfig:
    """Tests for SplineConfig and the flexible parsing helper."""

    def test_default_values(self):
        cfg = SplineConfig()
        assert cfg.n_splines == 5
        assert cfg.knot_method == "quantile"
        assert cfg.knots is None
        assert cfg.degree == 3

    def test_parse_none(self):
        assert _parse_spline_config(None) is None
        assert _parse_spline_config(False) is None

    def test_parse_true(self):
        result = _parse_spline_config(True)
        assert "__global__" in result
        assert result["__global__"].n_splines == 5

    def test_parse_int(self):
        result = _parse_spline_config(7)
        assert result["__global__"].n_splines == 7

    def test_parse_splineconfig_instance(self):
        cfg = SplineConfig(n_splines=4, knot_method="uniform")
        result = _parse_spline_config(cfg)
        assert result["__global__"] is cfg

    def test_parse_dict_int(self):
        result = _parse_spline_config({"feat_a": 6})
        assert result["feat_a"].n_splines == 6
        assert result["feat_a"].knot_method == "quantile"

    def test_parse_dict_list(self):
        result = _parse_spline_config({"feat_a": [4, "uniform"]})
        assert result["feat_a"].n_splines == 4
        assert result["feat_a"].knot_method == "uniform"

    def test_parse_dict_mixed(self):
        spec = {
            "feat_a": [4, "quantile"],
            "feat_b": 6,
            "feat_c": SplineConfig(n_splines=3),
        }
        result = _parse_spline_config(spec)
        assert result["feat_a"].n_splines == 4
        assert result["feat_b"].n_splines == 6
        assert result["feat_c"].n_splines == 3

    def test_parse_invalid_type_raises(self):
        with pytest.raises(TypeError):
            _parse_spline_config("invalid")


# ---------------------------------------------------------------------------
# B-spline basis expansion
# ---------------------------------------------------------------------------


class TestBSplineBasis:
    """Tests for knot computation and basis expansion."""

    def test_compute_knots_quantile(self):
        vals = np.arange(100, dtype=float)
        cfg = SplineConfig(n_splines=5, knot_method="quantile")
        knots = _compute_knots(vals, cfg)
        # Should start with repeated min and end with repeated max
        assert knots[0] == 0.0
        assert knots[-1] == 99.0

    def test_compute_knots_uniform(self):
        vals = np.arange(100, dtype=float)
        cfg = SplineConfig(n_splines=5, knot_method="uniform")
        knots = _compute_knots(vals, cfg)
        assert knots[0] == 0.0
        assert knots[-1] == 99.0

    def test_basis_shape(self):
        vals = np.linspace(0, 1, 50)
        cfg = SplineConfig(n_splines=5)
        basis = _bspline_basis(vals, cfg)
        assert basis.shape[0] == 50
        assert basis.shape[1] > 0  # at least one basis column

    def test_basis_no_nan(self):
        vals = np.linspace(0, 10, 100)
        cfg = SplineConfig(n_splines=6)
        basis = _bspline_basis(vals, cfg)
        assert np.all(np.isfinite(basis))


# ---------------------------------------------------------------------------
# DeconvolutionModel — construction
# ---------------------------------------------------------------------------


class TestDeconvolutionModelInit:
    """Tests for DeconvolutionModel instantiation and builder methods."""

    def test_defaults(self):
        model = DeconvolutionModel()
        assert model.additive_features == {}
        assert model.interactions == {}
        assert model.is_fitted is False
        assert model.coef_ is None
        assert "Ridge" in type(model.estimator).__name__

    def test_add_feature_chaining(self):
        model = (
            DeconvolutionModel()
            .add_feature("contrast", "contrast_col")
            .add_feature("log_rt", "rt", transform=np.log)
        )
        assert len(model.additive_features) == 2
        assert model.additive_features[0].name == "contrast"
        assert model.additive_features[1].transform is np.log

    def test_add_interaction(self):
        model = (
            DeconvolutionModel()
            .add_feature("a", "col_a")
            .add_feature("b", "col_b")
            .add_interaction("a", "b")
        )
        assert model.interactions == [("a", "b")]

    def test_repr_unfitted(self):
        model = DeconvolutionModel(tmin=-0.1, tmax=0.5, sfreq=500)
        r = repr(model)
        assert "DeconvolutionModel" in r
        assert "fitted=False" in r
        assert "sfreq=500" in r

    def test_spline_config_per_feature(self):
        model = DeconvolutionModel(
            spline_config={"contrast": [4, "quantile"], "saccade_amp": 6},
        )
        assert model._spline_map is not None
        assert model._spline_map["contrast"].n_splines == 4
        assert model._spline_map["saccade_amp"].n_splines == 6


# ---------------------------------------------------------------------------
# DeconvolutionModel — design matrix
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not _has_torch,
    reason="torch is required for shifted_matrix / build_design_matrix",
)
class TestBuildDesignMatrix:
    """Tests for build_design_matrix with a simple synthetic setup."""

    @pytest.fixture()
    def simple_events(self):
        return pd.DataFrame({
            "latency": [10, 50, 100],
            "type": ["stim", "stim", "stim"],
            "contrast": [0.5, 1.0, 0.8],
            "mss": [2, 4, 6],
        })

    def test_shape_with_features(self, simple_events):
        model = (
            DeconvolutionModel(tmin=0.0, tmax=0.1, sfreq=100)
            .add_feature("contrast", "contrast")
            .add_feature("mss", "mss")
        )
        X = model.build_design_matrix(simple_events, n_samples=200, use_gpu=False)
        n_delays = 11
        # intercept + contrast + mss = 3 columns
        assert X.shape == (200, 3 * n_delays)

    def test_shape_with_interaction(self, simple_events):
        model = (
            DeconvolutionModel(tmin=0.0, tmax=0.1, sfreq=100)
            .add_feature("contrast", "contrast")
            .add_feature("mss", "mss")
            .add_interaction("contrast", "mss")
        )
        X = model.build_design_matrix(simple_events, n_samples=200, use_gpu=False)
        n_delays = 11
        # intercept + contrast + mss + contrast:mss = 4
        assert X.shape == (200, 4 * n_delays)

    def test_feature_names_populated(self, simple_events):
        model = (
            DeconvolutionModel(tmin=0.0, tmax=0.1, sfreq=100)
            .add_feature("contrast", "contrast")
        )
        model.build_design_matrix(simple_events, n_samples=200, use_gpu=False)
        assert model.feature_names_ == ["intercept", "contrast"]

    def test_transform_applied(self, simple_events):
        model = (
            DeconvolutionModel(
                tmin=0.0, tmax=0.0, sfreq=100, has_intercept=False
            )
            .add_feature("sq_contrast", "contrast", transform=lambda x: x ** 2)
        )
        X = model.build_design_matrix(simple_events, n_samples=200, use_gpu=False)
        # At latency 10, the value should be 0.5**2 = 0.25
        assert np.isclose(X[10, 0], 0.25)

    def test_missing_latency_raises(self):
        events = pd.DataFrame({"type": ["a"]})
        model = DeconvolutionModel()
        with pytest.raises(ValueError, match="latency"):
            model.build_design_matrix(events, n_samples=100, use_gpu=False)

    def test_interaction_missing_feature_raises(self, simple_events):
        model = (
            DeconvolutionModel(tmin=0.0, tmax=0.0, sfreq=100)
            .add_feature("contrast", "contrast")
            .add_interaction("contrast", "nonexistent")
        )
        with pytest.raises(ValueError, match="Interaction"):
            model.build_design_matrix(simple_events, n_samples=200, use_gpu=False)

    def test_spline_expansion_adds_columns(self, simple_events):
        model = (
            DeconvolutionModel(
                tmin=0.0, tmax=0.0, sfreq=100,
                has_intercept=False,
                spline_config={"contrast": 5},
            )
            .add_feature("contrast", "contrast")
        )
        X = model.build_design_matrix(simple_events, n_samples=200, use_gpu=False)
        # With splines, contrast expands into multiple basis columns
        assert X.shape[1] > 1
        # Feature names should include spline suffixes
        assert any("spl" in n for n in model.feature_names_)


# ---------------------------------------------------------------------------
# DeconvolutionModel — fit / predict / score
# ---------------------------------------------------------------------------


class TestFitPredictScore:
    """Smoke tests for the fit → predict → score pipeline."""

    def test_fit_and_predict(self):
        rng = np.random.default_rng(42)
        n, p = 100, 5
        X = rng.standard_normal((n, p))
        w = rng.standard_normal((1, p))
        y = X @ w.T + rng.standard_normal((n, 1)) * 0.1

        model = DeconvolutionModel()
        model.feature_names_ = [f"f{i}" for i in range(p)]
        model.fit(X, y, standardize=True)

        assert model.is_fitted
        assert model.coef_ is not None
        assert model.coef_.shape[1] == p

        y_pred = model.predict(X)
        # sklearn Ridge.predict squeezes (n, 1) → (n,)
        assert y_pred.shape[0] == y.shape[0]

    def test_score_r2(self):
        rng = np.random.default_rng(0)
        n, p = 200, 3
        X = rng.standard_normal((n, p))
        y = X @ np.array([1.0, -2.0, 0.5]) + rng.standard_normal(n) * 0.01

        model = DeconvolutionModel()
        model.fit(X, y, standardize=False)
        r2 = model.score(X, y)
        assert r2 > 0.99

    def test_fit_without_standardize(self):
        rng = np.random.default_rng(7)
        X = rng.standard_normal((50, 3))
        y = rng.standard_normal(50)

        model = DeconvolutionModel()
        model.fit(X, y, standardize=False)
        assert model._feature_mean_ is None
        assert model.is_fitted

    def test_predict_before_fit_raises(self):
        model = DeconvolutionModel()
        with pytest.raises(RuntimeError, match="not been fitted"):
            model.predict(np.zeros((10, 3)))


@pytest.mark.skipif(
    not _has_torch,
    reason="torch is required for shifted_matrix / build_design_matrix",
)
class TestEventSpecificAnalysisWindows:
    """Tests for per-event delay windows in the shifted design matrix."""

    @pytest.fixture()
    def events_two_types(self):
        return pd.DataFrame({
            "latency": [10, 20, 40, 70],
            "type": ["stimulus", "stimulus", "response", "response"],
        })

    def test_window_can_infer_last_scoped_event(self):
        model = (
            DeconvolutionModel(tmin=-0.1, tmax=0.2, sfreq=100)
            .add_feature("stimulus", from_event="stimulus")
            .add_new_analysis_window(tmin=0.0, tmax=0.0)
        )
        assert model.analysis_windows["stimulus"] == (0.0, 0.0)

    def test_event_specific_window_masks_delays(self, events_two_types):
        model = (
            DeconvolutionModel(tmin=-0.1, tmax=0.1, sfreq=10)
            .add_feature("stimulus", from_event="stimulus")
            .add_new_analysis_window(tmin=0.0, tmax=0.0)
            .add_feature("response", from_event="response")
        )

        X = model.build_design_matrix(events_two_types, n_samples=120, use_gpu=False)

        n_delays = len(model.delays_)
        stim_idx = model.feature_names_.index("stimulus:intercept")
        resp_idx = model.feature_names_.index("response:intercept")

        stim_block = X[:, stim_idx * n_delays: (stim_idx + 1) * n_delays]
        resp_block = X[:, resp_idx * n_delays: (resp_idx + 1) * n_delays]

        zero_delay_idx = int(np.where(model.delays_ == 0)[0][0])
        non_zero_delay_idx = [i for i in range(n_delays) if i != zero_delay_idx]

        # Stimulus window is [0, 0], so all non-zero delay columns are masked.
        assert np.allclose(stim_block[:, non_zero_delay_idx], 0.0)
        assert np.any(np.abs(stim_block[:, zero_delay_idx]) > 0)

        # Response keeps the global window; therefore non-zero delays remain.
        assert np.any(np.abs(resp_block[:, non_zero_delay_idx]) > 0)
