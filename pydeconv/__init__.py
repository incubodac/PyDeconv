# Public API exports
#
# Lazy imports — avoids triggering heavy transitive dependencies
# (e.g. mne via utils) at package-init time.


def __getattr__(name: str):
    """Lazy-load public API symbols on first access."""
    _exports = {
        "DeconvolutionModel": ".core",
        "Feature": ".core",
        "SplineConfig": ".core",
        "Tridge": ".estimators",
    }
    if name in _exports:
        import importlib
        module = importlib.import_module(_exports[name], __package__)
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["DeconvolutionModel", "Feature", "SplineConfig", "Tridge"]
