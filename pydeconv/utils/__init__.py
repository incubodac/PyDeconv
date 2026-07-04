# Utility subpackage exports
#
# Lazy imports — avoids pulling in heavy dependencies (mne, torch)
# at utils package init time.  Individual modules can always be
# imported directly: ``from pydeconv.utils.design_matrix import ...``


def __getattr__(name: str):
    """Lazy-load public utility symbols on first access."""
    _window_rejection = {
        "cont_ArtifactDetect",
        "basicrap",
        "joinclosesegments",
    }
    _tfce = {
        "tfce",
        "get_channel_adjacency",
    }
    _design_matrix = {
        "create_design_matrix",
    }

    if name in _window_rejection:
        from . import window_rejection
        return getattr(window_rejection, name)

    if name in _tfce:
        from . import tfce
        return getattr(tfce, name)

    if name in _design_matrix:
        from . import design_matrix
        return getattr(design_matrix, name)

    raise AttributeError(
        f"module {__name__!r} has no attribute {name!r}"
    )


__all__ = [
    # Rejection
    'cont_ArtifactDetect',
    'basicrap',
    'joinclosesegments',
    # Stats / TFCE
    'tfce',
    'get_channel_adjacency',
    # Design matrix
    'create_design_matrix',
]
