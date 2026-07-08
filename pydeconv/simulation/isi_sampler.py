from typing import Callable

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

ISISampler = Callable[[pd.Series], int]
"""Callable that receives an event row and returns an inter-stimulus interval
(in samples)."""

# ---------------------------------------------------------------------------
# ISI helpers
# ---------------------------------------------------------------------------


def build_uniform_isi_sampler(
    width: int,
    offset: int = 0,
    *,
    rng: np.random.Generator | None = None,
) -> ISISampler:
    """Build an ISI sampler that draws uniformly from ``[offset, offset+width]``.

    Parameters
    ----------
    width : int
        Range of the uniform distribution (in samples).
    offset : int
        Minimum ISI value (in samples).
    rng : numpy.random.Generator or None
        Random number generator. If None, a new default generator is created.

    Returns
    -------
    sampler : ISISampler
        A callable ``(row) -> int``.
    """
    rng = rng or np.random.default_rng()

    def _sample(_row: pd.Series) -> int:
        return int(rng.integers(offset, offset + width + 1))

    return _sample

def build_gamma_isi_sampler(mean: int, scale: int = 1, offset: int = 0) -> ISISampler:
    """
    Build an ISI sampler that draws from a gamma distribution offseted by offset.
    
    Parameters
    ----------
    mean : int
        mean of the gamma distribution.
    scale : int
        scale of the gamma distribution. Equivalent to a rate value of 1/scale.
    offset : int
        offset in .samples used to shift the resulting value.
    """
    rng = rng or np.random.default_rng()

    def _sample(_row: pd.Series) -> int:
        return int(offset + rng.gamma(mean, scale))

    return _sample

def build_constant_isi_sampler(value: int = 0) -> ISISampler:
    """
    Build an ISI sampler that always returns value for an ISI.
    
    Parameters
    ----------
    value : int
        the ISI value it'll always return.
    """
    rng = rng or np.random.default_rng()

    def _sample(_row: pd.Series) -> int:
        return int(value)

    return _sample
