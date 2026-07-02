"""Background workers for file I/O and statistics computation.

Workers run heavy operations off the Qt main thread using ``QRunnable``
and ``QThreadPool``, communicating results back via ``QObject`` signals.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import mne
from PySide6.QtCore import QObject, QRunnable, Signal, Slot


# ---------------------------------------------------------------------------
# Signal containers (QRunnable cannot emit signals directly)
# ---------------------------------------------------------------------------


class WorkerSignals(QObject):
    """Signals emitted by background workers."""

    finished = Signal(object)
    """Emitted with the result object on success."""

    error = Signal(str)
    """Emitted with an error message on failure."""

    progress = Signal(int)
    """Emitted with a percentage (0–100) during long operations."""


# ---------------------------------------------------------------------------
# FileLoaderWorker
# ---------------------------------------------------------------------------


class FileLoaderWorker(QRunnable):
    """Load an EEG file with MNE in a background thread.

    Parameters
    ----------
    filepath : str
        Absolute path to the EEG file (.set, .edf, .bdf, .vhdr, .fif).
    """

    def __init__(self, filepath: str) -> None:
        super().__init__()
        self.filepath = filepath
        self.signals = WorkerSignals()
        self.setAutoDelete(True)

    @Slot()
    def run(self) -> None:
        """Execute the file loading."""
        try:
            self.signals.progress.emit(10)
            raw = _load_raw(self.filepath)
            self.signals.progress.emit(100)
            self.signals.finished.emit(raw)
        except Exception as exc:  # noqa: BLE001
            self.signals.error.emit(str(exc))


class StatsWorker(QRunnable):
    """Compute per-channel statistics from an MNE Raw object.

    Parameters
    ----------
    raw : mne.io.BaseRaw
        The loaded raw data.
    """

    def __init__(self, raw: mne.io.BaseRaw) -> None:
        super().__init__()
        self.raw = raw
        self.signals = WorkerSignals()
        self.setAutoDelete(True)

    @Slot()
    def run(self) -> None:
        """Compute statistics for every channel."""
        try:
            data = self.raw.get_data()  # (n_channels, n_times)
            n_ch = data.shape[0]
            stats: list[dict[str, Any]] = []

            for i in range(n_ch):
                ch = data[i]
                stats.append({
                    "channel": self.raw.ch_names[i],
                    "mean": float(np.mean(ch)),
                    "std": float(np.std(ch)),
                    "min": float(np.min(ch)),
                    "max": float(np.max(ch)),
                    "median": float(np.median(ch)),
                })
                pct = int((i + 1) / n_ch * 100)
                self.signals.progress.emit(pct)

            self.signals.finished.emit(stats)
        except Exception as exc:  # noqa: BLE001
            self.signals.error.emit(str(exc))


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _load_raw(filepath: str) -> mne.io.BaseRaw:
    """Dispatch to the correct MNE reader based on file extension.

    Parameters
    ----------
    filepath : str
        Path to the EEG data file.

    Returns
    -------
    raw : mne.io.BaseRaw
        Loaded raw data (preloaded into memory).
    """
    ext = filepath.rsplit(".", maxsplit=1)[-1].lower()
    loaders = {
        "set": lambda p: mne.io.read_raw_eeglab(p, preload=True),
        "edf": lambda p: mne.io.read_raw_edf(p, preload=True),
        "bdf": lambda p: mne.io.read_raw_bdf(p, preload=True),
        "vhdr": lambda p: mne.io.read_raw_brainvision(p, preload=True),
        "fif": lambda p: mne.io.read_raw_fif(p, preload=True),
    }
    loader = loaders.get(ext)
    if loader is None:
        raise ValueError(
            f"Unsupported file format '.{ext}'. "
            f"Supported: {', '.join(sorted(loaders))}"
        )
    return loader(filepath)


# ---------------------------------------------------------------------------
# EventsLoaderWorker
# ---------------------------------------------------------------------------


class EventsLoaderWorker(QRunnable):
    """Load an events / design-matrix file in a background thread.

    Supports CSV, TSV, Parquet, NumPy ``.npy``, and Excel ``.xlsx`` files.
    NumPy arrays are wrapped in a DataFrame with auto-generated column names
    (``feat_0``, ``feat_1``, …).

    Parameters
    ----------
    filepath : str
        Absolute path to the events file.
    """

    def __init__(self, filepath: str) -> None:
        super().__init__()
        self.filepath = filepath
        self.signals = WorkerSignals()
        self.setAutoDelete(True)

    @Slot()
    def run(self) -> None:
        """Execute the events file loading."""
        try:
            self.signals.progress.emit(10)
            df = _load_events(self.filepath)
            self.signals.progress.emit(100)
            self.signals.finished.emit(df)
        except Exception as exc:  # noqa: BLE001
            self.signals.error.emit(str(exc))


def _load_events(filepath: str) -> pd.DataFrame:
    """Dispatch to the correct reader based on file extension.

    Parameters
    ----------
    filepath : str
        Path to the events / design-matrix file.

    Returns
    -------
    df : pd.DataFrame
        Loaded events table.
    """
    ext = filepath.rsplit(".", maxsplit=1)[-1].lower()

    if ext in ("csv", "tsv", "txt"):
        sep = "\t" if ext == "tsv" else ","
        df = pd.read_csv(filepath, sep=sep)
    elif ext == "parquet":
        df = pd.read_parquet(filepath)
    elif ext == "npy":
        arr = np.load(filepath)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        cols = [f"feat_{i}" for i in range(arr.shape[1])]
        df = pd.DataFrame(arr, columns=cols)
    elif ext == "xlsx":
        df = pd.read_excel(filepath)
    else:
        raise ValueError(
            f"Unsupported events file format '.{ext}'. "
            f"Supported: csv, tsv, txt, parquet, npy, xlsx"
        )
    return df

