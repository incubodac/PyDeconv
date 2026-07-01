"""Data information panel — channel list and per-channel statistics.

This panel is the first functional view in the PyDeconv GUI.  It presents
two side-by-side tables: one listing channel metadata (name, type, unit)
and another showing basic descriptive statistics per channel.
"""

from __future__ import annotations

from typing import Any

import mne
from PySide6.QtCore import Qt, QAbstractTableModel, QModelIndex, QThreadPool
from PySide6.QtWidgets import (
    QHBoxLayout,
    QHeaderView,
    QPushButton,
    QTableView,
    QVBoxLayout,
    QWidget,
    QLabel,
    QProgressBar,
    QGroupBox,
)

from pydeconv.gui.workers import StatsWorker


# ---------------------------------------------------------------------------
# Table models
# ---------------------------------------------------------------------------


class ChannelInfoModel(QAbstractTableModel):
    """Model backing the channel-info table (name, type, unit).

    Parameters
    ----------
    raw : mne.io.BaseRaw or None
        If provided, immediately populate from the Raw's info dict.
    """

    _HEADERS = ("Channel", "Type", "Unit")

    def __init__(self, raw: mne.io.BaseRaw | None = None, parent: Any = None):
        super().__init__(parent)
        self._rows: list[tuple[str, str, str]] = []
        if raw is not None:
            self.set_raw(raw)

    # -- public --

    def set_raw(self, raw: mne.io.BaseRaw) -> None:
        """Populate the model from a loaded Raw object."""
        self.beginResetModel()
        self._rows = []
        for ch_info in raw.info["chs"]:
            name = ch_info["ch_name"]
            ch_type = mne.channel_type(raw.info, raw.ch_names.index(name))
            # MNE stores unit as FIFF constant; map common ones
            unit = _fiff_unit_label(ch_info.get("unit", 0))
            self._rows.append((name, ch_type, unit))
        self.endResetModel()

    # -- QAbstractTableModel interface --

    def rowCount(self, parent: QModelIndex = QModelIndex()) -> int:  # noqa: N802
        return len(self._rows)

    def columnCount(self, parent: QModelIndex = QModelIndex()) -> int:  # noqa: N802
        return len(self._HEADERS)

    def headerData(  # noqa: N802
        self, section: int, orientation: Qt.Orientation, role: int = Qt.DisplayRole
    ) -> str | None:
        if role == Qt.DisplayRole and orientation == Qt.Horizontal:
            return self._HEADERS[section]
        return None

    def data(self, index: QModelIndex, role: int = Qt.DisplayRole) -> str | None:
        if role == Qt.DisplayRole:
            return self._rows[index.row()][index.column()]
        return None


class ChannelStatsModel(QAbstractTableModel):
    """Model backing the per-channel statistics table.

    Parameters
    ----------
    stats : list of dict or None
        Each dict has keys: channel, mean, std, min, max, median.
    """

    _HEADERS = ("Channel", "Mean", "Std", "Min", "Max", "Median")
    _KEYS = ("channel", "mean", "std", "min", "max", "median")

    def __init__(self, stats: list[dict[str, Any]] | None = None, parent: Any = None):
        super().__init__(parent)
        self._stats: list[dict[str, Any]] = stats or []

    # -- public --

    def set_stats(self, stats: list[dict[str, Any]]) -> None:
        """Replace the statistics data."""
        self.beginResetModel()
        self._stats = stats
        self.endResetModel()

    # -- QAbstractTableModel interface --

    def rowCount(self, parent: QModelIndex = QModelIndex()) -> int:  # noqa: N802
        return len(self._stats)

    def columnCount(self, parent: QModelIndex = QModelIndex()) -> int:  # noqa: N802
        return len(self._HEADERS)

    def headerData(  # noqa: N802
        self, section: int, orientation: Qt.Orientation, role: int = Qt.DisplayRole
    ) -> str | None:
        if role == Qt.DisplayRole and orientation == Qt.Horizontal:
            return self._HEADERS[section]
        return None

    def data(self, index: QModelIndex, role: int = Qt.DisplayRole) -> str | None:
        if role != Qt.DisplayRole:
            return None
        row = self._stats[index.row()]
        key = self._KEYS[index.column()]
        val = row[key]
        if isinstance(val, float):
            return f"{val:.6g}"
        return str(val)


# ---------------------------------------------------------------------------
# DataInfoPanel widget
# ---------------------------------------------------------------------------


class DataInfoPanel(QWidget):
    """Side-by-side channel info and statistics panel.

    Parameters
    ----------
    parent : QWidget or None
        Parent widget.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._raw: mne.io.BaseRaw | None = None
        self._pool = QThreadPool.globalInstance()

        self._setup_ui()

    # -- public API --

    def set_raw(self, raw: mne.io.BaseRaw) -> None:
        """Load channel info and kick off stats computation.

        Parameters
        ----------
        raw : mne.io.BaseRaw
            The loaded raw EEG data.
        """
        self._raw = raw
        self._info_model.set_raw(raw)
        self._compute_stats()

    # -- UI construction --

    def _setup_ui(self) -> None:
        """Build the panel layout."""
        layout = QHBoxLayout(self)

        # ── Left: Channel info ──
        left_box = QGroupBox("Channel Info")
        left_layout = QVBoxLayout(left_box)

        self._info_model = ChannelInfoModel()
        self._info_table = QTableView()
        self._info_table.setModel(self._info_model)
        self._info_table.setSelectionBehavior(QTableView.SelectRows)
        self._info_table.setAlternatingRowColors(True)
        self._info_table.horizontalHeader().setStretchLastSection(True)
        self._info_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeToContents
        )
        left_layout.addWidget(self._info_table)
        layout.addWidget(left_box, stretch=1)

        # ── Right: Statistics ──
        right_box = QGroupBox("Channel Statistics")
        right_layout = QVBoxLayout(right_box)

        self._stats_model = ChannelStatsModel()
        self._stats_table = QTableView()
        self._stats_table.setModel(self._stats_model)
        self._stats_table.setSelectionBehavior(QTableView.SelectRows)
        self._stats_table.setAlternatingRowColors(True)
        self._stats_table.horizontalHeader().setStretchLastSection(True)
        self._stats_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeToContents
        )
        right_layout.addWidget(self._stats_table)

        # Progress bar + refresh button
        bottom_row = QHBoxLayout()
        self._progress = QProgressBar()
        self._progress.setRange(0, 100)
        self._progress.setValue(0)
        self._progress.setTextVisible(True)
        self._progress.setFixedHeight(18)
        bottom_row.addWidget(self._progress, stretch=1)

        self._refresh_btn = QPushButton("Refresh Stats")
        self._refresh_btn.setEnabled(False)
        self._refresh_btn.clicked.connect(self._compute_stats)
        bottom_row.addWidget(self._refresh_btn)

        right_layout.addLayout(bottom_row)
        layout.addWidget(right_box, stretch=2)

    # -- stats computation --

    def _compute_stats(self) -> None:
        """Launch background stats computation."""
        if self._raw is None:
            return
        self._refresh_btn.setEnabled(False)
        self._progress.setValue(0)

        worker = StatsWorker(self._raw)
        worker.signals.progress.connect(self._on_stats_progress)
        worker.signals.finished.connect(self._on_stats_done)
        worker.signals.error.connect(self._on_stats_error)
        self._pool.start(worker)

    def _on_stats_progress(self, pct: int) -> None:
        self._progress.setValue(pct)

    def _on_stats_done(self, stats: list[dict]) -> None:
        self._stats_model.set_stats(stats)
        self._progress.setValue(100)
        self._refresh_btn.setEnabled(True)

    def _on_stats_error(self, msg: str) -> None:
        self._progress.setValue(0)
        self._refresh_btn.setEnabled(True)
        # TODO: show error dialog
        print(f"[StatsWorker ERROR] {msg}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Subset of FIFF unit constants → human-readable labels
_FIFF_UNITS = {
    0: "—",
    107: "V",       # FIFF_UNIT_V
    201: "T",       # FIFF_UNIT_T
    202: "T/m",     # FIFF_UNIT_T_M
}


def _fiff_unit_label(unit_code: int) -> str:
    """Map an MNE/FIFF unit constant to a readable string."""
    return _FIFF_UNITS.get(unit_code, str(unit_code))
