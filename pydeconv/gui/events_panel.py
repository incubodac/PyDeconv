"""Events panel — load and preview event / design-matrix tables.

Displays a loaded ``pandas.DataFrame`` in a scrollable table view with a
summary header (filename, shape, column dtypes) and, when a ``type`` column
is present, basic event-type statistics.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
from PySide6.QtCore import Qt, QAbstractTableModel, QModelIndex
from PySide6.QtWidgets import (
    QGroupBox,
    QHeaderView,
    QLabel,
    QTableView,
    QVBoxLayout,
    QWidget,
)


# ---------------------------------------------------------------------------
# Table model
# ---------------------------------------------------------------------------


class EventsTableModel(QAbstractTableModel):
    """Qt model wrapping a ``pandas.DataFrame`` for display in a QTableView.

    Parameters
    ----------
    df : pd.DataFrame or None
        Initial data.  Can be replaced later via :meth:`set_dataframe`.
    """

    def __init__(
        self, df: pd.DataFrame | None = None, parent: Any = None
    ) -> None:
        super().__init__(parent)
        self._df: pd.DataFrame = df if df is not None else pd.DataFrame()

    # -- public --

    def set_dataframe(self, df: pd.DataFrame) -> None:
        """Replace the backing DataFrame and refresh the view."""
        self.beginResetModel()
        self._df = df
        self.endResetModel()

    # -- QAbstractTableModel interface --

    def rowCount(self, parent: QModelIndex = QModelIndex()) -> int:  # noqa: N802
        return len(self._df)

    def columnCount(self, parent: QModelIndex = QModelIndex()) -> int:  # noqa: N802
        return len(self._df.columns)

    def headerData(  # noqa: N802
        self,
        section: int,
        orientation: Qt.Orientation,
        role: int = Qt.DisplayRole,
    ) -> str | None:
        if role != Qt.DisplayRole:
            return None
        if orientation == Qt.Horizontal:
            return str(self._df.columns[section])
        return str(section)

    def data(self, index: QModelIndex, role: int = Qt.DisplayRole) -> str | None:
        if role != Qt.DisplayRole:
            return None
        val = self._df.iat[index.row(), index.column()]
        if isinstance(val, float):
            return f"{val:.6g}"
        return str(val)


# ---------------------------------------------------------------------------
# EventsPanel widget
# ---------------------------------------------------------------------------


class EventsPanel(QWidget):
    """Panel for previewing a loaded events / design-matrix DataFrame.

    Parameters
    ----------
    parent : QWidget or None
        Parent widget.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._df: pd.DataFrame | None = None
        self._setup_ui()

    # -- public API --

    def set_events(self, df: pd.DataFrame, filename: str = "") -> None:
        """Populate the panel with a new events DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            The events table.
        filename : str
            Display name for the source file.
        """
        self._df = df
        self._model.set_dataframe(df)

        # Update summary labels
        n_rows, n_cols = df.shape
        self._shape_label.setText(f"Shape: {n_rows} rows \u00d7 {n_cols} columns")
        self._file_label.setText(filename or "\u2014")

        # Column dtype summary
        dtype_parts = [f"{c} ({df[c].dtype})" for c in df.columns[:20]]
        if len(df.columns) > 20:
            dtype_parts.append(f"\u2026 +{len(df.columns) - 20} more")
        self._cols_label.setText("Columns: " + ", ".join(dtype_parts))

        # Event-type info (if a 'type' column exists)
        if "type" in df.columns:
            n_types = df["type"].nunique()
            counts = df["type"].value_counts()
            top = counts.head(5)
            parts = [f"{t} ({c})" for t, c in top.items()]
            if n_types > 5:
                parts.append(f"\u2026 +{n_types - 5} more")
            self._events_label.setText(
                f"Event types: {n_types}  \u2014  " + ", ".join(parts)
            )
            self._events_label.show()
        else:
            self._events_label.hide()

    # -- UI construction --

    def _setup_ui(self) -> None:
        """Build the panel layout."""
        layout = QVBoxLayout(self)

        # -- Summary header --
        header_box = QGroupBox("Events Summary")
        header_layout = QVBoxLayout(header_box)

        self._file_label = QLabel("No events file loaded")
        self._file_label.setStyleSheet("font-weight: bold;")
        header_layout.addWidget(self._file_label)

        self._shape_label = QLabel("")
        header_layout.addWidget(self._shape_label)

        self._cols_label = QLabel("")
        self._cols_label.setWordWrap(True)
        header_layout.addWidget(self._cols_label)

        self._events_label = QLabel("")
        self._events_label.setWordWrap(True)
        self._events_label.hide()
        header_layout.addWidget(self._events_label)

        layout.addWidget(header_box)

        # -- Table view --
        table_box = QGroupBox("Data Preview")
        table_layout = QVBoxLayout(table_box)

        self._model = EventsTableModel()
        self._table = QTableView()
        self._table.setModel(self._model)
        self._table.setSelectionBehavior(QTableView.SelectRows)
        self._table.setAlternatingRowColors(True)
        self._table.horizontalHeader().setStretchLastSection(True)
        self._table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeToContents
        )
        table_layout.addWidget(self._table)

        layout.addWidget(table_box, stretch=1)
