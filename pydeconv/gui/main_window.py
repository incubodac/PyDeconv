"""Main application window for the PyDeconv GUI.

Provides the menu bar, status bar, and hosts the data panel as the
first functional view.  Future panels (Events, Design Matrix, Model,
Results) will be added as tabs below the data panel.
"""

from __future__ import annotations

import mne
from PySide6.QtCore import Qt, QThreadPool
from PySide6.QtWidgets import (
    QFileDialog,
    QLabel,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QSplitter,
    QStatusBar,
    QTabWidget,
    QWidget,
)

from pydeconv.gui.data_panel import DataInfoPanel
from pydeconv.gui.workers import FileLoaderWorker


class MainWindow(QMainWindow):
    """PyDeconv main application window.

    Parameters
    ----------
    parent : QWidget or None
        Parent widget.
    """

    _FILE_FILTERS = (
        "EEG files (*.set *.edf *.bdf *.vhdr *.fif);;"
        "EEGLAB (*.set);;"
        "EDF/BDF (*.edf *.bdf);;"
        "BrainVision (*.vhdr);;"
        "FIF (*.fif);;"
        "All files (*)"
    )

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._raw: mne.io.BaseRaw | None = None
        self._pool = QThreadPool.globalInstance()

        self.setWindowTitle("PyDeconv")
        self.resize(1100, 700)

        self._setup_menu()
        self._setup_central()
        self._setup_status_bar()

    # -- menu --

    def _setup_menu(self) -> None:
        """Create the menu bar."""
        menu = self.menuBar()

        file_menu = menu.addMenu("&File")
        open_action = file_menu.addAction("&Open…")
        open_action.setShortcut("Ctrl+O")
        open_action.triggered.connect(self._on_open_file)

        file_menu.addSeparator()

        quit_action = file_menu.addAction("&Quit")
        quit_action.setShortcut("Ctrl+Q")
        quit_action.triggered.connect(self.close)

    # -- central widget --

    def _setup_central(self) -> None:
        """Build the central splitter layout."""
        splitter = QSplitter(Qt.Vertical, self)

        # Top: data info panel
        self._data_panel = DataInfoPanel()
        splitter.addWidget(self._data_panel)

        # Bottom: placeholder tabs for future panels
        self._tabs = QTabWidget()
        self._tabs.addTab(
            _placeholder("Events panel — coming soon"), "Events"
        )
        self._tabs.addTab(
            _placeholder("Design Matrix builder — coming soon"),
            "Design Matrix",
        )
        self._tabs.addTab(
            _placeholder("Model fitting — coming soon"), "Model"
        )
        self._tabs.addTab(
            _placeholder("Results viewer — coming soon"), "Results"
        )
        splitter.addWidget(self._tabs)

        # Give more space to the data panel by default
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 1)

        self.setCentralWidget(splitter)

    # -- status bar --

    def _setup_status_bar(self) -> None:
        """Create the status bar with file info labels."""
        sb = QStatusBar(self)
        self.setStatusBar(sb)

        self._file_label = QLabel("No file loaded")
        sb.addWidget(self._file_label, stretch=1)

        self._info_label = QLabel("")
        sb.addPermanentWidget(self._info_label)

        self._load_progress = QProgressBar()
        self._load_progress.setRange(0, 100)
        self._load_progress.setFixedWidth(120)
        self._load_progress.setFixedHeight(16)
        self._load_progress.setTextVisible(False)
        self._load_progress.hide()
        sb.addPermanentWidget(self._load_progress)

    # -- file loading --

    def _on_open_file(self) -> None:
        """Show file dialog and start loading."""
        filepath, _ = QFileDialog.getOpenFileName(
            self, "Open EEG File", "", self._FILE_FILTERS
        )
        if not filepath:
            return

        self._file_label.setText(f"Loading: {filepath}")
        self._load_progress.setValue(0)
        self._load_progress.show()

        worker = FileLoaderWorker(filepath)
        worker.signals.progress.connect(self._on_load_progress)
        worker.signals.finished.connect(self._on_load_done)
        worker.signals.error.connect(self._on_load_error)
        self._pool.start(worker)

    def _on_load_progress(self, pct: int) -> None:
        self._load_progress.setValue(pct)

    def _on_load_done(self, raw: mne.io.BaseRaw) -> None:
        self._raw = raw
        self._load_progress.hide()

        n_ch = len(raw.ch_names)
        duration = raw.times[-1]
        sfreq = raw.info["sfreq"]

        self._file_label.setText(raw.filenames[0])
        self._info_label.setText(
            f"{n_ch} channels  |  {duration:.1f} s  |  {sfreq:.0f} Hz"
        )
        self.statusBar().showMessage("File loaded successfully", 3000)

        # Push data to the panel
        self._data_panel.set_raw(raw)

    def _on_load_error(self, msg: str) -> None:
        self._load_progress.hide()
        self._file_label.setText("Load failed")
        QMessageBox.critical(self, "Load Error", msg)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _placeholder(text: str) -> QWidget:
    """Create a centred label widget as a placeholder tab."""
    w = QWidget()
    from PySide6.QtWidgets import QVBoxLayout

    layout = QVBoxLayout(w)
    lbl = QLabel(text)
    lbl.setAlignment(Qt.AlignCenter)
    lbl.setStyleSheet("color: #888; font-size: 14px;")
    layout.addWidget(lbl)
    return w
