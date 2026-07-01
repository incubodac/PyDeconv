"""QApplication factory and launch entry point."""

from __future__ import annotations

import sys

from PySide6.QtWidgets import QApplication

from pydeconv.gui.main_window import MainWindow


def launch() -> None:
    """Create the QApplication and show the main window.

    This is the primary public entry point for the GUI. It can be called
    from a script, from ``python -m pydeconv.gui``, or from the
    ``launch()`` function re-exported by :mod:`pydeconv.gui`.
    """
    app = QApplication.instance() or QApplication(sys.argv)
    app.setApplicationName("PyDeconv")
    app.setOrganizationName("PyDeconv")

    window = MainWindow()
    window.show()

    sys.exit(app.exec())
