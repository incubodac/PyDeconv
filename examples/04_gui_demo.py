"""
04 — GUI Demo
=============
Launch the PyDeconv graphical interface for interactive deconvolution
analysis. The GUI allows loading data, configuring the design matrix,
fitting models, and inspecting results — all without writing code.

Requirements
------------
Install PySide6 first::

    pip install PySide6

Usage
-----
Run this script or use the command line::

    python -m pydeconv.gui
"""

from pydeconv.gui import launch

launch()
