"""Entry point of SAVES Studio."""

from __future__ import annotations

import sys

from PySide6.QtWidgets import QApplication

from studio.ui.main_window import MainWindow


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv if argv is None else argv
    app = QApplication(argv)
    app.setApplicationName("SAVES Studio")
    app.setStyle("Fusion")
    window = MainWindow()
    window.show()
    if len(argv) > 1:
        if argv[1] == "--demo":
            window.open_demo()
        else:
            window.open_folder(argv[1])
    return app.exec()
