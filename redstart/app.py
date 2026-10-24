"""Application entry point."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Prefer PySide6 for pyqtgraph bindings.
import os

os.environ.setdefault("PYQTGRAPH_QT_LIB", "PySide6")

from PySide6.QtWidgets import QApplication

from redstart.main_window import MainWindow


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Redstart spectrogram player")
    parser.add_argument(
        "path",
        nargs="?",
        type=Path,
        help="WAV or FLAC file to open (optional; file dialog if omitted)",
    )
    args = parser.parse_args(argv)

    app = QApplication(sys.argv)
    app.setApplicationName("Redstart")

    initial = args.path
    if initial is not None and not initial.is_file():
        print(f"File not found: {initial}", file=sys.stderr)
        return 1

    window = MainWindow(initial_path=initial)
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
