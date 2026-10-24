"""Dialog to choose which companion detection CSVs to load."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtWidgets import (
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QLabel,
    QVBoxLayout,
    QWidget,
)


class DetectionFilesDialog(QDialog):
    """Checkbox list of matching detection CSVs; all selected by default."""

    def __init__(
        self,
        paths: list[Path],
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Detection files")
        self._checks: list[tuple[QCheckBox, Path]] = []

        root = QVBoxLayout()
        root.addWidget(
            QLabel("Select detection CSV files to overlay (all selected by default):")
        )
        for path in paths:
            check = QCheckBox(path.name)
            check.setChecked(True)
            check.setToolTip(str(path))
            self._checks.append((check, path))
            root.addWidget(check)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        root.addWidget(buttons)
        self.setLayout(root)

    def selected_paths(self) -> list[Path]:
        return [path for check, path in self._checks if check.isChecked()]
