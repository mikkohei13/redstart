"""Modal settings dialog with grouped sections for future growth."""

from __future__ import annotations

from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QGroupBox,
    QVBoxLayout,
    QWidget,
)

from redstart.settings import AppSettings, ViewMode


class SettingsDialog(QDialog):
    def __init__(self, settings: AppSettings, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Settings")
        self._settings = AppSettings(view_mode=settings.view_mode)

        self._view_mode = QComboBox()
        self._view_mode.addItem("Scroll", "scroll")
        self._view_mode.addItem("Page", "page")
        index = self._view_mode.findData(settings.view_mode)
        self._view_mode.setCurrentIndex(max(0, index))

        view_form = QFormLayout()
        view_form.addRow("View mode", self._view_mode)
        view_group = QGroupBox("View")
        view_group.setLayout(view_form)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        root = QVBoxLayout()
        root.addWidget(view_group)
        root.addStretch(1)
        root.addWidget(buttons)
        self.setLayout(root)

    def result_settings(self) -> AppSettings:
        mode: ViewMode = self._view_mode.currentData()
        return AppSettings(view_mode=mode)
