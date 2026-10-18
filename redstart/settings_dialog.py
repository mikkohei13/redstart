"""Modal settings dialog with grouped sections for future growth."""

from __future__ import annotations

from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QMessageBox,
    QVBoxLayout,
    QWidget,
)

from redstart.settings import COLOR_PRESETS, AppSettings, ColorPreset, ViewMode


class SettingsDialog(QDialog):
    def __init__(self, settings: AppSettings, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Settings")

        self._view_mode = QComboBox()
        self._view_mode.addItem("Scroll", "scroll")
        self._view_mode.addItem("Page", "page")
        index = self._view_mode.findData(settings.view_mode)
        self._view_mode.setCurrentIndex(max(0, index))

        view_form = QFormLayout()
        view_form.addRow("View mode", self._view_mode)
        view_group = QGroupBox("View")
        view_group.setLayout(view_form)

        self._colormap = QComboBox()
        for name in COLOR_PRESETS:
            self._colormap.addItem(name.capitalize(), name)
        color_index = self._colormap.findData(settings.colormap)
        self._colormap.setCurrentIndex(max(0, color_index))

        self._floor = QDoubleSpinBox()
        self._floor.setRange(-120.0, 0.0)
        self._floor.setDecimals(1)
        self._floor.setSingleStep(1.0)
        self._floor.setSuffix(" dB")
        self._floor.setValue(settings.db_floor)
        self._floor.setToolTip("Quiet end of the display range (dB).")

        self._ceiling = QDoubleSpinBox()
        self._ceiling.setRange(-120.0, 0.0)
        self._ceiling.setDecimals(1)
        self._ceiling.setSingleStep(1.0)
        self._ceiling.setSuffix(" dB")
        self._ceiling.setValue(settings.db_ceiling)
        self._ceiling.setToolTip("Loud end of the display range (dB).")

        self._contrast = QDoubleSpinBox()
        self._contrast.setRange(0.25, 4.0)
        self._contrast.setDecimals(2)
        self._contrast.setSingleStep(0.05)
        self._contrast.setValue(settings.contrast)
        self._contrast.setToolTip(
            "Colour curve steepness. 1 = linear; higher = more mid-range punch."
        )

        display_form = QFormLayout()
        display_form.addRow("Colour", self._colormap)
        display_form.addRow("Floor", self._floor)
        display_form.addRow("Ceiling", self._ceiling)
        display_form.addRow("Contrast", self._contrast)
        display_group = QGroupBox("Display")
        display_group.setLayout(display_form)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self._on_accept)
        buttons.rejected.connect(self.reject)

        root = QVBoxLayout()
        root.addWidget(view_group)
        root.addWidget(display_group)
        root.addStretch(1)
        root.addWidget(buttons)
        self.setLayout(root)

    def _on_accept(self) -> None:
        if self._floor.value() >= self._ceiling.value():
            QMessageBox.warning(
                self,
                "Invalid display levels",
                "Floor must be below ceiling.",
            )
            return
        self.accept()

    def result_settings(self) -> AppSettings:
        mode: ViewMode = self._view_mode.currentData()
        colormap: ColorPreset = self._colormap.currentData()
        return AppSettings(
            view_mode=mode,
            colormap=colormap,
            db_floor=float(self._floor.value()),
            db_ceiling=float(self._ceiling.value()),
            contrast=float(self._contrast.value()),
        )
