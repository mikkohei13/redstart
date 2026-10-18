"""Main window: spectrogram viewport and transport controls."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import QRectF, Qt, QTimer
from PySide6.QtGui import QAction, QCursor, QGuiApplication, QKeySequence
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from redstart.audio_document import AudioDocument, load_wav
from redstart.playback import PlaybackEngine
from redstart.settings import AppSettings, ColorPreset, ViewMode, load_settings, save_settings
from redstart.settings_dialog import SettingsDialog
from redstart.spectrogram import (
    FREQ_MAX_HZ,
    VIEWPORT_SECONDS,
    SpectrogramData,
    compute_full,
    page_x_range_for_time,
    scroll_x_range,
)

# Default folder for the open dialog.
DEFAULT_AUDIO_DIR = (
    Path(__file__).resolve().parent.parent / "audio" / "thorsvik" / "data"
)


def _spectrogram_lut(colormap: ColorPreset, contrast: float, n: int = 256) -> np.ndarray:
    """Build an RGB lookup table; contrast warps the colour curve (1 = linear)."""
    cmap = pg.colormap.get(colormap)
    contrast = max(float(contrast), 1e-3)
    if abs(contrast - 1.0) < 1e-6:
        return cmap.getLookupTable(nPts=n)
    x = np.linspace(0.0, 1.0, n, dtype=np.float64)
    mapped = cmap.map(np.power(x, 1.0 / contrast), mode="byte")
    return np.ascontiguousarray(mapped[:, :3])


class MainWindow(QMainWindow):
    def __init__(self, initial_path: Path | None = None) -> None:
        super().__init__()
        self.setWindowTitle("Redstart")
        self.resize(1200, 600)

        self._document: AudioDocument | None = None
        self._spectrogram: SpectrogramData | None = None
        self._engine = PlaybackEngine()
        self._settings = load_settings()
        # When True, Page mode drives X from the playhead (Scroll always does).
        self._drive_x = True
        # Ignore ViewBox signals caused by our own setXRange calls.
        self._ignore_range_signal = False

        self._status = QLabel("Open a WAV file to begin.")
        self._play_btn = QPushButton("Play")
        self._play_btn.setEnabled(False)
        # Avoid Space activating the button when focused; Space is the window shortcut.
        self._play_btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._play_btn.clicked.connect(self._toggle_play)

        self._mode_combo = QComboBox()
        self._mode_combo.addItem("Scroll", "scroll")
        self._mode_combo.addItem("Page", "page")
        self._mode_combo.setToolTip("View mode")
        self._sync_mode_combo()
        self._mode_combo.currentIndexChanged.connect(self._on_mode_combo_changed)

        self._plot = pg.PlotWidget()
        self._plot.setLabel("bottom", "Time", units="s")
        self._plot.setLabel("left", "Frequency", units="Hz")
        self._plot.showGrid(x=False, y=True, alpha=0.2)
        self._plot.setMenuEnabled(False)
        vb = self._plot.getViewBox()
        vb.setMouseEnabled(x=True, y=False)
        vb.setLimits(yMin=0.0, yMax=FREQ_MAX_HZ)
        self._image = pg.ImageItem(axisOrder="row-major")
        self._plot.addItem(self._image)
        self._apply_display()
        self._playhead = pg.InfiniteLine(
            pos=0,
            angle=90,
            pen=pg.mkPen("w", width=2),
            movable=False,
        )
        self._plot.addItem(self._playhead)
        self._plot.scene().sigMouseClicked.connect(self._on_plot_click)
        vb.sigRangeChangedManually.connect(self._on_manual_range)

        controls = QHBoxLayout()
        controls.addWidget(self._play_btn)
        controls.addWidget(QLabel("Mode"))
        controls.addWidget(self._mode_combo)
        controls.addWidget(self._status, stretch=1)

        root = QVBoxLayout()
        root.addWidget(self._plot, stretch=1)
        root.addLayout(controls)

        container = QWidget()
        container.setLayout(root)
        self.setCentralWidget(container)

        self._build_menu()

        self._timer = QTimer(self)
        self._timer.setInterval(33)  # ~30 Hz UI refresh
        self._timer.timeout.connect(self._on_tick)
        self._timer.start()

        if initial_path is not None:
            self.load_path(initial_path)
        else:
            QTimer.singleShot(0, self._open_file_dialog)

    def _build_menu(self) -> None:
        file_menu = self.menuBar().addMenu("&File")
        open_action = QAction("&Open…", self)
        open_action.setShortcut(QKeySequence.StandardKey.Open)
        open_action.triggered.connect(self._open_file_dialog)
        file_menu.addAction(open_action)

        settings_menu = self.menuBar().addMenu("&Settings")
        prefs_action = QAction("&Settings…", self)
        prefs_action.setShortcut(QKeySequence.StandardKey.Preferences)
        prefs_action.triggered.connect(self._open_settings)
        settings_menu.addAction(prefs_action)

        play_pause = QAction("Play/Pause", self)
        play_pause.setShortcut(QKeySequence(Qt.Key.Key_Space))
        play_pause.triggered.connect(self._toggle_play)
        self.addAction(play_pause)

    def closeEvent(self, event) -> None:  # noqa: N802
        self._engine.stop()
        super().closeEvent(event)

    def _sync_mode_combo(self) -> None:
        index = self._mode_combo.findData(self._settings.view_mode)
        self._mode_combo.blockSignals(True)
        self._mode_combo.setCurrentIndex(max(0, index))
        self._mode_combo.blockSignals(False)

    def _on_mode_combo_changed(self) -> None:
        mode: ViewMode = self._mode_combo.currentData()
        if mode == self._settings.view_mode:
            return
        self._settings.view_mode = mode
        save_settings(self._settings)
        self._drive_x = True
        self._align_view_to_playhead()

    def _open_settings(self) -> None:
        dialog = SettingsDialog(self._settings, self)
        if dialog.exec() != SettingsDialog.DialogCode.Accepted:
            return
        self._apply_settings(dialog.result_settings())

    def _apply_settings(self, settings: AppSettings) -> None:
        self._settings = settings
        save_settings(settings)
        self._sync_mode_combo()
        self._apply_display()
        self._drive_x = True
        self._align_view_to_playhead()

    def _apply_display(self) -> None:
        """Update colour LUT and dB levels without recomputing the STFT."""
        self._image.setLookupTable(
            _spectrogram_lut(self._settings.colormap, self._settings.contrast)
        )
        levels = (self._settings.db_floor, self._settings.db_ceiling)
        self._image.setLevels(levels)

    def load_path(self, path: Path) -> None:
        try:
            document = load_wav(path)
        except Exception as exc:  # noqa: BLE001 — show any load failure
            QMessageBox.critical(self, "Load failed", str(exc))
            return

        self._engine.stop()
        self._document = document
        self._spectrogram = None
        self._engine.load(document)
        self._play_btn.setEnabled(False)
        self._play_btn.setText("Play")
        self.setWindowTitle(f"Redstart — {path.name}")
        self._status.setText(
            f"Computing spectrogram for {path.name} "
            f"({document.duration_seconds:.0f}s)…"
        )
        self._playhead.setValue(0.0)
        QApplication.processEvents()

        QGuiApplication.setOverrideCursor(QCursor(Qt.CursorShape.WaitCursor))
        error: Exception | None = None
        spectrogram: SpectrogramData | None = None
        try:
            spectrogram = compute_full(document)
        except Exception as exc:  # noqa: BLE001
            error = exc
        finally:
            QGuiApplication.restoreOverrideCursor()

        if error is not None or spectrogram is None:
            self._document = None
            self._status.setText("Spectrogram failed.")
            QMessageBox.critical(self, "Spectrogram failed", str(error))
            return

        self._spectrogram = spectrogram
        self._show_spectrogram(spectrogram)
        self._drive_x = True
        width = VIEWPORT_SECONDS
        x0, x1 = self._mode_x_range(0.0, width)
        self._set_x_range(x0, x1)
        self._plot.setYRange(0.0, FREQ_MAX_HZ, padding=0)
        self._play_btn.setEnabled(True)
        self._status.setText(
            f"{path.name}  |  {document.duration_seconds:.1f}s  |  "
            f"{document.sample_rate} Hz"
        )

    def _show_spectrogram(self, data: SpectrogramData) -> None:
        if data.db.size == 0 or data.times_s.size == 0:
            return

        levels = (self._settings.db_floor, self._settings.db_ceiling)
        self._image.setImage(data.db, autoLevels=False, levels=levels)
        self._apply_display()

        t_start = float(data.times_s[0])
        t_end = float(data.times_s[-1])
        if t_end <= t_start:
            t_end = t_start + 1e-3
        dt = (t_end - t_start) / max(data.db.shape[1] - 1, 1)
        # Extend rect by half a frame on each side so columns fill the timeline.
        x0 = t_start - dt / 2
        x1 = t_end + dt / 2

        f_top = float(data.freqs_hz[-1]) if len(data.freqs_hz) else FREQ_MAX_HZ
        self._image.setRect(QRectF(x0, 0.0, x1 - x0, f_top))
        self._plot.setYRange(0.0, FREQ_MAX_HZ, padding=0)

    def _open_file_dialog(self) -> None:
        start_dir = (
            str(DEFAULT_AUDIO_DIR) if DEFAULT_AUDIO_DIR.is_dir() else str(Path.home())
        )
        path_str, _ = QFileDialog.getOpenFileName(
            self,
            "Open WAV",
            start_dir,
            "WAV files (*.wav *.WAV);;All files (*)",
        )
        if path_str:
            self.load_path(Path(path_str))

    def _toggle_play(self) -> None:
        if self._document is None:
            return
        snap = self._engine.snapshot()
        if snap.playing:
            self._engine.pause()
            self._play_btn.setText("Play")
        else:
            self._drive_x = True
            self._engine.play()
            self._play_btn.setText("Pause")

    def _set_x_range(self, x0: float, x1: float) -> None:
        self._ignore_range_signal = True
        try:
            self._plot.setXRange(x0, x1, padding=0)
        finally:
            self._ignore_range_signal = False

    def _on_manual_range(self, *_args) -> None:
        if self._ignore_range_signal:
            return
        self._plot.setYRange(0.0, FREQ_MAX_HZ, padding=0)
        x0, x1 = self._visible_x_range()
        if self._document is None:
            return

        if self._settings.view_mode == "scroll":
            # Pan/zoom navigates time: seek to the new view center; playhead stays centered.
            width = max(x1 - x0, 1e-3)
            center = 0.5 * (x0 + x1)
            center = max(0.0, min(center, self._document.duration_seconds))
            self._engine.seek(center)
            self._playhead.setValue(center)
            nx0, nx1 = scroll_x_range(center, width)
            self._set_x_range(nx0, nx1)
            return

        # Page: user may free-pan until playback hits a page boundary.
        self._drive_x = False

    def _visible_x_range(self) -> tuple[float, float]:
        x0, x1 = self._plot.getViewBox().viewRange()[0]
        return float(x0), float(x1)

    def _visible_width(self) -> float:
        x0, x1 = self._visible_x_range()
        return max(x1 - x0, 1e-3)

    def _mode_x_range(self, playhead: float, width: float) -> tuple[float, float]:
        assert self._document is not None
        duration_s = self._document.duration_seconds
        if self._settings.view_mode == "scroll":
            return scroll_x_range(playhead, width, duration_s=duration_s)
        return page_x_range_for_time(playhead, width, duration_s=duration_s)

    def _align_view_to_playhead(self) -> None:
        if self._document is None:
            return
        t = self._engine.snapshot().position_seconds
        width = self._visible_width()
        x0, x1 = self._mode_x_range(t, width)
        self._set_x_range(x0, x1)
        self._plot.setYRange(0.0, FREQ_MAX_HZ, padding=0)
        self._playhead.setValue(t)

    def _on_plot_click(self, event) -> None:
        # Single-click is ignored; double-click seeks the playhead.
        if self._document is None or event.button() != Qt.MouseButton.LeftButton:
            return
        if not event.double():
            return
        if not self._plot.sceneBoundingRect().contains(event.scenePos()):
            return
        mouse_point = self._plot.getViewBox().mapSceneToView(event.scenePos())
        t = float(mouse_point.x())
        t = max(0.0, min(t, self._document.duration_seconds))
        self._engine.seek(t)
        self._playhead.setValue(t)
        self._drive_x = True
        self._align_view_to_playhead()

    def _on_tick(self) -> None:
        if self._document is None or self._spectrogram is None:
            return
        snap = self._engine.snapshot()
        t = snap.position_seconds
        self._playhead.setValue(t)

        if snap.finished:
            self._play_btn.setText("Play")

        width = self._visible_width()
        x0, x1 = self._visible_x_range()

        if self._settings.view_mode == "scroll":
            # Always keep the playhead at the horizontal center.
            nx0, nx1 = scroll_x_range(t, width)
            # Skip no-op range updates; each setXRange triggers a full plot paint.
            if abs(nx0 - x0) > 1e-4 or abs(nx1 - x1) > 1e-4:
                self._set_x_range(nx0, nx1)
        else:
            # Page: jump when playhead leaves the right edge while driving/playing.
            if self._drive_x or snap.playing:
                if t >= x1 - 1e-6 or t < x0:
                    self._drive_x = True
                    nx0, nx1 = page_x_range_for_time(
                        t, width, duration_s=self._document.duration_seconds
                    )
                    self._set_x_range(nx0, nx1)

        if not snap.playing and self._play_btn.text() == "Pause":
            self._play_btn.setText("Play")
