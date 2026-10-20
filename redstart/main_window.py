"""Main window: spectrogram viewport and transport controls."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import QEvent, QPointF, QRectF, Qt, QTimer
from PySide6.QtGui import QAction, QCursor, QGuiApplication, QKeySequence
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QFileDialog,
    QGraphicsRectItem,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from redstart.audio_document import AudioDocument, load_wav
from redstart.isolate import IsolationError, bandpass_slice
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

        # Isolation tool state.
        self._isolation_active = False
        self._isolation_selection: tuple[float, float, float, float] | None = None
        self._iso_buffer_active = False  # engine currently holds the filtered loop buffer
        self._saved_position_s = 0.0
        self._draw_anchor: QPointF | None = None
        self._box_item: QGraphicsRectItem | None = None

        self._status = QLabel("Open a WAV file to begin.")
        self._play_btn = QPushButton("Play")
        self._play_btn.setEnabled(False)
        # Avoid Space activating the button when focused; Space is the window shortcut.
        self._play_btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._play_btn.clicked.connect(self._toggle_play)

        self._isolate_btn = QPushButton("Isolate")
        self._isolate_btn.setCheckable(True)
        self._isolate_btn.setEnabled(False)
        self._isolate_btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._isolate_btn.setToolTip("Draw a box to loop-play a band-passed selection")
        self._isolate_btn.toggled.connect(self._on_isolate_toggled)

        self._iso_play_btn = QPushButton("Play selection")
        self._iso_play_btn.setEnabled(False)
        self._iso_play_btn.setVisible(False)
        self._iso_play_btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._iso_play_btn.clicked.connect(self._toggle_isolation_play)

        self._iso_close_btn = QPushButton("Close tool")
        self._iso_close_btn.setVisible(False)
        self._iso_close_btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._iso_close_btn.clicked.connect(self._close_isolation_tool)

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
        self._plot.viewport().setMouseTracking(True)
        self._plot.viewport().installEventFilter(self)

        controls = QHBoxLayout()
        controls.addWidget(self._play_btn)
        controls.addWidget(self._isolate_btn)
        controls.addWidget(self._iso_play_btn)
        controls.addWidget(self._iso_close_btn)
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

    def eventFilter(self, watched, event) -> bool:  # noqa: N802
        if watched is self._plot.viewport() and self._isolation_active:
            et = event.type()
            if et == QEvent.Type.MouseButtonPress:
                if event.button() == Qt.MouseButton.LeftButton:
                    self._on_iso_press(event)
                    return True
            elif et == QEvent.Type.MouseMove:
                if self._draw_anchor is not None:
                    self._on_iso_move(event)
                    return True
            elif et == QEvent.Type.MouseButtonRelease:
                if (
                    event.button() == Qt.MouseButton.LeftButton
                    and self._draw_anchor is not None
                ):
                    self._on_iso_release(event)
                    return True
        return super().eventFilter(watched, event)

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
        if self._isolation_active:
            self._exit_isolation(restore_document=False)

        try:
            document = load_wav(path)
        except Exception as exc:  # noqa: BLE001 — show any load failure
            QMessageBox.critical(self, "Load failed", str(exc))
            return

        self._engine.stop()
        self._document = document
        self._spectrogram = None
        # Drop the previous ImageItem buffer before allocating a new one.
        self._image.clear()
        self._engine.load(document)
        self._play_btn.setEnabled(False)
        self._play_btn.setText("Play")
        self._isolate_btn.setEnabled(False)
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
        self._isolate_btn.setEnabled(True)
        self._status.setText(
            f"{path.name}  |  {document.duration_seconds:.1f}s  |  "
            f"{document.sample_rate} Hz"
        )

    def _show_spectrogram(self, data: SpectrogramData) -> None:
        if data.db.size == 0 or data.times_s.size == 0:
            return

        self._image.setImage(data.db, autoLevels=False)
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
        if self._isolation_active:
            if self._isolation_selection is not None:
                self._toggle_isolation_play()
            return
        snap = self._engine.snapshot()
        if snap.playing:
            self._engine.pause()
            self._play_btn.setText("Play")
        else:
            self._drive_x = True
            self._engine.play()
            self._play_btn.setText("Pause")

    # --- Isolation tool -------------------------------------------------

    def _on_isolate_toggled(self, checked: bool) -> None:
        if checked:
            self._enter_isolation()
        else:
            self._exit_isolation(restore_document=True)

    def _close_isolation_tool(self) -> None:
        if self._isolate_btn.isChecked():
            self._isolate_btn.setChecked(False)
        else:
            self._exit_isolation(restore_document=True)

    def _enter_isolation(self) -> None:
        if self._document is None:
            self._isolate_btn.blockSignals(True)
            self._isolate_btn.setChecked(False)
            self._isolate_btn.blockSignals(False)
            return

        snap = self._engine.snapshot()
        self._saved_position_s = snap.position_seconds
        if snap.playing:
            self._engine.pause()
            self._play_btn.setText("Play")

        self._isolation_active = True
        self._isolation_selection = None
        self._iso_buffer_active = False
        self._draw_anchor = None
        self._clear_box_item()
        self._play_btn.setEnabled(False)
        self._iso_play_btn.setVisible(False)
        self._iso_play_btn.setEnabled(False)
        self._iso_play_btn.setText("Play selection")
        self._iso_close_btn.setVisible(True)
        self._plot.getViewBox().setMouseEnabled(x=False, y=False)
        self._status.setText("Isolation: drag a box on the spectrogram.")

    def _exit_isolation(self, *, restore_document: bool) -> None:
        self._isolation_active = False
        self._isolation_selection = None
        self._iso_buffer_active = False
        self._draw_anchor = None
        self._clear_box_item()
        self._iso_play_btn.setVisible(False)
        self._iso_play_btn.setEnabled(False)
        self._iso_play_btn.setText("Play selection")
        self._iso_close_btn.setVisible(False)
        self._plot.getViewBox().setMouseEnabled(x=True, y=False)

        if self._isolate_btn.isChecked():
            self._isolate_btn.blockSignals(True)
            self._isolate_btn.setChecked(False)
            self._isolate_btn.blockSignals(False)

        if self._document is not None:
            self._play_btn.setEnabled(True)
            if restore_document:
                self._engine.load(self._document)
                self._engine.seek(self._saved_position_s)
                self._playhead.setValue(self._saved_position_s)
                self._play_btn.setText("Play")
                self._status.setText(
                    f"{self._document.path.name}  |  "
                    f"{self._document.duration_seconds:.1f}s  |  "
                    f"{self._document.sample_rate} Hz"
                )

    def _clear_box_item(self) -> None:
        if self._box_item is not None:
            self._plot.removeItem(self._box_item)
            self._box_item = None

    def _ensure_box_item(self) -> QGraphicsRectItem:
        if self._box_item is None:
            item = QGraphicsRectItem()
            item.setPen(pg.mkPen((255, 220, 40), width=2))
            item.setBrush(pg.mkBrush(255, 220, 40, 40))
            item.setZValue(10)
            self._plot.addItem(item)
            self._box_item = item
        return self._box_item

    def _view_pos_from_viewport_event(self, event) -> QPointF:
        pos = event.position().toPoint()
        scene_pos = self._plot.mapToScene(pos)
        return self._plot.getViewBox().mapSceneToView(scene_pos)

    def _clamp_selection(
        self, t0: float, t1: float, f_low: float, f_high: float
    ) -> tuple[float, float, float, float]:
        assert self._document is not None
        dur = self._document.duration_seconds
        t0 = max(0.0, min(t0, dur))
        t1 = max(0.0, min(t1, dur))
        f_low = max(0.0, min(f_low, FREQ_MAX_HZ))
        f_high = max(0.0, min(f_high, FREQ_MAX_HZ))
        if t1 < t0:
            t0, t1 = t1, t0
        if f_high < f_low:
            f_low, f_high = f_high, f_low
        return t0, t1, f_low, f_high

    def _on_iso_press(self, event) -> None:
        if self._document is None:
            return
        # Redraw replaces any existing selection and stops isolation playback.
        if self._engine.snapshot().playing:
            self._engine.pause()
        self._iso_buffer_active = False
        self._iso_play_btn.setText("Play selection")
        self._draw_anchor = self._view_pos_from_viewport_event(event)
        self._ensure_box_item().setRect(QRectF(self._draw_anchor, self._draw_anchor))

    def _on_iso_move(self, event) -> None:
        if self._draw_anchor is None:
            return
        current = self._view_pos_from_viewport_event(event)
        self._ensure_box_item().setRect(QRectF(self._draw_anchor, current).normalized())

    def _on_iso_release(self, event) -> None:
        if self._document is None or self._draw_anchor is None:
            return
        current = self._view_pos_from_viewport_event(event)
        rect = QRectF(self._draw_anchor, current).normalized()
        self._draw_anchor = None

        t0, t1, f_low, f_high = self._clamp_selection(
            rect.left(), rect.right(), rect.top(), rect.bottom()
        )
        if t1 - t0 < 1e-3 or f_high - f_low < 1.0:
            self._isolation_selection = None
            self._clear_box_item()
            self._iso_play_btn.setVisible(False)
            self._iso_play_btn.setEnabled(False)
            self._status.setText("Isolation: selection too small — drag again.")
            return

        self._isolation_selection = (t0, t1, f_low, f_high)
        self._ensure_box_item().setRect(QRectF(t0, f_low, t1 - t0, f_high - f_low))
        self._iso_play_btn.setVisible(True)
        self._iso_play_btn.setEnabled(True)
        self._iso_play_btn.setText("Play selection")
        self._status.setText(
            f"Isolation: {t1 - t0:.2f}s × {f_low:.0f}–{f_high:.0f} Hz"
        )

    def _toggle_isolation_play(self) -> None:
        if not self._isolation_active or self._isolation_selection is None:
            return
        snap = self._engine.snapshot()
        if snap.playing:
            self._engine.pause()
            self._iso_play_btn.setText("Play selection")
            return
        if self._iso_buffer_active:
            self._engine.play()
            self._iso_play_btn.setText("Pause selection")
            return
        self._start_isolation_playback()

    def _start_isolation_playback(self) -> None:
        if self._document is None or self._isolation_selection is None:
            return
        t0, t1, f_low, f_high = self._isolation_selection
        try:
            filtered = bandpass_slice(
                self._document.samples,
                self._document.sample_rate,
                t0,
                t1,
                f_low,
                f_high,
            )
        except IsolationError as exc:
            self._status.setText(f"Isolation: {exc}")
            return

        self._engine.load_buffer(
            filtered, self._document.sample_rate, loop=True
        )
        self._iso_buffer_active = True
        self._engine.play()
        self._iso_play_btn.setText("Pause selection")
        self._status.setText(
            f"Isolation playing: {t1 - t0:.2f}s × {f_low:.0f}–{f_high:.0f} Hz (looping)"
        )

    def _set_x_range(self, x0: float, x1: float) -> None:
        self._ignore_range_signal = True
        try:
            self._plot.setXRange(x0, x1, padding=0)
        finally:
            self._ignore_range_signal = False

    def _on_manual_range(self, *_args) -> None:
        if self._ignore_range_signal or self._isolation_active:
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
        if self._settings.view_mode == "scroll":
            return scroll_x_range(playhead, width)
        return page_x_range_for_time(
            playhead, width, duration_s=self._document.duration_seconds
        )

    def _align_view_to_playhead(self) -> None:
        if self._document is None or self._isolation_active:
            return
        t = self._engine.snapshot().position_seconds
        width = self._visible_width()
        x0, x1 = self._mode_x_range(t, width)
        self._set_x_range(x0, x1)
        self._plot.setYRange(0.0, FREQ_MAX_HZ, padding=0)
        self._playhead.setValue(t)

    def _on_plot_click(self, event) -> None:
        # Isolation tool owns mouse interaction while active.
        if self._isolation_active:
            return
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

        if self._isolation_active:
            if self._iso_buffer_active and self._isolation_selection is not None:
                # Map isolated buffer time back into document coordinates.
                t0, t1, _, _ = self._isolation_selection
                span = max(t1 - t0, 1e-9)
                local = snap.position_seconds % span
                self._playhead.setValue(t0 + local)
            else:
                self._playhead.setValue(self._saved_position_s)
            if self._iso_buffer_active:
                self._iso_play_btn.setText(
                    "Pause selection" if snap.playing else "Play selection"
                )
            return

        t = snap.position_seconds
        self._playhead.setValue(t)
        self._play_btn.setText("Pause" if snap.playing else "Play")

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
