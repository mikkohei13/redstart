"""Application settings model and persistence."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from PySide6.QtCore import QSettings

ViewMode = Literal["scroll", "page"]
ColorPreset = Literal["magma", "viridis", "cividis", "turbo"]

COLOR_PRESETS: tuple[ColorPreset, ...] = ("magma", "viridis", "cividis", "turbo")

_ORG = "Redstart"
_APP = "Redstart"
_KEY_VIEW_MODE = "view/mode"
_KEY_COLOR = "display/colormap"
_KEY_FLOOR = "display/db_floor"
_KEY_CEILING = "display/db_ceiling"
_KEY_CONTRAST = "display/contrast"

_DEFAULT_FLOOR = -80.0
_DEFAULT_CEILING = -20.0
_DEFAULT_CONTRAST = 1.0


@dataclass
class AppSettings:
    """User preferences. Add fields here as new settings appear."""

    view_mode: ViewMode = "scroll"
    colormap: ColorPreset = "magma"
    db_floor: float = _DEFAULT_FLOOR
    db_ceiling: float = _DEFAULT_CEILING
    contrast: float = _DEFAULT_CONTRAST


def _read_float(store: QSettings, key: str, default: float) -> float:
    raw = store.value(key, default)
    try:
        return float(raw)
    except (TypeError, ValueError):
        return default


def _read_colormap(store: QSettings) -> ColorPreset:
    raw = store.value(_KEY_COLOR, "magma")
    if raw in COLOR_PRESETS:
        return raw  # type: ignore[return-value]
    return "magma"


def load_settings() -> AppSettings:
    store = QSettings(_ORG, _APP)
    raw_mode = store.value(_KEY_VIEW_MODE, "scroll")
    mode: ViewMode = "page" if raw_mode == "page" else "scroll"
    floor = _read_float(store, _KEY_FLOOR, _DEFAULT_FLOOR)
    ceiling = _read_float(store, _KEY_CEILING, _DEFAULT_CEILING)
    if floor >= ceiling:
        floor, ceiling = _DEFAULT_FLOOR, _DEFAULT_CEILING
    contrast = _read_float(store, _KEY_CONTRAST, _DEFAULT_CONTRAST)
    contrast = max(0.25, min(contrast, 4.0))
    return AppSettings(
        view_mode=mode,
        colormap=_read_colormap(store),
        db_floor=floor,
        db_ceiling=ceiling,
        contrast=contrast,
    )


def save_settings(settings: AppSettings) -> None:
    store = QSettings(_ORG, _APP)
    store.setValue(_KEY_VIEW_MODE, settings.view_mode)
    store.setValue(_KEY_COLOR, settings.colormap)
    store.setValue(_KEY_FLOOR, settings.db_floor)
    store.setValue(_KEY_CEILING, settings.db_ceiling)
    store.setValue(_KEY_CONTRAST, settings.contrast)
