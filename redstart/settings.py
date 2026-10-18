"""Application settings model and persistence."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from PySide6.QtCore import QSettings

ViewMode = Literal["scroll", "page"]

_ORG = "Redstart"
_APP = "Redstart"
_KEY_VIEW_MODE = "view/mode"


@dataclass
class AppSettings:
    """User preferences. Add fields here as new settings appear."""

    view_mode: ViewMode = "scroll"


def load_settings() -> AppSettings:
    store = QSettings(_ORG, _APP)
    raw = store.value(_KEY_VIEW_MODE, "scroll")
    mode: ViewMode = "page" if raw == "page" else "scroll"
    return AppSettings(view_mode=mode)


def save_settings(settings: AppSettings) -> None:
    store = QSettings(_ORG, _APP)
    store.setValue(_KEY_VIEW_MODE, settings.view_mode)
