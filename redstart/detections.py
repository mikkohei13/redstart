"""BirdNET-style detection CSV loading and label-lane layout."""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np


# Distinct overlay colours for comparing detection CSVs (models) side by side.
DETECTION_COLORS: tuple[tuple[int, int, int], ...] = (
    (255, 220, 40),   # yellow
    (80, 220, 255),   # cyan
    (255, 100, 200),  # magenta
    (120, 255, 120),  # lime
    (255, 140, 60),   # orange
    (200, 180, 255),  # lavender
)


@dataclass(frozen=True)
class Detection:
    start_s: float
    end_s: float
    scientific_name: str
    confidence: float
    # Index into DETECTION_COLORS / loaded source list (for multi-model compare).
    source_index: int = 0
    source_label: str = ""


def counts_per_minute(
    detections: list[Detection],
    threshold: float,
    duration_seconds: float,
) -> np.ndarray:
    """Count detections at/above threshold in each whole-file minute bin.

    Bin count is ceil(duration/60), at least 1. Each detection is assigned by
    floor(start_s / 60), clamped into range.
    """
    n_bins = max(1, int(math.ceil(max(duration_seconds, 0.0) / 60.0)))
    counts = np.zeros(n_bins, dtype=np.int32)
    for det in detections:
        if det.confidence < threshold:
            continue
        minute = int(det.start_s // 60)
        if minute < 0:
            minute = 0
        elif minute >= n_bins:
            minute = n_bins - 1
        counts[minute] += 1
    return counts


_TAXON_ABBREVIATIONS: dict[str, str] | None = None


def _load_taxon_abbreviations() -> dict[str, str]:
    """scientific name → abbreviation from packaged taxon_names.csv."""
    global _TAXON_ABBREVIATIONS
    if _TAXON_ABBREVIATIONS is not None:
        return _TAXON_ABBREVIATIONS
    path = Path(__file__).resolve().parent / "data" / "taxon_names.csv"
    mapping: dict[str, str] = {}
    # utf-8-sig strips a BOM if present on the header row.
    with path.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            scientific = (row.get("scientific") or "").strip()
            abbreviation = (row.get("abbreviation") or "").strip()
            if scientific and abbreviation:
                mapping[scientific] = abbreviation
    _TAXON_ABBREVIATIONS = mapping
    return mapping


def abbreviate_scientific_name(name: str) -> str:
    """Abbreviation from taxon_names.csv, else ``G. epithet`` (e.g. L. canus)."""
    name = (name or "").strip()
    if not name:
        return name
    found = _load_taxon_abbreviations().get(name)
    if found:
        return found
    parts = name.split()
    if len(parts) >= 2:
        return f"{parts[0][0].upper()}. {parts[1]}"
    return name


def find_detections_csvs(audio_path: Path) -> list[Path]:
    """Return matching CSVs beside the audio, then in the parent folder.

    A match is a ``*.csv`` whose filename starts with the audio stem.
    Same-folder hits come first; paths are unique and sorted within each folder.
    """
    audio_path = Path(audio_path)
    stem = audio_path.stem
    found: list[Path] = []
    seen: set[Path] = set()
    for directory in (audio_path.parent, audio_path.parent.parent):
        if not directory.is_dir():
            continue
        matches = sorted(
            p for p in directory.glob("*.csv") if p.name.startswith(stem)
        )
        for path in matches:
            key = path.resolve()
            if key in seen:
                continue
            seen.add(key)
            found.append(path)
    return found


def detection_source_label(csv_path: Path, audio_stem: str) -> str:
    """Short label from a companion CSV name (e.g. BirdNET, Muuttolinnut)."""
    name = Path(csv_path).name
    if name.startswith(audio_stem):
        rest = name[len(audio_stem) :].lstrip(".")
        if rest.lower().endswith(".csv"):
            rest = rest[:-4]
        if rest.lower().endswith(".results"):
            rest = rest[: -len(".results")]
        rest = rest.strip(".")
        if rest:
            return rest
    return Path(csv_path).stem


def load_detections(
    csv_path: Path,
    *,
    source_index: int = 0,
    source_label: str = "",
) -> list[Detection]:
    """Parse a BirdNET results CSV into Detection rows."""
    csv_path = Path(csv_path)
    detections: list[Detection] = []
    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = ("Start (s)", "End (s)", "Scientific name", "Confidence")
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {csv_path}")
        missing = [c for c in required if c not in reader.fieldnames]
        if missing:
            raise ValueError(
                f"CSV missing columns {missing}: {csv_path}"
            )
        for row in reader:
            detections.append(
                Detection(
                    start_s=float(row["Start (s)"]),
                    end_s=float(row["End (s)"]),
                    scientific_name=row["Scientific name"].strip(),
                    confidence=float(row["Confidence"]),
                    source_index=source_index,
                    source_label=source_label,
                )
            )
    return detections


_TOP_SPECIES_LANES = 4


def rank_species(
    detections: list[Detection],
    n: int = _TOP_SPECIES_LANES,
) -> list[str]:
    """Most common scientific names first, up to n (name break for ties).

    Counts pool detections from every loaded source file.
    """
    if not detections or n <= 0:
        return []
    counts: dict[str, int] = {}
    for det in detections:
        counts[det.scientific_name] = counts.get(det.scientific_name, 0) + 1
    ranked = sorted(counts.keys(), key=lambda name: (-counts[name], name))
    return ranked[:n]


def assign_lanes(detections: list[Detection]) -> list[tuple[Detection, int]]:
    """Place the 4 most common species on fixed lanes; stagger the rest below.

    Lanes 0..3 are the top species by pooled detection count across all sources
    (most common on top). Same-species hits from different sources share a lane;
    the view offsets each source vertically. All other species start at lane 4
    and are staggered so time ranges do not share a y.
    """
    if not detections:
        return []

    top = rank_species(detections)
    top_lane = {name: i for i, name in enumerate(top)}
    other_base = len(top)  # usually 4; fewer if fewer than 4 species overall

    result: list[tuple[Detection, int]] = []
    # Relative lane ends for the "other" pool (index 0 == other_base).
    other_lane_ends: list[float] = []

    for det in sorted(detections, key=lambda d: (d.start_s, d.end_s)):
        if det.scientific_name in top_lane:
            result.append((det, top_lane[det.scientific_name]))
            continue

        lane = 0
        while lane < len(other_lane_ends) and det.start_s < other_lane_ends[lane]:
            lane += 1
        if lane == len(other_lane_ends):
            other_lane_ends.append(det.end_s)
        else:
            other_lane_ends[lane] = det.end_s
        result.append((det, other_base + lane))

    return result
