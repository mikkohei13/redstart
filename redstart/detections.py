"""BirdNET-style detection CSV loading and label-lane layout."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Detection:
    start_s: float
    end_s: float
    scientific_name: str
    confidence: float


def abbreviate_scientific_name(name: str) -> str:
    """First three letters of genus and epithet, no space (e.g. Larus canus → Larcan)."""
    parts = name.split()
    if len(parts) >= 2:
        return f"{parts[0][:3]}{parts[1][:3]}"
    return name[:6] if name else name


def find_detections_csv(audio_path: Path) -> Path | None:
    """Return the first CSV in the audio's directory whose name starts with the stem."""
    audio_path = Path(audio_path)
    stem = audio_path.stem
    directory = audio_path.parent
    if not directory.is_dir():
        return None
    matches = sorted(
        p for p in directory.glob("*.csv") if p.name.startswith(stem)
    )
    return matches[0] if matches else None


def load_detections(csv_path: Path) -> list[Detection]:
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
                )
            )
    return detections


_TOP_SPECIES_LANES = 4


def assign_lanes(detections: list[Detection]) -> list[tuple[Detection, int]]:
    """Place the 4 most common species on fixed lanes; stagger the rest below.

    Lanes 0..3 are the top species by detection count (most common on top).
    Detections of the same top species share one lane and may overlap.
    All other species start at lane 4 and are staggered so time ranges do not
    share a y.
    """
    if not detections:
        return []

    counts: dict[str, int] = {}
    for det in detections:
        counts[det.scientific_name] = counts.get(det.scientific_name, 0) + 1

    # Most common first; name break for stable ties.
    ranked = sorted(counts.keys(), key=lambda name: (-counts[name], name))
    top = ranked[:_TOP_SPECIES_LANES]
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
