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


def assign_lanes(detections: list[Detection]) -> list[tuple[Detection, int]]:
    """Greedy lowest-lane assignment so overlapping time ranges do not share a y."""
    ordered = sorted(detections, key=lambda d: (d.start_s, d.end_s))
    # lane_ends[i] = end time of the last detection placed in lane i
    lane_ends: list[float] = []
    result: list[tuple[Detection, int]] = []
    for det in ordered:
        lane = 0
        while lane < len(lane_ends) and det.start_s < lane_ends[lane]:
            lane += 1
        if lane == len(lane_ends):
            lane_ends.append(det.end_s)
        else:
            lane_ends[lane] = det.end_s
        result.append((det, lane))
    return result
