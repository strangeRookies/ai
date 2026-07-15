"""Import-safe DTOs for the keyframe de-identification boundary."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from ai.vlm.keyframe_extractor import ExtractedKeyframe


@dataclass(frozen=True, slots=True)
class DeidentificationFrameReport:
    index: int
    status: str
    detected_person_count: int
    deidentified_person_count: int


@dataclass(frozen=True, slots=True)
class DeidentificationOutcome:
    frames: tuple[ExtractedKeyframe, ...]
    reports: tuple[DeidentificationFrameReport, ...]


DeidentifyFrames = Callable[
    [tuple[ExtractedKeyframe, ...]],
    DeidentificationOutcome,
]
