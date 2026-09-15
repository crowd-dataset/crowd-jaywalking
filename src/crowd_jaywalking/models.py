"""Shared immutable data contracts for the jaywalking pipeline."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any


class DecisionLabel(str, Enum):
    """Final classification labels."""

    JAYWALKING = "JAYWALKING"
    COMPLIANT = "COMPLIANT"
    UNCERTAIN = "UNCERTAIN"


class Ternary(str, Enum):
    """Three state answer for observable context."""

    YES = "YES"
    NO = "NO"
    UNCERTAIN = "UNCERTAIN"


class Visibility(str, Enum):
    """Quality of the visual context supplied to the VLM."""

    CLEAR = "CLEAR"
    PARTIAL = "PARTIAL"
    INSUFFICIENT = "INSUFFICIENT"


class RejectionReason(str, Enum):
    """Reasons for rejecting an apparent crossing."""

    NONE = "NONE"
    INSUFFICIENT_TRACK = "INSUFFICIENT_TRACK"
    INSUFFICIENT_ROAD_CONTACT = "INSUFFICIENT_ROAD_CONTACT"
    NO_COMPLETE_TRANSITION = "NO_COMPLETE_TRANSITION"
    INSUFFICIENT_LATERAL_MOTION = "INSUFFICIENT_LATERAL_MOTION"
    LONG_WEAK_TRACK = "LONG_WEAK_TRACK"
    VERTICAL_JITTER = "VERTICAL_JITTER"
    TINY_UNVERIFIED_TRACK = "TINY_UNVERIFIED_TRACK"
    CAMERA_MOTION = "CAMERA_MOTION"
    RIDER = "RIDER"
    CLASSIFIER_NEGATIVE = "CLASSIFIER_NEGATIVE"


@dataclass(frozen=True)
class BoundingBox:
    """Normalised XYXY bounding box."""

    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def width(self) -> float:
        return max(0.0, self.x2 - self.x1)

    @property
    def height(self) -> float:
        return max(0.0, self.y2 - self.y1)

    @property
    def centre_x(self) -> float:
        return (self.x1 + self.x2) / 2.0

    @property
    def centre_y(self) -> float:
        return (self.y1 + self.y2) / 2.0


@dataclass(frozen=True)
class TrackObservation:
    """One tracked object observation in one frame."""

    frame_index: int
    track_id: int
    class_id: int
    confidence: float
    box: BoundingBox


@dataclass(frozen=True)
class CrossingFeatures:
    """Features used to accept or reject a crossing candidate."""

    track_frames: int
    road_frames: int
    x_range: float
    x_speed_per_frame: float
    y_gross_motion: float
    median_width: float
    median_height: float
    static_shared_frames: int
    static_x_range: float
    relative_x_range: float
    camera_motion_ratio: float


@dataclass(frozen=True)
class CrossingEvent:
    """A person-specific crossing event or rejected candidate."""

    person_id: int
    # Complete continuous track segment used by the geometric filters.
    start_frame: int
    end_frame: int
    # Narrow side to road to opposite side interval used to centre VLM evidence.
    transition_start_frame: int
    transition_end_frame: int
    valid: bool
    rejection_reason: RejectionReason
    features: CrossingFeatures


@dataclass(frozen=True)
class EvidenceImage:
    """One visual input generated for a crossing event."""

    frame_index: int
    context_path: Path
    focus_path: Path
    road_path: Path | None = None
    trajectory_path: Path | None = None
    control_left_path: Path | None = None
    control_right_path: Path | None = None


@dataclass(frozen=True)
class ContextAssessment:
    """Structured observable scene context returned by the VLM."""

    marked_crosswalk: Ternary
    permissive_pedestrian_signal: Ternary
    authorised_crossing_sign: Ternary
    crossing_guard_permission: Ternary
    prohibitive_pedestrian_signal: Ternary
    visibility: Visibility
    evidence_summary: str


@dataclass(frozen=True)
class PersonDecision:
    """Final decision for one valid crossing person."""

    person_id: int
    label: DecisionLabel
    reason: str
    event: CrossingEvent
    context: ContextAssessment


@dataclass(frozen=True)
class CrossingClassification:
    """Frozen model score and audit information for one person track."""

    person_id: int
    probability: float
    threshold: float
    predicted_crossing: bool
    rule_outcome: str
    event: CrossingEvent
    track_features: dict[str, Any]


@dataclass(frozen=True)
class VideoResult:
    """Complete result for one input video."""

    video_path: str
    prediction: DecisionLabel
    person_decisions: list[PersonDecision]
    rejected_candidates: list[CrossingEvent]
    latency_seconds: float
    crossing_classifications: list[CrossingClassification] = field(default_factory=list)


def to_jsonable(value: Any) -> Any:
    """Convert nested pipeline values into JSON serialisable values."""

    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "__dataclass_fields__"):
        return {key: to_jsonable(item) for key, item in asdict(value).items()}
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    return value
