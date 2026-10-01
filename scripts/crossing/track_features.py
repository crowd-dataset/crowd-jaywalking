"""Inference safe feature extraction for one YOLO pedestrian track."""

from __future__ import annotations

from collections import defaultdict
from statistics import median
from typing import Any, Iterable

from scripts.crossing.crossing import CrossingDetector
from scripts.core.models import TrackObservation


PERSON_CLASS = 0
STATIC_REFERENCE_CLASSES = {9, 10, 11, 12, 13}
EPS = 1e-9


def _quantile(values: Iterable[float], probability: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return 0.0
    index = round((len(ordered) - 1) * probability)
    return ordered[max(0, min(index, len(ordered) - 1))]


def _robust_range(values: Iterable[float]) -> float:
    materialised = list(values)
    if not materialised:
        return 0.0
    return max(0.0, _quantile(materialised, 0.95) - _quantile(materialised, 0.05))


def _longest_state_run(states: list[str], required: str) -> int:
    longest = current = 0
    for state in states:
        if state == required:
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return longest


def person_tracks(observations: list[TrackObservation]) -> dict[int, list[TrackObservation]]:
    """Return confidence deduplicated person observations grouped by track ID."""

    grouped: dict[int, dict[int, TrackObservation]] = defaultdict(dict)
    for observation in observations:
        if observation.class_id != PERSON_CLASS:
            continue
        previous = grouped[observation.track_id].get(observation.frame_index)
        if previous is None or observation.confidence > previous.confidence:
            grouped[observation.track_id][observation.frame_index] = observation
    return {
        track_id: sorted(frames.values(), key=lambda item: item.frame_index)
        for track_id, frames in grouped.items()
    }


class TrackFeatureExtractor:
    """Create the exact feature schema used to train the frozen JAAD model."""

    def __init__(self, crossing_settings: dict[str, Any]) -> None:
        self.settings = crossing_settings
        self.detector = CrossingDetector(crossing_settings)

    def extract(
        self,
        track: list[TrackObservation],
        observations: list[TrackObservation],
        fps: float,
    ) -> dict[str, Any]:
        """Extract one benchmark shaped row from a complete tracked person."""

        if not track:
            raise ValueError("Cannot extract crossing features from an empty track")
        track = sorted(track, key=lambda item: item.frame_index)
        xs = [item.box.centre_x for item in track]
        ys = [item.box.centre_y for item in track]
        widths = [item.box.width for item in track]
        heights = [item.box.height for item in track]
        bottom_ys = [item.box.y2 for item in track]
        frame_gaps = [
            current.frame_index - previous.frame_index
            for previous, current in zip(track, track[1:])
        ]
        gross_x = sum(abs(current - previous) for previous, current in zip(xs, xs[1:]))
        gross_y = sum(abs(current - previous) for previous, current in zip(ys, ys[1:]))
        signed_x = xs[-1] - xs[0]
        signed_y = ys[-1] - ys[0]
        median_height = float(median(heights))
        edge_count = max(1, len(track) // 5)
        initial_height = float(median(heights[:edge_count]))
        final_height = float(median(heights[-edge_count:]))

        fps_value = max(float(fps), 1.0)
        max_gap = max(
            1,
            round(float(self.settings["max_track_gap_seconds"]) * fps_value),
        )
        segments: list[list[TrackObservation]] = [[track[0]]]
        for item in track[1:]:
            if item.frame_index - segments[-1][-1].frame_index > max_gap:
                segments.append([item])
            else:
                segments[-1].append(item)

        states = self.detector.track_states(track)
        min_road = max(
            1,
            round(float(self.settings["min_road_seconds"]) * fps_value),
        )
        complete_transition = any(
            self._has_complete_transition(self.detector.track_states(segment), min_road)
            for segment in segments
        )
        static = self._static_diagnostics(track, observations)
        duration_frames = track[-1].frame_index - track[0].frame_index + 1
        x_range = max(xs) - min(xs)
        y_range = max(ys) - min(ys)

        return {
            "matched_track_start_frame": track[0].frame_index,
            "matched_track_end_frame": track[-1].frame_index,
            "matched_track_frames": len(track),
            "matched_track_duration_frames": duration_frames,
            "matched_track_duration_seconds": round(duration_frames / fps_value, 6),
            "matched_track_max_gap_frames": max(frame_gaps, default=0),
            "matched_track_segment_count": len(segments),
            "matched_track_longest_segment_frames": max(len(segment) for segment in segments),
            "matched_track_x_range": round(x_range, 6),
            "matched_track_net_x_displacement": round(abs(signed_x), 6),
            "matched_track_signed_x_displacement": round(signed_x, 6),
            "matched_track_gross_x_motion": round(gross_x, 6),
            "matched_track_x_direction_consistency": round(
                abs(signed_x) / gross_x if gross_x > 0.0 else 0.0, 6
            ),
            "matched_track_x_range_over_height": round(x_range / max(median_height, EPS), 6),
            "matched_track_y_range": round(y_range, 6),
            "matched_track_net_y_displacement": round(abs(signed_y), 6),
            "matched_track_signed_y_displacement": round(signed_y, 6),
            "matched_track_gross_y_motion": round(gross_y, 6),
            "matched_track_y_direction_consistency": round(
                abs(signed_y) / gross_y if gross_y > 0.0 else 0.0, 6
            ),
            "matched_track_y_range_over_height": round(y_range / max(median_height, EPS), 6),
            "matched_track_bottom_y_range": round(max(bottom_ys) - min(bottom_ys), 6),
            "matched_track_height_change_ratio": round(
                final_height / max(initial_height, EPS), 6
            ),
            "matched_track_median_width": round(float(median(widths)), 6),
            "matched_track_median_height": round(median_height, 6),
            "matched_track_left_frames": states.count("LEFT"),
            "matched_track_road_frames": states.count("ROAD"),
            "matched_track_right_frames": states.count("RIGHT"),
            "matched_track_longest_road_run": _longest_state_run(states, "ROAD"),
            "matched_track_start_state": states[0],
            "matched_track_end_state": states[-1],
            "matched_track_complete_transition": complete_transition,
            "matched_track_static_shared_frames": static["shared_frames"],
            "matched_track_static_x_range": round(float(static["static_x_range"]), 6),
            "matched_track_relative_x_range": round(float(static["relative_x_range"]), 6),
            "matched_track_camera_motion_ratio": round(float(static["camera_motion_ratio"]), 6),
        }

    @staticmethod
    def _has_complete_transition(states: list[str], min_road: int) -> bool:
        if states.count("ROAD") < min_road:
            return False
        for index, state in enumerate(states):
            if state != "ROAD":
                continue
            before = states[:index]
            after = states[index + 1 :]
            if ("LEFT" in before and "RIGHT" in after) or (
                "RIGHT" in before and "LEFT" in after
            ):
                return True
        return False

    @staticmethod
    def _static_diagnostics(
        track: list[TrackObservation],
        observations: list[TrackObservation],
    ) -> dict[str, float | int]:
        people = {item.frame_index: item for item in track}
        references: dict[int, dict[int, TrackObservation]] = defaultdict(dict)
        for item in observations:
            if not (
                track[0].frame_index <= item.frame_index <= track[-1].frame_index
                and item.class_id in STATIC_REFERENCE_CLASSES
            ):
                continue
            previous = references[item.track_id].get(item.frame_index)
            if previous is None or item.confidence > previous.confidence:
                references[item.track_id][item.frame_index] = item

        best: dict[str, float | int] | None = None
        for reference in references.values():
            shared = sorted(set(people).intersection(reference))
            if not shared:
                continue
            person_x = [people[frame].box.centre_x for frame in shared]
            static_x = [reference[frame].box.centre_x for frame in shared]
            static_range = _robust_range(static_x)
            candidate: dict[str, float | int] = {
                "shared_frames": len(shared),
                "static_x_range": static_range,
                "relative_x_range": _robust_range(
                    person_value - static_value
                    for person_value, static_value in zip(person_x, static_x)
                ),
                "camera_motion_ratio": static_range / max(_robust_range(person_x), EPS),
            }
            if best is None or (
                int(candidate["shared_frames"]), float(candidate["static_x_range"])
            ) > (int(best["shared_frames"]), float(best["static_x_range"])):
                best = candidate
        return best or {
            "shared_frames": 0,
            "static_x_range": 0.0,
            "relative_x_range": 0.0,
            "camera_motion_ratio": 0.0,
        }
