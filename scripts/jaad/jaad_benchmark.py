"""Stage specific validation against official JAAD pedestrian annotations."""

from __future__ import annotations

import csv
from dataclasses import dataclass
import json
from pathlib import Path
from statistics import mean, median
from typing import Any

from scripts.core.config import ProjectConfig
from scripts.crossing.crossing import CrossingDetectionResult, CrossingDetector
from scripts.jaad.jaad import JAADDataset, JAADVideoAnnotations
from scripts.core.models import BoundingBox, CrossingEvent, TrackObservation
from scripts.core.tracking import PersonTracker, load_observations_csv, save_observations_csv
from scripts.crossing.track_features import TrackFeatureExtractor
from custom_logger import CustomLogger

logger = CustomLogger(__name__)  # use custom logger


PERSON_CLASS = 0
STATIC_REFERENCE_CLASSES = {9, 10, 11, 12, 13}

PER_PEDESTRIAN_FIELDS = (
    "video_id",
    "pedestrian_id",
    "ground_truth_crossing",
    "track_matched",
    "matched_track_id",
    "visible_ground_truth_frames",
    "matched_iou_frames",
    "track_coverage",
    "mean_matched_iou",
    "matched_track_start_frame",
    "matched_track_end_frame",
    "matched_track_frames",
    "matched_track_duration_frames",
    "matched_track_duration_seconds",
    "matched_track_max_gap_frames",
    "matched_track_segment_count",
    "matched_track_longest_segment_frames",
    "matched_track_x_range",
    "matched_track_net_x_displacement",
    "matched_track_signed_x_displacement",
    "matched_track_gross_x_motion",
    "matched_track_x_direction_consistency",
    "matched_track_x_range_over_height",
    "matched_track_y_range",
    "matched_track_net_y_displacement",
    "matched_track_signed_y_displacement",
    "matched_track_gross_y_motion",
    "matched_track_y_direction_consistency",
    "matched_track_y_range_over_height",
    "matched_track_bottom_y_range",
    "matched_track_height_change_ratio",
    "matched_track_median_width",
    "matched_track_median_height",
    "matched_track_left_frames",
    "matched_track_road_frames",
    "matched_track_right_frames",
    "matched_track_longest_road_run",
    "matched_track_start_state",
    "matched_track_end_state",
    "matched_track_complete_transition",
    "matched_track_static_shared_frames",
    "matched_track_static_x_range",
    "matched_track_relative_x_range",
    "matched_track_camera_motion_ratio",
    "predicted_crossing",
    "candidate_outcome",
    "candidate_track_frames",
    "candidate_road_frames",
    "candidate_x_range",
    "candidate_x_speed_per_frame",
    "candidate_y_gross_motion",
    "candidate_median_width",
    "candidate_median_height",
    "candidate_static_shared_frames",
    "candidate_static_x_range",
    "candidate_relative_x_range",
    "candidate_camera_motion_ratio",
    "ground_truth_start_frame",
    "ground_truth_end_frame",
    "predicted_transition_start_frame",
    "predicted_transition_end_frame",
    "transition_temporal_iou",
    "correct",
)

UNMATCHED_PREDICTION_FIELDS = (
    "video_id",
    "track_id",
    "start_frame",
    "end_frame",
    "transition_start_frame",
    "transition_end_frame",
    "outcome",
)


@dataclass(frozen=True)
class TrackMatch:
    """One independent JAAD pedestrian to YOLO track association."""

    pedestrian_id: str
    track_id: int
    visible_ground_truth_frames: int
    matched_iou_frames: int
    coverage: float
    mean_matched_iou: float


def box_iou(first: BoundingBox, second: BoundingBox) -> float:
    """Calculate intersection over union for two normalised boxes."""

    intersection_width = max(0.0, min(first.x2, second.x2) - max(first.x1, second.x1))
    intersection_height = max(0.0, min(first.y2, second.y2) - max(first.y1, second.y1))
    intersection = intersection_width * intersection_height
    first_area = first.width * first.height
    second_area = second.width * second.height
    union = first_area + second_area - intersection
    return intersection / union if union > 0.0 else 0.0


def _nearest_quantile(values: list[float], probability: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    index = round((len(ordered) - 1) * probability)
    return ordered[max(0, min(index, len(ordered) - 1))]


def _robust_range(values: list[float]) -> float:
    if not values:
        return 0.0
    return max(
        0.0,
        _nearest_quantile(values, 0.95) - _nearest_quantile(values, 0.05),
    )


def _longest_run(states: list[str], required_state: str) -> int:
    longest = current = 0
    for state in states:
        if state == required_state:
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return longest


def _person_tracks(
    observations: list[TrackObservation],
) -> dict[int, dict[int, TrackObservation]]:
    tracks: dict[int, dict[int, TrackObservation]] = {}
    for observation in observations:
        if observation.class_id != PERSON_CLASS:
            continue
        track = tracks.setdefault(observation.track_id, {})
        previous = track.get(observation.frame_index)
        if previous is None or observation.confidence > previous.confidence:
            track[observation.frame_index] = observation
    return tracks


def match_person_tracks(
    annotations: JAADVideoAnnotations,
    observations: list[TrackObservation],
    iou_threshold: float,
    min_match_frames: int,
    min_track_coverage: float,
) -> tuple[dict[str, TrackMatch], set[int]]:
    """Greedily create one to one matches using frame aligned box IoU."""

    predicted = _person_tracks(observations)
    candidates: list[tuple[int, float, float, str, int, int]] = []

    for ground_truth in annotations.behaviour_tracks:
        visible_frames = ground_truth.visible_frames
        if not visible_frames:
            continue
        for track_id, predicted_by_frame in predicted.items():
            matched_ious: list[float] = []
            for frame in visible_frames:
                if frame not in predicted_by_frame:
                    continue
                overlap = box_iou(ground_truth.boxes[frame], predicted_by_frame[frame].box)
                if overlap >= float(iou_threshold):
                    matched_ious.append(overlap)
            matched_frames = len(matched_ious)
            coverage = matched_frames / len(visible_frames)
            if matched_frames < int(min_match_frames) or coverage < float(min_track_coverage):
                continue
            mean_iou = mean(matched_ious) if matched_ious else 0.0
            candidates.append(
                (
                    matched_frames,
                    coverage,
                    mean_iou,
                    ground_truth.pedestrian_id,
                    track_id,
                    len(visible_frames),
                )
            )

    candidates.sort(reverse=True)
    matches: dict[str, TrackMatch] = {}
    used_track_ids: set[int] = set()
    for matched_frames, coverage, mean_iou, pedestrian_id, track_id, visible_frames in candidates:
        if pedestrian_id in matches or track_id in used_track_ids:
            continue
        matches[pedestrian_id] = TrackMatch(
            pedestrian_id=pedestrian_id,
            track_id=track_id,
            visible_ground_truth_frames=visible_frames,
            matched_iou_frames=matched_frames,
            coverage=coverage,
            mean_matched_iou=mean_iou,
        )
        used_track_ids.add(track_id)

    return matches, set(predicted).difference(used_track_ids)


def _event_by_track(events: list[CrossingEvent]) -> dict[int, list[CrossingEvent]]:
    result: dict[int, list[CrossingEvent]] = {}
    for event in events:
        result.setdefault(event.person_id, []).append(event)
    return result


def _temporal_iou(ground_truth_frames: tuple[int, ...], event: CrossingEvent | None) -> float:
    if not ground_truth_frames or event is None:
        return 0.0
    ground_truth = set(ground_truth_frames)
    predicted = set(range(event.transition_start_frame, event.transition_end_frame + 1))
    union = ground_truth.union(predicted)
    return len(ground_truth.intersection(predicted)) / len(union) if union else 0.0


def _best_event(
    events: list[CrossingEvent],
    ground_truth_frames: tuple[int, ...],
) -> CrossingEvent | None:
    if not events:
        return None
    if not ground_truth_frames:
        return min(events, key=lambda item: (item.transition_start_frame, item.person_id))
    return max(events, key=lambda item: _temporal_iou(ground_truth_frames, item))


def _classification_metrics(
    rows: list[dict[str, Any]],
) -> dict[str, float | int | None]:
    tp = sum(row["ground_truth_crossing"] and row["predicted_crossing"] for row in rows)
    tn = sum(not row["ground_truth_crossing"] and not row["predicted_crossing"] for row in rows)
    fp = sum(not row["ground_truth_crossing"] and row["predicted_crossing"] for row in rows)
    fn = sum(row["ground_truth_crossing"] and not row["predicted_crossing"] for row in rows)

    def percentage(numerator: float, denominator: float) -> float | None:
        return 100.0 * numerator / denominator if denominator else None

    precision = percentage(tp, tp + fp)
    recall = percentage(tp, tp + fn)
    specificity = percentage(tn, tn + fp)
    f1 = percentage(2 * tp, 2 * tp + fp + fn)
    balanced = (
        (recall + specificity) / 2.0
        if recall is not None and specificity is not None
        else None
    )
    return {
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "accuracy_percent": percentage(tp + tn, len(rows)),
        "precision_percent": precision,
        "recall_percent": recall,
        "specificity_percent": specificity,
        "balanced_accuracy_percent": balanced,
        "f1_percent": f1,
    }


class JAADCrossingBenchmark:
    """Run tracking and crossing validation without loading the VLM."""

    def __init__(self, config: ProjectConfig) -> None:
        self.config = config
        self.dataset = JAADDataset(config.path("jaad_root"))
        self.split = str(config.get("jaad_benchmark_split")).strip().lower()
        self.output_dir = config.path("jaad_benchmark_results") / self.split
        self.tracks_dir = self.output_dir / "tracks"
        self.per_pedestrian_csv = self.output_dir / "per_pedestrian.csv"
        self.partial_candidates_csv = self.output_dir / "partial_candidate_features.csv"
        self.unmatched_csv = self.output_dir / "unmatched_crossing_predictions.csv"
        self.summary_json = self.output_dir / "summary.json"
        self.detector = CrossingDetector(config.crossing_settings())
        self.feature_extractor = TrackFeatureExtractor(config.crossing_settings())
        self._tracker: PersonTracker | None = None

    @property
    def tracker(self) -> PersonTracker:
        if self._tracker is None:
            self._tracker = PersonTracker(self.config.tracking_settings(), self.config.root)
        return self._tracker

    def run(self) -> dict[str, Any]:
        video_ids = self.dataset.video_ids(self.split)
        selected_video = str(self.config.get("jaad_video_id") or "").strip()
        if selected_video:
            if selected_video not in video_ids:
                raise ValueError(
                    f"{selected_video} is not in the official JAAD {self.split} split"
                )
            video_ids = [selected_video]
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.tracks_dir.mkdir(parents=True, exist_ok=True)

        rows: list[dict[str, Any]] = []
        unmatched_rows: list[dict[str, Any]] = []
        for index, video_id in enumerate(video_ids, start=1):
            logger.info("[{:03d}/{:03d}] {}", index, len(video_ids), video_id)
            annotations = self.dataset.load_video(video_id)
            video_path = self.dataset.clip_path(video_id)
            fps, observations = self._observations(video_id, video_path)
            detections = self.detector.detect(observations, fps)
            video_rows, video_unmatched = self._evaluate_video(
                annotations,
                observations,
                detections,
                fps,
            )
            rows.extend(video_rows)
            unmatched_rows.extend(video_unmatched)

        self._write_csv(self.per_pedestrian_csv, PER_PEDESTRIAN_FIELDS, rows)
        partial_rows = [
            row
            for row in rows
            if row["track_matched"]
            and row["candidate_outcome"] == "NO_CROSSING_CANDIDATE"
        ]
        self._write_csv(
            self.partial_candidates_csv,
            PER_PEDESTRIAN_FIELDS,
            partial_rows,
        )
        self._write_csv(self.unmatched_csv, UNMATCHED_PREDICTION_FIELDS, unmatched_rows)
        summary = self._summarise(video_ids, rows, unmatched_rows)
        with self.summary_json.open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2)
        self._print_summary(summary)
        return summary

    def _observations(self, video_id: str, video_path: Path) -> tuple[float, list[TrackObservation]]:
        import cv2

        tracking_csv = self.tracks_dir / f"{video_id}.csv"
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise RuntimeError(f"Could not open JAAD video: {video_path}")
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 30.0)
        capture.release()

        if tracking_csv.is_file():
            return fps, load_observations_csv(tracking_csv)
        tracked_fps, observations = self.tracker.track(video_path)
        save_observations_csv(tracking_csv, observations)
        return tracked_fps, observations

    def _evaluate_video(
        self,
        annotations: JAADVideoAnnotations,
        observations: list[TrackObservation],
        detections: CrossingDetectionResult,
        fps: float,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        matches, unmatched_track_ids = match_person_tracks(
            annotations=annotations,
            observations=observations,
            iou_threshold=float(self.config.get("jaad_match_iou")),
            min_match_frames=int(self.config.get("jaad_min_match_frames")),
            min_track_coverage=float(self.config.get("jaad_min_track_coverage")),
        )
        valid = _event_by_track(detections.valid_events)
        rejected = _event_by_track(detections.rejected_events)
        predicted_tracks = _person_tracks(observations)

        rows: list[dict[str, Any]] = []
        for ground_truth in annotations.behaviour_tracks:
            match = matches.get(ground_truth.pedestrian_id)
            valid_events = valid.get(match.track_id, []) if match else []
            rejected_events = rejected.get(match.track_id, []) if match else []
            event = _best_event(valid_events, ground_truth.crossing_frames)
            if event is None:
                event = _best_event(rejected_events, ground_truth.crossing_frames)
            predicted_crossing = bool(valid_events)
            intervals = ground_truth.crossing_intervals()
            ground_truth_start = intervals[0][0] if intervals else ""
            ground_truth_end = intervals[-1][1] if intervals else ""
            matched_track: list[TrackObservation] = []
            if match:
                matched_track = sorted(
                    predicted_tracks.get(match.track_id, {}).values(),
                    key=lambda item: item.frame_index,
                )
                if ground_truth.visible_frames:
                    first_visible = min(ground_truth.visible_frames)
                    last_visible = max(ground_truth.visible_frames)
                    matched_track = [
                        item
                        for item in matched_track
                        if first_visible <= item.frame_index <= last_visible
                    ]
            track_diagnostics = self._partial_track_diagnostics(
                matched_track,
                observations,
                fps,
            )
            candidate_diagnostics = self._candidate_diagnostics(event)

            if valid_events:
                outcome = "ACCEPTED"
            elif rejected_events:
                outcome = event.rejection_reason.value if event else "REJECTED"
            elif match:
                outcome = "NO_CROSSING_CANDIDATE"
            else:
                outcome = "NO_MATCHED_TRACK"

            rows.append(
                {
                    "video_id": annotations.video_id,
                    "pedestrian_id": ground_truth.pedestrian_id,
                    "ground_truth_crossing": ground_truth.is_crossing,
                    "track_matched": match is not None,
                    "matched_track_id": match.track_id if match else "",
                    "visible_ground_truth_frames": len(ground_truth.visible_frames),
                    "matched_iou_frames": match.matched_iou_frames if match else 0,
                    "track_coverage": round(match.coverage, 6) if match else 0.0,
                    "mean_matched_iou": round(match.mean_matched_iou, 6) if match else 0.0,
                    **track_diagnostics,
                    "predicted_crossing": predicted_crossing,
                    "candidate_outcome": outcome,
                    **candidate_diagnostics,
                    "ground_truth_start_frame": ground_truth_start,
                    "ground_truth_end_frame": ground_truth_end,
                    "predicted_transition_start_frame": event.transition_start_frame if event else "",
                    "predicted_transition_end_frame": event.transition_end_frame if event else "",
                    "transition_temporal_iou": round(
                        _temporal_iou(ground_truth.crossing_frames, event),
                        6,
                    ),
                    "correct": ground_truth.is_crossing == predicted_crossing,
                }
            )

        unmatched_rows: list[dict[str, Any]] = []
        for event in detections.valid_events:
            if event.person_id not in unmatched_track_ids:
                continue
            unmatched_rows.append(
                {
                    "video_id": annotations.video_id,
                    "track_id": event.person_id,
                    "start_frame": event.start_frame,
                    "end_frame": event.end_frame,
                    "transition_start_frame": event.transition_start_frame,
                    "transition_end_frame": event.transition_end_frame,
                    "outcome": "REQUIRES_MANUAL_REVIEW",
                }
            )
        return rows, unmatched_rows

    @staticmethod
    def _candidate_diagnostics(event: CrossingEvent | None) -> dict[str, Any]:
        if event is None:
            return {
                "candidate_track_frames": 0,
                "candidate_road_frames": 0,
                "candidate_x_range": "",
                "candidate_x_speed_per_frame": "",
                "candidate_y_gross_motion": "",
                "candidate_median_width": "",
                "candidate_median_height": "",
                "candidate_static_shared_frames": 0,
                "candidate_static_x_range": "",
                "candidate_relative_x_range": "",
                "candidate_camera_motion_ratio": "",
            }
        features = event.features
        return {
            "candidate_track_frames": features.track_frames,
            "candidate_road_frames": features.road_frames,
            "candidate_x_range": round(features.x_range, 6),
            "candidate_x_speed_per_frame": round(features.x_speed_per_frame, 8),
            "candidate_y_gross_motion": round(features.y_gross_motion, 6),
            "candidate_median_width": round(features.median_width, 6),
            "candidate_median_height": round(features.median_height, 6),
            "candidate_static_shared_frames": features.static_shared_frames,
            "candidate_static_x_range": round(features.static_x_range, 6),
            "candidate_relative_x_range": round(features.relative_x_range, 6),
            "candidate_camera_motion_ratio": round(features.camera_motion_ratio, 6),
        }

    @staticmethod
    def _track_diagnostics(track: list[TrackObservation]) -> dict[str, Any]:
        """Retain the version 1.2 diagnostic API used by existing tests."""

        if not track:
            return {
                "matched_track_start_frame": "",
                "matched_track_end_frame": "",
                "matched_track_frames": 0,
                "matched_track_x_range": "",
                "matched_track_net_x_displacement": "",
                "matched_track_y_range": "",
                "matched_track_median_width": "",
                "matched_track_median_height": "",
            }
        xs = [item.box.centre_x for item in track]
        ys = [item.box.centre_y for item in track]
        widths = [item.box.width for item in track]
        heights = [item.box.height for item in track]
        return {
            "matched_track_start_frame": track[0].frame_index,
            "matched_track_end_frame": track[-1].frame_index,
            "matched_track_frames": len(track),
            "matched_track_x_range": round(max(xs) - min(xs), 6),
            "matched_track_net_x_displacement": round(abs(xs[-1] - xs[0]), 6),
            "matched_track_y_range": round(max(ys) - min(ys), 6),
            "matched_track_median_width": round(float(median(widths)), 6),
            "matched_track_median_height": round(float(median(heights)), 6),
        }

    def _partial_track_diagnostics(
        self,
        track: list[TrackObservation],
        observations: list[TrackObservation],
        fps: float,
    ) -> dict[str, Any]:
        if track:
            return self.feature_extractor.extract(track, observations, fps)
        names = (
            "matched_track_start_frame",
            "matched_track_end_frame",
            "matched_track_frames",
            "matched_track_duration_frames",
            "matched_track_duration_seconds",
            "matched_track_max_gap_frames",
            "matched_track_segment_count",
            "matched_track_longest_segment_frames",
            "matched_track_x_range",
            "matched_track_net_x_displacement",
            "matched_track_signed_x_displacement",
            "matched_track_gross_x_motion",
            "matched_track_x_direction_consistency",
            "matched_track_x_range_over_height",
            "matched_track_y_range",
            "matched_track_net_y_displacement",
            "matched_track_signed_y_displacement",
            "matched_track_gross_y_motion",
            "matched_track_y_direction_consistency",
            "matched_track_y_range_over_height",
            "matched_track_bottom_y_range",
            "matched_track_height_change_ratio",
            "matched_track_median_width",
            "matched_track_median_height",
            "matched_track_left_frames",
            "matched_track_road_frames",
            "matched_track_right_frames",
            "matched_track_longest_road_run",
            "matched_track_start_state",
            "matched_track_end_state",
            "matched_track_complete_transition",
            "matched_track_static_shared_frames",
            "matched_track_static_x_range",
            "matched_track_relative_x_range",
            "matched_track_camera_motion_ratio",
        )
        if not track:
            empty = {name: "" for name in names}
            for name in (
                "matched_track_frames",
                "matched_track_duration_frames",
                "matched_track_max_gap_frames",
                "matched_track_segment_count",
                "matched_track_longest_segment_frames",
                "matched_track_left_frames",
                "matched_track_road_frames",
                "matched_track_right_frames",
                "matched_track_longest_road_run",
                "matched_track_static_shared_frames",
            ):
                empty[name] = 0
            empty["matched_track_complete_transition"] = False
            return empty

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

        max_gap = max(
            1,
            round(
                float(self.config.get("max_track_gap_seconds"))
                * max(float(fps), 1.0)
            ),
        )
        segments: list[list[TrackObservation]] = [[track[0]]]
        for item in track[1:]:
            if item.frame_index - segments[-1][-1].frame_index > max_gap:
                segments.append([item])
            else:
                segments[-1].append(item)

        states = self._corridor_states(track)
        min_road = max(
            1,
            round(
                float(self.config.get("min_road_seconds"))
                * max(float(fps), 1.0)
            ),
        )
        complete_transition = any(
            self._has_complete_transition(self._corridor_states(segment), min_road)
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
            "matched_track_duration_seconds": round(duration_frames / max(fps, 1.0), 6),
            "matched_track_max_gap_frames": max(frame_gaps, default=0),
            "matched_track_segment_count": len(segments),
            "matched_track_longest_segment_frames": max(len(segment) for segment in segments),
            "matched_track_x_range": round(x_range, 6),
            "matched_track_net_x_displacement": round(abs(signed_x), 6),
            "matched_track_signed_x_displacement": round(signed_x, 6),
            "matched_track_gross_x_motion": round(gross_x, 6),
            "matched_track_x_direction_consistency": round(
                abs(signed_x) / gross_x if gross_x > 0.0 else 0.0,
                6,
            ),
            "matched_track_x_range_over_height": round(
                x_range / max(median_height, 1e-9),
                6,
            ),
            "matched_track_y_range": round(y_range, 6),
            "matched_track_net_y_displacement": round(abs(signed_y), 6),
            "matched_track_signed_y_displacement": round(signed_y, 6),
            "matched_track_gross_y_motion": round(gross_y, 6),
            "matched_track_y_direction_consistency": round(
                abs(signed_y) / gross_y if gross_y > 0.0 else 0.0,
                6,
            ),
            "matched_track_y_range_over_height": round(
                y_range / max(median_height, 1e-9),
                6,
            ),
            "matched_track_bottom_y_range": round(max(bottom_ys) - min(bottom_ys), 6),
            "matched_track_height_change_ratio": round(
                final_height / max(initial_height, 1e-9),
                6,
            ),
            "matched_track_median_width": round(float(median(widths)), 6),
            "matched_track_median_height": round(median_height, 6),
            "matched_track_left_frames": states.count("LEFT"),
            "matched_track_road_frames": states.count("ROAD"),
            "matched_track_right_frames": states.count("RIGHT"),
            "matched_track_longest_road_run": _longest_run(states, "ROAD"),
            "matched_track_start_state": states[0],
            "matched_track_end_state": states[-1],
            "matched_track_complete_transition": complete_transition,
            "matched_track_static_shared_frames": static["shared_frames"],
            "matched_track_static_x_range": round(static["static_x_range"], 6),
            "matched_track_relative_x_range": round(static["relative_x_range"], 6),
            "matched_track_camera_motion_ratio": round(static["camera_motion_ratio"], 6),
        }

    def _corridor_states(self, track: list[TrackObservation]) -> list[str]:
        return self.detector.track_states(track)

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
        references: dict[int, dict[int, TrackObservation]] = {}
        for item in observations:
            if (
                track[0].frame_index <= item.frame_index <= track[-1].frame_index
                and item.class_id in STATIC_REFERENCE_CLASSES
            ):
                reference = references.setdefault(item.track_id, {})
                previous = reference.get(item.frame_index)
                if previous is None or item.confidence > previous.confidence:
                    reference[item.frame_index] = item

        best: dict[str, float | int] | None = None
        for reference in references.values():
            shared = sorted(set(people).intersection(reference))
            if not shared:
                continue
            person_x = [people[frame].box.centre_x for frame in shared]
            static_x = [reference[frame].box.centre_x for frame in shared]
            person_range = _robust_range(person_x)
            static_range = _robust_range(static_x)
            candidate: dict[str, float | int] = {
                "shared_frames": len(shared),
                "static_x_range": static_range,
                "relative_x_range": _robust_range(
                    [
                        person_value - static_value
                        for person_value, static_value in zip(person_x, static_x)
                    ]
                ),
                "camera_motion_ratio": static_range / max(person_range, 1e-9),
            }
            if best is None or (
                int(candidate["shared_frames"]),
                float(candidate["static_x_range"]),
            ) > (
                int(best["shared_frames"]),
                float(best["static_x_range"]),
            ):
                best = candidate

        return best or {
            "shared_frames": 0,
            "static_x_range": 0.0,
            "relative_x_range": 0.0,
            "camera_motion_ratio": 0.0,
        }

    def _summarise(
        self,
        video_ids: list[str],
        rows: list[dict[str, Any]],
        unmatched_rows: list[dict[str, Any]],
    ) -> dict[str, Any]:
        matched_rows = [row for row in rows if row["track_matched"]]
        true_positives = [
            row
            for row in rows
            if row["ground_truth_crossing"] and row["predicted_crossing"]
        ]
        false_negative_outcomes: dict[str, int] = {}
        for row in rows:
            if not row["ground_truth_crossing"] or row["predicted_crossing"]:
                continue
            outcome = str(row["candidate_outcome"])
            false_negative_outcomes[outcome] = false_negative_outcomes.get(outcome, 0) + 1
        return {
            "split": self.split,
            "videos": len(video_ids),
            "behaviour_annotated_pedestrians": len(rows),
            "ground_truth_crossing_pedestrians": sum(
                bool(row["ground_truth_crossing"]) for row in rows
            ),
            "matched_pedestrian_tracks": len(matched_rows),
            "track_match_recall_percent": 100.0 * len(matched_rows) / len(rows) if rows else 0.0,
            "mean_track_coverage_percent": (
                100.0 * mean(float(row["track_coverage"]) for row in matched_rows)
                if matched_rows
                else 0.0
            ),
            "mean_matched_iou": (
                mean(float(row["mean_matched_iou"]) for row in matched_rows)
                if matched_rows
                else 0.0
            ),
            "end_to_end_crossing_metrics": _classification_metrics(rows),
            "crossing_metrics_when_track_matched": _classification_metrics(matched_rows),
            "mean_transition_temporal_iou_on_true_positives": (
                mean(float(row["transition_temporal_iou"]) for row in true_positives)
                if true_positives
                else 0.0
            ),
            "unmatched_accepted_crossings_requiring_review": len(unmatched_rows),
            "partial_candidate_diagnostic_rows": sum(
                bool(row["track_matched"])
                and row["candidate_outcome"] == "NO_CROSSING_CANDIDATE"
                for row in rows
            ),
            "partial_candidate_features_csv": str(self.partial_candidates_csv),
            "false_negative_outcomes": dict(
                sorted(
                    false_negative_outcomes.items(),
                    key=lambda item: (-item[1], item[0]),
                )
            ),
            "configuration": {
                "match_iou": self.config.get("jaad_match_iou"),
                "min_match_frames": self.config.get("jaad_min_match_frames"),
                "min_track_coverage": self.config.get("jaad_min_track_coverage"),
                "crossing": self.config.crossing_settings(),
            },
        }

    @staticmethod
    def _write_csv(
        path: Path,
        fields: tuple[str, ...],
        rows: list[dict[str, Any]],
    ) -> None:
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)

    @staticmethod
    def _print_summary(summary: dict[str, Any]) -> None:
        metrics = summary["end_to_end_crossing_metrics"]

        def display(value: float | int | None) -> str:
            return "N/A" if value is None else f"{float(value):.2f}%"

        logger.info("\nJAAD crossing benchmark")
        logger.info("Split: {}", summary['split'])
        logger.info("Videos: {}", summary['videos'])
        logger.info("Annotated pedestrians: {}", summary['behaviour_annotated_pedestrians'])
        logger.info("Track match recall: {:.2f}%", summary['track_match_recall_percent'])
        logger.info(
            "Confusion matrix: TP={} TN={} FP={} FN={}",
            metrics['tp'],
            metrics['tn'],
            metrics['fp'],
            metrics['fn'],
        )
        logger.info("Crossing accuracy: {}", display(metrics['accuracy_percent']))
        logger.info("Crossing precision: {}", display(metrics['precision_percent']))
        logger.info("Crossing recall: {}", display(metrics['recall_percent']))
        logger.info("Crossing F1: {}", display(metrics['f1_percent']))
        logger.info("Balanced accuracy: {}", display(metrics['balanced_accuracy_percent']))
        logger.info("Partial candidate features: {}", summary['partial_candidate_features_csv'])
