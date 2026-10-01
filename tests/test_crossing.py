"""Synthetic tests for crossing detection and false crossing filters."""

import unittest

from scripts.crossing.crossing import CrossingDetector
from config_helpers import load_config
from scripts.core.models import BoundingBox, RejectionReason, TrackObservation


def settings() -> dict:
    return load_config().crossing_settings()


def observation(
    frame: int,
    x: float,
    *,
    track_id: int = 1,
    class_id: int = 0,
    width: float = 0.08,
    height: float = 0.20,
    y: float = 0.50,
) -> TrackObservation:
    return TrackObservation(
        frame_index=frame,
        track_id=track_id,
        class_id=class_id,
        confidence=0.95,
        box=BoundingBox(
            x1=x - width / 2,
            y1=y - height / 2,
            x2=x + width / 2,
            y2=y + height / 2,
        ),
    )


def crossing_track() -> list[TrackObservation]:
    return [
        observation(0, 0.30),
        observation(1, 0.46),
        observation(2, 0.50),
        observation(3, 0.54),
        observation(4, 0.70),
    ]


class CrossingDetectorTests(unittest.TestCase):
    def test_accepts_complete_crossing(self) -> None:
        result = CrossingDetector(settings()).detect(crossing_track(), fps=10.0)
        self.assertEqual(len(result.valid_events), 1)
        self.assertEqual(result.valid_events[0].person_id, 1)
        self.assertEqual(result.valid_events[0].rejection_reason, RejectionReason.NONE)

    def test_does_not_count_off_centre_motion_as_a_crossing(self) -> None:
        track = [
            observation(frame, x)
            for frame, x in enumerate([0.05, 0.10, 0.16, 0.23, 0.31])
        ]

        result = CrossingDetector(settings()).detect(track, fps=10.0)

        self.assertFalse(result.valid_events)
        self.assertFalse(result.rejected_events)

    def test_validates_full_segment_not_narrow_transition_slice(self) -> None:
        track = [
            observation(frame, x)
            for frame, x in enumerate([0.20, 0.35, 0.44, 0.46, 0.50, 0.54, 0.56, 0.65, 0.80])
        ]

        result = CrossingDetector(settings()).detect(track, fps=10.0)

        self.assertEqual(len(result.valid_events), 1)
        event = result.valid_events[0]
        self.assertAlmostEqual(event.features.x_range, 0.60)
        self.assertEqual((event.start_frame, event.end_frame), (0, 8))
        self.assertEqual((event.transition_start_frame, event.transition_end_frame), (1, 7))

    def test_rejects_insufficient_lateral_motion(self) -> None:
        custom = settings()
        custom["perspective_corridor_enabled"] = False
        track = [
            observation(0, 0.44),
            observation(1, 0.47),
            observation(2, 0.50),
            observation(3, 0.53),
            observation(4, 0.56),
        ]
        result = CrossingDetector(custom).detect(track, fps=10.0)
        self.assertFalse(result.valid_events)
        self.assertEqual(
            result.rejected_events[0].rejection_reason,
            RejectionReason.INSUFFICIENT_LATERAL_MOTION,
        )

    def test_stationary_track_is_not_a_crossing_candidate(self) -> None:
        track = [observation(frame, 0.20) for frame in range(5)]

        result = CrossingDetector(settings()).detect(track, fps=10.0)

        self.assertFalse(result.valid_events)
        self.assertFalse(result.rejected_events)

    def test_accepts_partial_side_to_road_entry(self) -> None:
        track = [
            observation(frame, x)
            for frame, x in enumerate([0.30, 0.40, 0.46, 0.50, 0.53])
        ]

        result = CrossingDetector(settings()).detect(track, fps=10.0)

        self.assertEqual(len(result.valid_events), 1)
        self.assertFalse(result.rejected_events)
        self.assertEqual(
            (
                result.valid_events[0].transition_start_frame,
                result.valid_events[0].transition_end_frame,
            ),
            (0, 4),
        )

    def test_partial_entry_requires_sustained_road_contact(self) -> None:
        custom = settings()
        custom["min_track_seconds"] = 0.10
        custom["min_road_seconds"] = 0.30
        track = [
            observation(frame, x)
            for frame, x in enumerate([0.20, 0.30, 0.40, 0.46])
        ]

        result = CrossingDetector(custom).detect(track, fps=10.0)

        self.assertFalse(result.valid_events)
        self.assertFalse(result.rejected_events)

    def test_accepts_partial_road_to_side_exit_at_tuned_range(self) -> None:
        track = [
            observation(frame, x)
            for frame, x in enumerate([0.50, 0.47, 0.40, 0.25, 0.10])
        ]

        result = CrossingDetector(settings()).detect(track, fps=10.0)

        self.assertEqual(len(result.valid_events), 1)
        self.assertFalse(result.rejected_events)

    def test_rejects_partial_exit_below_tuned_range(self) -> None:
        track = [
            observation(frame, x)
            for frame, x in enumerate([0.50, 0.47, 0.40, 0.25, 0.15])
        ]

        result = CrossingDetector(settings()).detect(track, fps=10.0)

        self.assertFalse(result.valid_events)
        self.assertFalse(result.rejected_events)

    def test_rejects_directionally_inconsistent_partial_exit(self) -> None:
        track = [
            observation(frame, x)
            for frame, x in enumerate([0.50, 0.42, 0.10, 0.40, 0.10])
        ]

        result = CrossingDetector(settings()).detect(track, fps=10.0)

        self.assertFalse(result.valid_events)
        self.assertFalse(result.rejected_events)

    def test_partial_crossings_can_be_disabled(self) -> None:
        custom = settings()
        custom["partial_crossing_enabled"] = False
        track = [
            observation(frame, x)
            for frame, x in enumerate([0.30, 0.40, 0.46, 0.50, 0.53])
        ]

        result = CrossingDetector(custom).detect(track, fps=10.0)

        self.assertFalse(result.valid_events)
        self.assertFalse(result.rejected_events)

    def test_partial_crossing_still_rejects_camera_motion(self) -> None:
        people = [
            observation(frame, x)
            for frame, x in enumerate([0.05, 0.20, 0.40, 0.48, 0.53])
        ]
        static_objects = [
            observation(
                item.frame_index,
                item.box.centre_x,
                track_id=99,
                class_id=9,
                width=0.05,
                height=0.10,
            )
            for item in people
        ]

        result = CrossingDetector(settings()).detect(people + static_objects, fps=10.0)

        self.assertFalse(result.valid_events)
        self.assertEqual(
            result.rejected_events[0].rejection_reason,
            RejectionReason.CAMERA_MOTION,
        )

    def test_weak_static_overlap_does_not_imply_camera_motion(self) -> None:
        people = [
            observation(frame, 0.10 + 0.80 * frame / 99)
            for frame in range(100)
        ]
        shared_frames = [round(index * 99 / 8) for index in range(9)]
        static_objects = [
            observation(
                frame,
                people[frame].box.centre_x,
                track_id=99,
                class_id=9,
                width=0.05,
                height=0.10,
            )
            for frame in shared_frames
        ]

        result = CrossingDetector(settings()).detect(people + static_objects, fps=10.0)

        self.assertEqual(len(result.valid_events), 1)
        self.assertFalse(result.rejected_events)

    def test_strong_complete_crossing_overrides_tiny_size_filter(self) -> None:
        track = [
            observation(
                frame,
                0.20 + 0.60 * frame / 59,
                width=0.02,
                height=0.10,
            )
            for frame in range(60)
        ]

        result = CrossingDetector(settings()).detect(track, fps=10.0)

        self.assertEqual(len(result.valid_events), 1)
        self.assertFalse(result.rejected_events)

    def test_perspective_corridor_uses_the_pedestrian_foot_point(self) -> None:
        detector = CrossingDetector(settings())
        near_horizon = observation(0, 0.35, y=0.28, height=0.10)
        near_camera = observation(1, 0.35, y=0.85, height=0.10)

        states = detector.track_states([near_horizon, near_camera])

        self.assertEqual(states, ["LEFT", "ROAD"])

    def test_rejects_camera_motion(self) -> None:
        people = crossing_track()
        static_objects = [
            observation(
                item.frame_index,
                item.box.centre_x,
                track_id=99,
                class_id=9,
                width=0.05,
                height=0.10,
            )
            for item in people
        ]
        result = CrossingDetector(settings()).detect(people + static_objects, fps=10.0)
        self.assertFalse(result.valid_events)
        self.assertEqual(
            result.rejected_events[0].rejection_reason,
            RejectionReason.CAMERA_MOTION,
        )

    def test_rejects_rider(self) -> None:
        people = crossing_track()
        vehicles = [
            observation(
                item.frame_index,
                item.box.centre_x,
                track_id=50,
                class_id=1,
                width=0.12,
                height=0.24,
                y=0.55,
            )
            for item in people
        ]
        result = CrossingDetector(settings()).detect(people + vehicles, fps=10.0)
        self.assertFalse(result.valid_events)
        self.assertEqual(result.rejected_events[0].rejection_reason, RejectionReason.RIDER)

    def test_does_not_treat_car_overlap_as_rider(self) -> None:
        people = crossing_track()
        cars = [
            observation(
                item.frame_index,
                item.box.centre_x,
                track_id=50,
                class_id=2,
                width=0.12,
                height=0.24,
            )
            for item in people
        ]

        result = CrossingDetector(settings()).detect(people + cars, fps=10.0)

        self.assertEqual(len(result.valid_events), 1)
        self.assertFalse(result.rejected_events)

    def test_does_not_treat_brief_bicycle_overlap_as_rider(self) -> None:
        people = [
            observation(frame, 0.20 + 0.06 * frame)
            for frame in range(10)
        ]
        bicycles = [
            observation(
                item.frame_index,
                item.box.centre_x,
                track_id=50,
                class_id=1,
                width=0.12,
                height=0.24,
            )
            for item in people[4:6]
        ]

        result = CrossingDetector(settings()).detect(people + bicycles, fps=10.0)

        self.assertEqual(len(result.valid_events), 1)
        self.assertFalse(result.rejected_events)


if __name__ == "__main__":
    unittest.main()
