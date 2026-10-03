"""Deterministic country-independent jaywalking policy."""

from __future__ import annotations

from typing import Any, Sequence

from scripts.core.models import ContextAssessment, DecisionLabel, Ternary, Visibility


PERMISSION_CUES = {
    "marked_crosswalk": "marked crosswalk",
    "permissive_pedestrian_signal": "permissive pedestrian signal",
    "authorised_crossing_sign": "authorised crossing sign",
    "crossing_guard_permission": "crossing guard permission",
    "traffic_light": "traffic light",
}
LEGACY_PERMISSION_CUES = (
    "marked_crosswalk",
    "permissive_pedestrian_signal",
    "authorised_crossing_sign",
    "crossing_guard_permission",
)
INFRASTRUCTURE_FIELDS = (*PERMISSION_CUES, "prohibitive_pedestrian_signal")
CONTEXT_SCOPES = ("person", "scene")


class JaywalkingPolicy:
    """Apply the operational definition to observable VLM context."""

    def __init__(self, settings: dict[str, Any]) -> None:
        self.prohibitive_overrides = bool(
            settings.get("prohibitive_signal_overrides_crosswalk", True)
        )
        self.permission_cues = tuple(settings.get("permission_cues", LEGACY_PERMISSION_CUES))
        unknown = set(self.permission_cues).difference(PERMISSION_CUES)
        if not self.permission_cues or unknown:
            raise ValueError(
                "permission_cues must be a non-empty subset of: " + ", ".join(PERMISSION_CUES)
            )
        self.partial_visibility_uncertain = bool(
            settings.get("partial_visibility_uncertain", True)
        )
        self.context_scope = str(settings.get("context_scope", "person")).strip().lower()
        if self.context_scope not in CONTEXT_SCOPES:
            raise ValueError("context_scope must be one of: " + ", ".join(CONTEXT_SCOPES))
        # A positive then requires every infrastructure field, including cues that do
        # not grant permission, to be NO. Any YES or UNCERTAIN gives UNCERTAIN instead.
        self.strict_absence = bool(settings.get("strict_absence", False))
        # Scene scope only joins crossings this close in time (seconds between their
        # crossing intervals). None joins every crossing in the video.
        window = settings.get("context_scope_window_seconds")
        self.scope_window = None if window is None else float(window)
        if self.scope_window is not None and self.scope_window < 0:
            raise ValueError("context_scope_window_seconds must be null or non-negative")

    def decide_all(
        self,
        contexts: Sequence[ContextAssessment],
        times: Sequence[tuple[float, float]] | None = None,
    ) -> list[tuple[DecisionLabel, str]]:
        """Decide every valid crossing person in one video.

        In scene scope, a permission cue seen for any crossing person in the video
        counts for every crossing person, because a marked crosswalk or signal is a
        property of the scene that one sampled path can easily miss. With a scope
        window, ``times`` gives each person's crossing interval in seconds and only
        crossings at most the window apart share their cues: a long CROWD segment
        passes many places, a JAAD clip only one.
        """

        if self.context_scope == "person":
            return [self.decide(context) for context in contexts]

        if self.scope_window is None:
            groups = [list(range(len(contexts)))] * len(contexts)
        else:
            if times is None or len(times) != len(contexts):
                raise ValueError("A context scope window needs one crossing interval per person")
            groups = [
                [
                    other
                    for other, (start, end) in enumerate(times)
                    if max(start, times[index][0]) - min(end, times[index][1]) <= self.scope_window
                ]
                for index in range(len(contexts))
            ]

        decisions = []
        for context, group in zip(contexts, groups):
            nearby = [contexts[other] for other in group]
            scene = [
                name
                for name in self.permission_cues
                if any(getattr(item, name) == Ternary.YES for item in nearby)
            ]
            absent = all(self._infrastructure_absent(item) for item in nearby)
            decisions.append(
                self.decide(context, scene_permissions=scene, scene_absence_confirmed=absent)
            )
        return decisions

    def decide(
        self,
        context: ContextAssessment,
        scene_permissions: Sequence[str] = (),
        scene_absence_confirmed: bool | None = None,
    ) -> tuple[DecisionLabel, str]:
        """Return the final label and a concise deterministic reason."""

        if context.visibility == Visibility.INSUFFICIENT and not scene_permissions:
            return DecisionLabel.UNCERTAIN, "Crossing infrastructure is not sufficiently visible"

        if (
            self.prohibitive_overrides
            and context.prohibitive_pedestrian_signal == Ternary.YES
        ):
            return DecisionLabel.JAYWALKING, "A prohibitive pedestrian signal is visibly active"

        permissions = {name: getattr(context, name) for name in self.permission_cues}
        visible_permissions = [
            PERMISSION_CUES[name] for name, value in permissions.items() if value == Ternary.YES
        ]
        if visible_permissions:
            return DecisionLabel.COMPLIANT, f"Visible permission: {', '.join(visible_permissions)}"

        if scene_permissions:
            names = ", ".join(PERMISSION_CUES[name] for name in scene_permissions)
            return DecisionLabel.COMPLIANT, f"Scene permission seen for another crossing: {names}"

        if self.partial_visibility_uncertain and context.visibility == Visibility.PARTIAL:
            return DecisionLabel.UNCERTAIN, "No permission is visible, but scene visibility is partial"

        if any(value == Ternary.UNCERTAIN for value in permissions.values()):
            return DecisionLabel.UNCERTAIN, "At least one relevant crossing control is uncertain"

        if self.strict_absence:
            absent = (
                self._infrastructure_absent(context)
                if scene_absence_confirmed is None
                else scene_absence_confirmed
            )
            if not absent:
                return (
                    DecisionLabel.UNCERTAIN,
                    "Crossing infrastructure is not conclusively absent from the scene",
                )

        return (
            DecisionLabel.JAYWALKING,
            "Valid road crossing without a marked crossing or permissive traffic control",
        )

    @staticmethod
    def _infrastructure_absent(context: ContextAssessment) -> bool:
        # A field the prompt mode did not assess (None) neither proves nor blocks absence.
        return all(
            getattr(context, name) in (Ternary.NO, None) for name in INFRASTRUCTURE_FIELDS
        )
