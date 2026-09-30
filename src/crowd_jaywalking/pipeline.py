"""End-to-end per-person jaywalking pipeline."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from .config import ProjectConfig
from .crossing import CrossingDetector
from .crossing_classifier import CrossingClassifier
from .crossing_gate import CrossingGate
from .evidence import EvidenceBuilder
from .model_crossing import ModelCrossingDetector
from .models import (
    ContextAssessment,
    CrossingCheck,
    CrossingEvent,
    DecisionLabel,
    Ternary,
    PersonDecision,
    TrackObservation,
    VideoResult,
)
from .policy import JaywalkingPolicy
from .tracking import PersonTracker
from .vlm import HuggingFaceContextClassifier


class JaywalkingPipeline:
    """Detect crossings first, then classify visible global context per person."""

    def __init__(self, config: ProjectConfig) -> None:
        self.config = config

        # Load the configured local models once and reuse them for every video.
        self.context_classifier: HuggingFaceContextClassifier | None = None
        self._vlm_settings = config.vlm_settings()
        self._context_ready = False
        self._vlm_check = False
        self._vlm_check_version = config.crossing_gate_settings()["vlm_check_version"]
        self._rescue: tuple[float, float] | None = None

        self.tracker = PersonTracker(config.tracking_settings(), config.root)
        classifier_settings = config.crossing_classifier_settings()
        self.crossing_method = classifier_settings["decision_mode"]
        if self.crossing_method == "crowd_city":
            from .crowd_city_crossing import CrowdCityCrossingDetector

            # crowd-city's detector only proposes snippets; the VLM decides whether they cross.
            self._vlm_check = config.crossing_gate_settings()["vlm_check"]
            self.crossing_detector = CrowdCityCrossingDetector()
        elif self.crossing_method == "classifier":
            try:
                classifier = CrossingClassifier.load(classifier_settings["model"])
            except (FileNotFoundError, ValueError):
                if not classifier_settings["fallback_to_rules"]:
                    raise
                self.crossing_method = "rules"
                self.crossing_detector = CrossingDetector(config.crossing_settings())
            else:
                gate_settings = config.crossing_gate_settings()
                self._vlm_check = gate_settings["vlm_check"]
                self._rescue = (
                    None
                    if gate_settings["rescue_min_first_stage"] is None
                    else (
                        float(gate_settings["rescue_min_first_stage"]),
                        float(gate_settings["rescue_min_gate"]),
                    )
                )
                self.crossing_detector = ModelCrossingDetector(
                    classifier,
                    config.crossing_settings(),
                    classifier_settings["min_track_frames"],
                    # A configured gate that is missing is an error, never a silent skip.
                    CrossingGate.load(gate_settings["model"], gate_settings["min_precision"])
                    if gate_settings["enabled"]
                    else None,
                    gate_settings["min_scene_x_range"],
                )
        else:
            self.crossing_detector = CrossingDetector(config.crossing_settings())
        self.evidence_builder = EvidenceBuilder(config.evidence_settings())
        self._infrastructure_span = config.evidence_settings()["infrastructure_span"]
        self.policy = JaywalkingPolicy(config.policy_settings())


    def process_video(self, video_path: str | Path, evidence_root: str | Path) -> VideoResult:
        """Process one video and return video and person level decisions."""

        started = time.perf_counter()
        source = Path(video_path).resolve()
        fps, observations = self.tracker.track(source)
        return self.process_observations(
            source,
            evidence_root,
            fps,
            observations,
            started=started,
        )

    def process_observations(
        self,
        video_path: str | Path,
        evidence_root: str | Path,
        fps: float,
        observations: list[TrackObservation],
        *,
        started: float | None = None,
        camera_motion: Any | None = None,
    ) -> VideoResult:
        """Classify precomputed observations from the configured tracker."""

        started = time.perf_counter() if started is None else started
        source = Path(video_path).resolve()
        if getattr(self.crossing_detector, "needs_camera_motion", False):
            if camera_motion is None:
                from .camera_motion import estimate_camera_motion

                # BoT-SORT's own GMC estimate, recomputed from the frames.
                camera_motion = estimate_camera_motion(source)
            crossings = self.crossing_detector.detect(observations, fps, camera_motion)
        else:
            crossings = self.crossing_detector.detect(observations, fps)

        candidates = [(event, "gate") for event in crossings.valid_events]
        if self._rescue is not None:
            minimum_first_stage, minimum_gate = self._rescue
            candidates += [
                (item.event, "rescue")
                for item in getattr(crossings, "classifications", [])
                if not item.predicted_crossing
                and item.probability >= minimum_first_stage
                and item.gate_probability is not None
                and item.gate_probability >= minimum_gate
            ]

        assessed: list[tuple[CrossingEvent, ContextAssessment]] = []
        measurements = []
        checks: list[CrossingCheck] = []
        for event, source_name in candidates:
            if not self._context_ready:
                self.context_classifier = HuggingFaceContextClassifier(self._vlm_settings)
                self.context_classifier.ensure_ready()
                self._context_ready = True
            evidence = self.evidence_builder.build(
                video_path=source,
                event=event,
                observations=observations,
                output_root=evidence_root,
                fps=fps,
            )
            if self.context_classifier is None:
                raise RuntimeError("The configured Hugging Face context model is unavailable")
            if self._vlm_check:
                answer, summary = self.context_classifier.confirm_crossing(
                    evidence, version=getattr(self, "_vlm_check_version", "v1")
                )
                checks.append(CrossingCheck(event.person_id, source_name, answer, summary))
                # Only a clear YES counts: the positive output must be precise.
                if answer != Ternary.YES:
                    continue
            if getattr(self, "_infrastructure_span", "transition") == "track":
                evidence = self.evidence_builder.build(
                    video_path=source,
                    event=event,
                    observations=observations,
                    output_root=evidence_root,
                    fps=fps,
                    span="track",
                )
            assessed.append((event, self.context_classifier.classify(evidence)))

        outcomes = self.policy.decide_all([context for _, context in assessed])

        decisions = [
            PersonDecision(
                person_id=event.person_id,
                label=label,
                reason=reason,
                event=event,
                context=context,
            )
            for (event, context), (label, reason) in zip(assessed, outcomes)
        ]

        if any(item.label == DecisionLabel.JAYWALKING for item in decisions):
            prediction = DecisionLabel.JAYWALKING
        elif any(item.label == DecisionLabel.UNCERTAIN for item in decisions):
            prediction = DecisionLabel.UNCERTAIN
        else:
            prediction = DecisionLabel.COMPLIANT

        return VideoResult(
            video_path=str(source),
            prediction=prediction,
            person_decisions=decisions,
            rejected_candidates=crossings.rejected_events,
            latency_seconds=round(time.perf_counter() - started, 2),
            crossing_classifications=list(getattr(crossings, "classifications", [])),
            crossing_checks=checks,
        )
