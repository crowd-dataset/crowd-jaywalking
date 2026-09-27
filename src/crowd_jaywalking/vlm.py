"""Local Hugging Face VLM inference for observable crossing context."""

from __future__ import annotations

import gc
import json
from pathlib import Path
from typing import Any

from .models import ContextAssessment, EvidenceImage, Ternary, Visibility


BASELINE_PROMPT_VERSION = "global-context-v3"
FOCUSED_PROMPT_VERSION = "global-context-v5"
ZEBRA_LIGHT_PROMPT_VERSION = "zebra-light-v1"
ZEBRA_LIGHT_V2_PROMPT_VERSION = "zebra-light-v2"
ZEBRA_LIGHT_V3_PROMPT_VERSION = "zebra-light-v3"
ZEBRA_LIGHT_V4_PROMPT_VERSION = "zebra-light-v4"
PROMPT_VERSION = FOCUSED_PROMPT_VERSION
PROMPT_MODES = (
    "baseline_v3",
    "focused_v5",
    "zebra_light_v1",
    "zebra_light_v2",
    "zebra_light_v3",
    "zebra_light_v4",
)
# Signs and signal heads are static, so the controls task needs fewer moments.
CONTROL_TASK_FRAMES = 2

BASELINE_CONTEXT_PROMPT = """You are inspecting one tracked pedestrian crossing event.

The images are chronological. For each time point, the full scene image is followed by an enlarged focus image.
The same target pedestrian is enclosed by a RED bounding box labelled TARGET PERSON in every image.

Report only observable scene facts that apply to this target person's crossing location. Do not decide whether the person is jaywalking. Do not use assumptions about a country or local law.

Return one JSON object with exactly these keys:
{
  "marked_crosswalk": "YES|NO|UNCERTAIN",
  "permissive_pedestrian_signal": "YES|NO|UNCERTAIN",
  "authorised_crossing_sign": "YES|NO|UNCERTAIN",
  "crossing_guard_permission": "YES|NO|UNCERTAIN",
  "prohibitive_pedestrian_signal": "YES|NO|UNCERTAIN",
  "visibility": "CLEAR|PARTIAL|INSUFFICIENT",
  "evidence_summary": "one short sentence describing visible evidence"
}

Judge every field independently from visible evidence. A nearby feature counts only when it applies to the exact path traversed by the RED-boxed target person.

Field rules:
1. marked_crosswalk
   - YES only when zebra stripes or another unambiguous pedestrian road marking intersects the target's actual crossing trajectory.
   - A crosswalk elsewhere in the image, beside the trajectory, or merely near the target is NO.
2. permissive_pedestrian_signal and prohibitive_pedestrian_signal
   - First locate a dedicated pedestrian signal head that applies to the target's crossing direction. It normally shows a walking person, standing person, hand, or WALK/DON'T WALK symbol.
   - Circular red, amber, or green lamps for vehicles are NOT pedestrian signals. Ignore them completely for both fields.
   - If no qualifying pedestrian signal head is visible, set BOTH signal fields to NO.
   - If a qualifying pedestrian signal head is visible but its active state cannot be read, set BOTH signal fields to UNCERTAIN.
   - Set permissive_pedestrian_signal to YES only when that pedestrian signal visibly permits the target to cross; then prohibitive_pedestrian_signal must be NO.
   - Set prohibitive_pedestrian_signal to YES only when that pedestrian signal visibly tells the target not to cross; then permissive_pedestrian_signal must be NO.
   - The two signal fields can never both be YES.
3. authorised_crossing_sign
   - YES only for a clearly visible sign explicitly designating the target's crossing trajectory as a pedestrian crossing. Generic road signs and signs at another crossing are NO.
4. crossing_guard_permission
   - YES only when an identifiable authorised person is visibly directing this target to cross. The presence of police, workers, or other people without a visible directing gesture is NO.
5. visibility
   - CLEAR when the target trajectory and the relevant road markings/control locations are sufficiently visible for confident decisions.
   - PARTIAL when some relevant areas are visible but occlusion, distance, blur, or framing prevents at least one confident decision.
   - INSUFFICIENT when the evidence cannot establish the target trajectory or inspect its crossing context.

Use NO when the relevant area is sufficiently visible and the feature is absent, including when no dedicated pedestrian signal exists. Use UNCERTAIN only for a relevant feature or control whose presence or state genuinely cannot be resolved from the supplied images. Do not use UNCERTAIN merely because a feature is absent. Output JSON only."""

SHARED_INSTRUCTIONS = """You are inspecting one tracked pedestrian crossing event.
The images are chronological. The target pedestrian has a RED box. A separate TRAJECTORY MAP may show the observed foot point path as a YELLOW line.
Report only visible scene facts applying to this target's crossing location. Do not decide whether the person is jaywalking and do not apply country specific law.
Use YES only for positive visible evidence. Use NO when the relevant area is sufficiently visible and the feature is absent. Use UNCERTAIN when distance, blur, occlusion, or framing prevents a reliable decision.
Return JSON only, without markdown."""

ROAD_MARKING_PROMPT = f"""{SHARED_INSTRUCTIONS}

Task: inspect only the road markings at the target's crossing location.
TARGET ROAD AREA views contain clean road pixels. TRAJECTORY MAP views are spatial guides and may have an imperfect path estimate.

Return exactly:
{{
  "marked_crosswalk": "YES|NO|UNCERTAIN",
  "evidence_summary": "one short sentence"
}}

Answer YES when zebra stripes or another unambiguous pedestrian crossing marking designates the road area used by the target. Do not require the estimated yellow line to touch every painted stripe exactly. Answer NO for ordinary lane lines, stop lines, arrows, text, or a crosswalk serving a clearly different location. Answer UNCERTAIN when the relevant road surface is hidden, too distant, blurred, or only partly outside the image."""

PEDESTRIAN_SIGNAL_PROMPT = f"""{SHARED_INSTRUCTIONS}

Task: inspect only dedicated pedestrian signals applying to the target's crossing direction. CONTROL SEARCH views enlarge likely control locations.

Return exactly:
{{
  "dedicated_pedestrian_signal_visible": "YES|NO|UNCERTAIN",
  "permissive_pedestrian_signal": "YES|NO|UNCERTAIN",
  "prohibitive_pedestrian_signal": "YES|NO|UNCERTAIN",
  "evidence_summary": "one short sentence"
}}

A dedicated pedestrian signal depicts a walking person, standing person, raised hand, or WALK or DON'T WALK text. Circular vehicle traffic lamps are never pedestrian signals, regardless of colour. If no dedicated pedestrian signal is visible, answer NO for visibility and both states. If a possible dedicated signal is visible but its identity or active state cannot be read, answer UNCERTAIN for visibility and both states. A yellow or amber vehicle lamp is not a prohibitive pedestrian signal. The permissive and prohibitive states can never both be YES."""

AUTHORISATION_PROMPT = f"""{SHARED_INSTRUCTIONS}

Task: inspect only physical pedestrian crossing signs and crossing guard directions applying to the target's crossing location.

Return exactly:
{{
  "authorised_crossing_sign": "YES|NO|UNCERTAIN",
  "crossing_guard_permission": "YES|NO|UNCERTAIN",
  "evidence_summary": "one short sentence"
}}

An authorised crossing sign must be a physical signboard or sign assembly with a visible pedestrian symbol or text that designates this crossing. Zebra stripes, painted road symbols, lane markings, and any other paint on the road are never signs. Generic warning, parking, speed, and vehicle direction signs are NO. crossing_guard_permission is YES only when an identifiable authorised person visibly directs this target to cross; merely being present is NO."""

VISIBILITY_PROMPT = f"""{SHARED_INSTRUCTIONS}

Task: rate whether the supplied evidence supports reliable observable context decisions for this target.
Use the full scene and target detail views.

Return exactly:
{{
  "visibility": "CLEAR|PARTIAL|INSUFFICIENT",
  "evidence_summary": "one short sentence"
}}

CLEAR means the target trajectory and relevant crossing context are sufficiently visible for confident decisions. PARTIAL means the event is identifiable but occlusion, distance, blur, or framing prevents at least one confident context decision. INSUFFICIENT means the evidence cannot establish the target trajectory or inspect its crossing context."""

ZEBRA_LIGHT_PROMPT = """You are inspecting one tracked pedestrian crossing a road, seen from a vehicle dashcam.
The images are chronological moments. The target pedestrian has a RED box labelled TARGET. TARGET ROAD AREA views show the road the target crosses. CONTROL SEARCH views enlarge the upper scene where traffic lights usually are.

Report only what is visible on the stretch of road the target crosses: the road between the kerbs on the target's path, and about one car length to either side of that path. Ignore features on other streets or far along the road. Do not decide whether the person is jaywalking.

Return one JSON object with exactly these keys:
{
  "zebra_crossing": "YES|NO|UNCERTAIN",
  "traffic_light": "YES|NO|UNCERTAIN",
  "visibility": "CLEAR|PARTIAL|INSUFFICIENT",
  "evidence_summary": "one short sentence"
}

zebra_crossing:
- YES when painted pedestrian crossing markings are on that stretch of road: zebra stripes, a ladder pattern, or two parallel painted lines bounding a crosswalk. The target may walk slightly beside the markings.
- NO when the road surface of that stretch is visible and has no such markings. Lane lines, centre lines, stop lines, arrows, text, and parking bays are not crossings.
- UNCERTAIN only when that road surface is hidden, too dark, or too blurred to judge.

traffic_light:
- YES when any traffic signal head, for vehicles or for pedestrians, of any colour, on a pole, mast arm, or overhead wire, controls the junction or crossing on that stretch of road.
- NO when the surroundings of that stretch are visible and no signal head is there. A signal at a different junction further along the road is NO.
- UNCERTAIN only when the surroundings of that stretch cannot be seen well enough to judge.

visibility:
- CLEAR when the road surface and surroundings of the crossing location are clearly visible in at least one image.
- PARTIAL when only part of them is visible.
- INSUFFICIENT when neither can be judged.

Output JSON only."""

# Version 2 adds a full resolution crossing close-up, reports pedestrian crossing
# signs, and prefers UNCERTAIN over NO for faint markings, because the positive
# answer (no zebra, no light) must be precise.
ZEBRA_LIGHT_V2_PROMPT = """You are inspecting one tracked pedestrian crossing a road, seen from a vehicle dashcam.
The images are chronological moments. The target pedestrian has a RED box. CROSSING CLOSE-UP views are enlarged, full resolution crops of the road around the target's feet and path; look there for painted markings. CONTROL SEARCH views enlarge the upper scene where traffic lights and signs usually are.

Report only what is visible at the place where the target crosses: the road between the kerbs on the target's path, and about one car length to either side of that path. Ignore features on other streets or far along the road. Do not decide whether the person is jaywalking.

Return one JSON object with exactly these keys:
{
  "zebra_crossing": "YES|NO|UNCERTAIN",
  "traffic_light": "YES|NO|UNCERTAIN",
  "crossing_sign": "YES|NO|UNCERTAIN",
  "visibility": "CLEAR|PARTIAL|INSUFFICIENT",
  "evidence_summary": "one short sentence"
}

zebra_crossing:
- YES when painted pedestrian crossing markings are on the road at that place: zebra stripes, a ladder pattern, or two parallel painted lines bounding a crosswalk, even if faded, partly covered, or small in the distance. The target may walk beside the markings.
- NO only when the road surface at that place is clearly visible in a CROSSING CLOSE-UP and has no such markings. Lane lines, centre lines, stop lines, arrows, text, and parking bays are not crossings.
- UNCERTAIN when faint, worn, distant, or partly hidden stripes might be a crossing, or when the road surface is hidden, dark, or blurred.

traffic_light:
- YES when any traffic signal head, for vehicles or for pedestrians, of any colour, on a pole, mast arm, or overhead wire, controls the junction or crossing at that place.
- NO when the surroundings of that place are visible and no signal head is there. A signal at a different junction further along the road is NO.
- UNCERTAIN when the surroundings cannot be seen well enough to judge.

crossing_sign:
- YES when a road sign marking a pedestrian crossing is at that place, such as a sign with a walking person, often yellow, blue, or white.
- NO when the surroundings are visible and there is no such sign. Other road signs are NO.
- UNCERTAIN when a sign is there but cannot be read.

visibility:
- CLEAR when the road surface and surroundings of the crossing place are clearly visible in at least one image.
- PARTIAL when only part of them is visible.
- INSUFFICIENT when neither can be judged.

Output JSON only."""

# Version 3 splits the question in two. Markings use a close-up and the whole road
# ahead; signals and signs use near native resolution tiles of the upper scene.
MARKINGS_V3_PROMPT = """You are inspecting one tracked pedestrian crossing a road, seen from a vehicle dashcam.
The images are chronological moments. The target pedestrian has a RED box. CROSSING CLOSE-UP views are enlarged crops of the road around the target's feet and path. ROAD AHEAD views show the whole road between the camera and the target.

Task: decide whether painted pedestrian crossing markings are on the road the target crosses, within about 20 metres of the target's path, including between the camera and the target. Ignore side streets. Do not decide whether the person is jaywalking.

Return one JSON object with exactly these keys:
{
  "zebra_crossing": "YES|NO|UNCERTAIN",
  "visibility": "CLEAR|PARTIAL|INSUFFICIENT",
  "evidence_summary": "one short sentence"
}

zebra_crossing:
- YES when zebra stripes, a ladder pattern, or two parallel painted lines bounding a crosswalk are on that road within about 20 metres of the target's path, even if faded, partly covered by snow or vehicles, or small in the distance.
- NO only when that road surface is clearly visible and has no such markings. Lane lines, centre lines, stop lines, arrows, text, and parking bays are not crossings.
- UNCERTAIN when faint, worn, or partly hidden stripes might be a crossing, or when the road surface is hidden, dark, or blurred.

visibility:
- CLEAR when the road surface around the target's path is clearly visible in at least one image.
- PARTIAL when only part of it is visible.
- INSUFFICIENT when it cannot be judged.

Output JSON only."""

CONTROLS_V3_PROMPT = """You are inspecting one tracked pedestrian crossing a road, seen from a vehicle dashcam.
The FULL SCENE views show the whole image; the target pedestrian has a RED box. CONTROL TILE views are enlarged, overlapping crops of the upper part of the same images, where traffic lights and road signs usually are.

Task: decide whether traffic lights or pedestrian crossing signs are at the place where the target crosses, within about 20 metres of the target's path, on the road the target crosses. Ignore side streets and junctions far along the road. Do not decide whether the person is jaywalking.

Return one JSON object with exactly these keys:
{
  "traffic_light": "YES|NO|UNCERTAIN",
  "crossing_sign": "YES|NO|UNCERTAIN",
  "evidence_summary": "one short sentence"
}

traffic_light:
- YES when any traffic signal head, for vehicles or pedestrians, of any colour, on a pole, mast arm, or overhead wire, is within about 20 metres of the target's path.
- NO when that area is visible and has no signal head. A signal at a junction clearly further along the road is NO.
- UNCERTAIN when a possible signal head cannot be made out.

crossing_sign:
- YES when a road sign marking a pedestrian crossing, usually showing a walking person, often blue, yellow, or white, is within about 20 metres of the target's path. Such signs are small; check every CONTROL TILE.
- NO when that area is visible and has no such sign. Other road signs are NO.
- UNCERTAIN when a sign is there but cannot be read.

Output JSON only."""

CROSSING_CHECK_PROMPT_VERSION = "crossing-check-v1"
CROSSING_CHECK_PROMPT = """You are inspecting one tracked pedestrian in a vehicle dashcam video.
The images are chronological moments. The target pedestrian has a RED box. TRAJECTORY MAP views are the full scene with the target's observed foot point path drawn as a YELLOW line ending in an arrow. TARGET DETAIL views are close-ups of the target.

Task: decide whether the target walks across the road carriageway, the part of the road used by vehicles, moving from one side of it towards the other during these moments. Crossing only part of the carriageway still counts, if the target is on the vehicle road surface and moving across it.

Return one JSON object with exactly these keys:
{
  "crosses_road": "YES|NO|UNCERTAIN",
  "evidence_summary": "one short sentence"
}

- YES when the target is on the vehicle road surface and moves across it, not along it.
- NO when the target walks along the pavement or verge, walks along the road edge, stands or waits at the kerb, walks in a car park away from any road, gets into or out of a vehicle, or is riding a bicycle, scooter, or motorcycle.
- UNCERTAIN when the target's position or movement cannot be judged from the images.
The yellow path is estimated from a moving camera and may be distorted; judge the target's real movement relative to the road and kerbs.

Output JSON only."""

# Version 4 keeps the version 3 views but asks the model to list what it sees
# before answering, so small markings, signs, and signal heads are not skipped.
MARKINGS_V4_PROMPT = MARKINGS_V3_PROMPT.replace(
    """{
  "zebra_crossing": "YES|NO|UNCERTAIN",""",
    """{
  "road_markings_seen": ["every painted marking on that road near the target, e.g. lane line, centre line, stop line, zebra stripes, arrow"],
  "zebra_crossing": "YES|NO|UNCERTAIN",""",
).replace(
    "Output JSON only.",
    "First fill road_markings_seen by checking every CROSSING CLOSE-UP and ROAD AHEAD view, then decide zebra_crossing from that list. Output JSON only.",
)
CONTROLS_V4_PROMPT = CONTROLS_V3_PROMPT.replace(
    """{
  "traffic_light": "YES|NO|UNCERTAIN",""",
    """{
  "signals_and_signs_seen": ["every traffic signal head and road sign visible in any CONTROL TILE, with its tile number and whether it is near the target's path"],
  "traffic_light": "YES|NO|UNCERTAIN",""",
).replace(
    "Output JSON only.",
    "First fill signals_and_signs_seen by checking every CONTROL TILE, then decide traffic_light and crossing_sign from that list. Output JSON only.",
)

CROSSING_CHECK_V2_PROMPT_VERSION = "crossing-check-v2"
CROSSING_CHECK_V2_PROMPT = """You are inspecting one tracked pedestrian in a vehicle dashcam video.
The images are chronological moments spread over the whole time the target is visible. The target pedestrian has a RED box. TRAJECTORY MAP views are the full scene with the target's observed foot point path drawn as a YELLOW line ending in an arrow. TARGET DETAIL views are close-ups of the target.

Task: decide whether the target crosses the road: at some moment the target's feet are on the vehicle road surface (the carriageway where cars drive, not the pavement, verge, or a car park) and the target is moving across the road rather than along it. The target may be on the pavement in the other moments, before or after crossing.

Return one JSON object with exactly these keys:
{
  "crosses_road": "YES|NO|UNCERTAIN",
  "evidence_summary": "one short sentence"
}

- YES when, in at least one image, the target stands or walks on the vehicle road surface and the moments together show movement across the road.
- NO when the target stays on the pavement, verge, or kerb in every image, walks along the road edge without crossing, walks in a car park away from any road, gets into or out of a vehicle, or is riding a bicycle, scooter, or motorcycle.
- UNCERTAIN when the target's position cannot be judged from the images.
The yellow path is estimated from a moving camera and may be distorted; judge the target's real position relative to the kerbs.

Output JSON only."""

# Retained as a single fingerprintable value for manifests and compatibility.
FOCUSED_CONTEXT_PROMPT = "\n\n".join(
    (
        ROAD_MARKING_PROMPT,
        PEDESTRIAN_SIGNAL_PROMPT,
        AUTHORISATION_PROMPT,
        VISIBILITY_PROMPT,
    )
)
CONTEXT_PROMPT = FOCUSED_CONTEXT_PROMPT


def normalise_prompt_mode(value: str) -> str:
    """Validate and normalise a configured VLM prompt mode."""

    mode = str(value).strip().lower()
    if mode not in PROMPT_MODES:
        raise VLMError(
            "vlm_prompt_mode must be one of: " + ", ".join(PROMPT_MODES)
        )
    return mode


def prompt_version_for_mode(mode: str) -> str:
    """Return the immutable prompt version associated with one mode."""

    return {
        "baseline_v3": BASELINE_PROMPT_VERSION,
        "focused_v5": FOCUSED_PROMPT_VERSION,
        "zebra_light_v1": ZEBRA_LIGHT_PROMPT_VERSION,
        "zebra_light_v2": ZEBRA_LIGHT_V2_PROMPT_VERSION,
        "zebra_light_v3": ZEBRA_LIGHT_V3_PROMPT_VERSION,
        "zebra_light_v4": ZEBRA_LIGHT_V4_PROMPT_VERSION,
    }[normalise_prompt_mode(mode)]


def prompt_content_for_mode(mode: str) -> str:
    """Return all prompt text used by one mode for result fingerprinting."""

    return {
        "baseline_v3": BASELINE_CONTEXT_PROMPT,
        "focused_v5": FOCUSED_CONTEXT_PROMPT,
        "zebra_light_v1": ZEBRA_LIGHT_PROMPT,
        "zebra_light_v2": ZEBRA_LIGHT_V2_PROMPT,
        "zebra_light_v3": MARKINGS_V3_PROMPT + "\n\n" + CONTROLS_V3_PROMPT,
        "zebra_light_v4": MARKINGS_V4_PROMPT + "\n\n" + CONTROLS_V4_PROMPT,
    }[normalise_prompt_mode(mode)]


class VLMError(RuntimeError):
    """Raised when local VLM inference cannot produce a valid assessment."""


def _local_image_reference(path: Path) -> str:
    """Return a native absolute path accepted on Windows, macOS, and Linux."""

    return str(path.resolve())


def _sample_evidence(
    evidence: list[EvidenceImage],
    maximum: int,
) -> list[EvidenceImage]:
    """Select evenly spaced evidence without losing the first or final frame."""

    ordered = sorted(evidence, key=lambda item: item.frame_index)
    if len(ordered) <= maximum:
        return ordered
    if maximum == 1:
        return [ordered[len(ordered) // 2]]
    indices = [
        round(index * (len(ordered) - 1) / (maximum - 1))
        for index in range(maximum)
    ]
    return [ordered[index] for index in indices]


def _view_paths(
    item: EvidenceImage,
    views: tuple[str, ...],
) -> list[tuple[str, Path]]:
    available = {
        "full scene": item.context_path,
        "target detail": item.focus_path,
        "target road area": item.road_path or item.context_path,
        "trajectory map": item.trajectory_path or item.context_path,
        "control search left": item.control_left_path or item.context_path,
        "control search right": item.control_right_path or item.context_path,
        "crossing close-up": item.crossing_path or item.road_path or item.context_path,
    }
    selected: list[tuple[str, Path]] = []
    seen: set[Path] = set()
    for view in views:
        if view == "control tiles":
            for index, tile in enumerate(item.tile_paths or ()):
                selected.append((f"control tile {index + 1}", tile))
            continue
        if view == "road ahead":
            path = item.lower_road_path or item.context_path
            if path.resolve() not in seen:
                selected.append((view, path))
                seen.add(path.resolve())
            continue
        path = available[view]
        resolved = path.resolve()
        if resolved not in seen:
            selected.append((view, path))
            seen.add(resolved)
    return selected


class HuggingFaceContextClassifier:
    """Run focused observable context tasks with one local Hugging Face VLM."""

    def __init__(self, settings: dict[str, Any]) -> None:
        self.model_id = str(settings["model_id"])
        self.model_family = self._model_family(self.model_id)
        self.prompt_mode = normalise_prompt_mode(
            str(settings.get("prompt_mode", "baseline_v3"))
        )
        self.max_new_tokens = int(settings.get("max_new_tokens", 300))
        self.task_max_frames = int(settings.get("task_max_frames", 4))
        self.model = None
        self.processor = None
        self._process_vision_info = None
        self._load(settings)

    def _load(self, settings: dict[str, Any]) -> None:
        try:
            import torch
            from transformers import AutoProcessor
        except ImportError as error:
            raise VLMError(
                "Hugging Face VLM dependencies are missing. Run 'uv sync' from the repository root."
            ) from error

        cache_dir_value = settings.get("cache_dir")
        cache_dir = str(cache_dir_value) if cache_dir_value else None
        local_files_only = bool(settings.get("local_files_only", False))
        dtype = self._resolve_dtype(str(settings.get("torch_dtype", "auto")), torch)

        processor_kwargs: dict[str, Any] = {
            "cache_dir": cache_dir,
            "local_files_only": local_files_only,
        }
        model_kwargs: dict[str, Any] = {
            "cache_dir": cache_dir,
            "device_map": settings.get("device_map", "auto"),
            "local_files_only": local_files_only,
            "low_cpu_mem_usage": True,
            "dtype": dtype,
        }
        attention = settings.get("attn_implementation")
        if attention:
            model_kwargs["attn_implementation"] = str(attention)
        if settings.get("quantization") == "4bit":
            # 4-bit weights let a 27B model fit in 32 GB of VRAM with evidence images.
            from transformers import BitsAndBytesConfig

            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
            )

        try:
            if self.model_family == "qwen":
                from qwen_vl_utils import process_vision_info

                processor_kwargs.update(
                    {
                        "min_pixels": int(settings.get("min_pixels", 200704)),
                        "max_pixels": int(settings.get("max_pixels", 401408)),
                    }
                )
                if "qwen3.5" in self.model_id.lower():
                    from transformers import Qwen3_5ForConditionalGeneration

                    model_class = Qwen3_5ForConditionalGeneration
                elif "qwen3" in self.model_id.lower():
                    from transformers import Qwen3VLForConditionalGeneration

                    model_class = Qwen3VLForConditionalGeneration
                else:
                    from transformers import Qwen2_5_VLForConditionalGeneration

                    model_class = Qwen2_5_VLForConditionalGeneration
                self._process_vision_info = process_vision_info
            else:
                from transformers import AutoModelForMultimodalLM

                model_class = AutoModelForMultimodalLM
                processor_kwargs["padding_side"] = "left"

            self.processor = AutoProcessor.from_pretrained(self.model_id, **processor_kwargs)
            self.model = model_class.from_pretrained(self.model_id, **model_kwargs)
            self.model.eval()
        except Exception as error:
            raise VLMError(
                f"Could not load '{self.model_id}' from Hugging Face. "
                "Check disk space, memory, network access, and any required Hugging Face login."
            ) from error

    def ensure_ready(self) -> None:
        """Confirm that the processor and model finished loading."""

        if self.processor is None or self.model is None:
            raise VLMError(f"Hugging Face model '{self.model_id}' is not ready")
        if self.model_family == "qwen" and self._process_vision_info is None:
            raise VLMError(f"Qwen vision utilities for '{self.model_id}' are not ready")

    def classify(self, evidence: list[EvidenceImage]) -> ContextAssessment:
        """Run the configured baseline or focused observable context tasks."""

        if not evidence:
            raise VLMError("No evidence images were supplied to the VLM")
        self.ensure_ready()
        if self.prompt_mode == "baseline_v3":
            return self._validate_response(
                self._classify_task(
                    sorted(evidence, key=lambda item: item.frame_index),
                    BASELINE_CONTEXT_PROMPT,
                    ("full scene", "target detail"),
                    "baseline context",
                )
            )

        selected = _sample_evidence(evidence, self.task_max_frames)

        if self.prompt_mode == "zebra_light_v1":
            return self._validate_zebra_light_response(
                self._classify_task(
                    selected,
                    ZEBRA_LIGHT_PROMPT,
                    (
                        "full scene",
                        "target road area",
                        "control search left",
                        "control search right",
                    ),
                    "zebra and traffic light",
                )
            )

        if self.prompt_mode in ("zebra_light_v3", "zebra_light_v4"):
            listing = self.prompt_mode == "zebra_light_v4"
            markings = self._validate_markings_v3_response(
                self._classify_task(
                    selected,
                    MARKINGS_V4_PROMPT if listing else MARKINGS_V3_PROMPT,
                    ("full scene", "crossing close-up", "road ahead"),
                    "crossing markings",
                )
            )
            controls = self._validate_controls_v3_response(
                self._classify_task(
                    _sample_evidence(evidence, CONTROL_TASK_FRAMES),
                    CONTROLS_V4_PROMPT if listing else CONTROLS_V3_PROMPT,
                    ("full scene", "control tiles"),
                    "traffic lights and crossing signs",
                )
            )
            return ContextAssessment(
                marked_crosswalk=markings["zebra_crossing"],
                permissive_pedestrian_signal=None,
                authorised_crossing_sign=controls["crossing_sign"],
                crossing_guard_permission=None,
                prohibitive_pedestrian_signal=None,
                visibility=markings["visibility"],
                evidence_summary=" ".join(
                    item
                    for item in (markings["evidence_summary"], controls["evidence_summary"])
                    if item
                ),
                traffic_light=controls["traffic_light"],
            )

        if self.prompt_mode == "zebra_light_v2":
            return self._validate_zebra_light_response(
                self._classify_task(
                    selected,
                    ZEBRA_LIGHT_V2_PROMPT,
                    (
                        "full scene",
                        "crossing close-up",
                        "control search left",
                        "control search right",
                    ),
                    "zebra, traffic light, and crossing sign",
                ),
                with_sign=True,
            )

        road = self._validate_road_response(
            self._classify_task(
                selected,
                ROAD_MARKING_PROMPT,
                ("full scene", "target road area", "trajectory map"),
                "road marking",
            )
        )
        signals = self._validate_signal_response(
            self._classify_task(
                selected,
                PEDESTRIAN_SIGNAL_PROMPT,
                ("full scene", "control search left", "control search right"),
                "pedestrian signal",
            )
        )
        authorisation = self._validate_authorisation_response(
            self._classify_task(
                selected,
                AUTHORISATION_PROMPT,
                ("full scene", "control search left", "control search right"),
                "crossing authorisation",
            )
        )
        visibility = self._validate_visibility_response(
            self._classify_task(
                selected,
                VISIBILITY_PROMPT,
                ("full scene", "target detail"),
                "visibility",
            )
        )

        summaries = [
            str(road["evidence_summary"]).strip(),
            str(signals["evidence_summary"]).strip(),
            str(authorisation["evidence_summary"]).strip(),
            str(visibility["evidence_summary"]).strip(),
        ]
        return ContextAssessment(
            marked_crosswalk=road["marked_crosswalk"],
            permissive_pedestrian_signal=signals["permissive_pedestrian_signal"],
            authorised_crossing_sign=authorisation["authorised_crossing_sign"],
            crossing_guard_permission=authorisation["crossing_guard_permission"],
            prohibitive_pedestrian_signal=signals["prohibitive_pedestrian_signal"],
            visibility=visibility["visibility"],
            evidence_summary=" ".join(item for item in summaries if item),
        )

    def confirm_crossing(
        self,
        evidence: list[EvidenceImage],
        version: str = "v1",
        max_frames: int | None = None,
    ) -> tuple[Ternary, str]:
        """Ask whether the target really walks across the carriageway."""

        if not evidence:
            raise VLMError("No evidence images were supplied to the VLM")
        self.ensure_ready()
        prompt = {"v1": CROSSING_CHECK_PROMPT, "v2": CROSSING_CHECK_V2_PROMPT}[version]
        payload = self._decode_payload(
            self._classify_task(
                _sample_evidence(evidence, max_frames or self.task_max_frames),
                prompt,
                ("trajectory map", "target detail"),
                "crossing check",
            )
        )
        if set(payload) != {"crosses_road", "evidence_summary"}:
            raise VLMError(f"VLM returned unexpected crossing check keys: {payload}")
        return (
            self._ternary(payload["crosses_road"], payload),
            str(payload["evidence_summary"]).strip(),
        )

    def _classify_task(
        self,
        evidence: list[EvidenceImage],
        prompt: str,
        views: tuple[str, ...],
        task_name: str,
    ) -> str:
        if self.model_family == "qwen":
            return self._classify_qwen(evidence, prompt, views, task_name)
        return self._classify_gemma(evidence, prompt, views, task_name)

    def _classify_qwen(
        self,
        evidence: list[EvidenceImage],
        prompt_text: str,
        views: tuple[str, ...],
        task_name: str,
    ) -> str:
        """Run one focused task with Qwen2.5 VL or Qwen3 VL."""

        content: list[dict[str, str]] = []
        for index, item in enumerate(evidence, start=1):
            for label, path in _view_paths(item, views):
                content.extend(
                    [
                        {"type": "text", "text": f"Time {index}: {label}."},
                        {"type": "image", "image": _local_image_reference(path)},
                    ]
                )
        content.append({"type": "text", "text": prompt_text})
        messages = [{"role": "user", "content": content}]

        try:
            prompt = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                # Qwen3.5 thinks by default; the answers must be plain JSON.
                enable_thinking=False,
            )
            image_inputs, video_inputs = self._process_vision_info(messages)
            inputs = self.processor(
                text=[prompt],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            inputs = inputs.to(self.model.device)

            import torch

            with torch.inference_mode():
                generated_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,
                    use_cache=True,
                )
            trimmed_ids = [
                output_ids[len(input_ids) :]
                for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
            ]
            return self.processor.batch_decode(
                trimmed_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0]
        except Exception as error:
            raise VLMError(
                f"Local Hugging Face {task_name} inference failed for '{self.model_id}'. "
                "The evaluation stopped instead of inventing a label."
            ) from error

    def _classify_gemma(
        self,
        evidence: list[EvidenceImage],
        prompt_text: str,
        views: tuple[str, ...],
        task_name: str,
    ) -> str:
        """Run one focused task with Gemma 4."""

        content: list[dict[str, str]] = []
        image_order: list[str] = []
        image_number = 0
        for index, item in enumerate(evidence, start=1):
            for label, path in _view_paths(item, views):
                image_number += 1
                content.append(
                    {"type": "image", "url": _local_image_reference(path)}
                )
                image_order.append(f"Image {image_number} is time {index}: {label}.")
        content.append(
            {
                "type": "text",
                "text": "\n".join([*image_order, prompt_text]),
            }
        )
        messages = [{"role": "user", "content": content}]

        try:
            inputs = self.processor.apply_chat_template(
                messages,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
                add_generation_prompt=True,
                enable_thinking=False,
            ).to(self.model.device)
            input_length = inputs["input_ids"].shape[-1]

            import torch

            with torch.inference_mode():
                generated_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,
                    use_cache=True,
                )
            return self.processor.decode(
                generated_ids[0][input_length:],
                skip_special_tokens=True,
            )
        except Exception as error:
            raise VLMError(
                f"Local Hugging Face {task_name} inference failed for '{self.model_id}'. "
                "The evaluation stopped instead of inventing a label."
            ) from error

    def close(self) -> None:
        """Release model memory before another comparison model is loaded."""

        self.model = None
        self.processor = None
        self._process_vision_info = None
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    @staticmethod
    def _model_family(model_id: str) -> str:
        normalised = model_id.strip().lower()
        if "qwen2.5-vl" in normalised or "qwen3-vl" in normalised or "qwen3.5" in normalised:
            return "qwen"
        if "gemma-4" in normalised:
            return "gemma"
        raise VLMError(
            "Unsupported VLM. Use a Qwen2.5 VL, Qwen3 VL, or Gemma 4 model ID."
        )

    @staticmethod
    def _resolve_dtype(value: str, torch_module: Any) -> Any:
        normalised = value.strip().lower()
        if normalised == "auto":
            return "auto"
        allowed = {
            "float16": torch_module.float16,
            "bfloat16": torch_module.bfloat16,
            "float32": torch_module.float32,
        }
        if normalised not in allowed:
            raise VLMError(
                "vlm.torch_dtype must be one of: auto, float16, bfloat16, float32"
            )
        return allowed[normalised]

    @staticmethod
    def _decode_payload(content: str) -> dict[str, Any]:
        cleaned = content.strip()
        if cleaned.startswith("```"):
            lines = cleaned.splitlines()
            if lines and lines[0].strip().startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            cleaned = "\n".join(lines).strip()
        try:
            payload = json.loads(cleaned)
        except json.JSONDecodeError as error:
            raise VLMError(f"VLM returned invalid JSON: {content[:300]}") from error
        if not isinstance(payload, dict):
            raise VLMError(f"VLM returned a non-object response: {payload}")
        return payload

    @staticmethod
    def _ternary(value: Any, payload: dict[str, Any]) -> Ternary:
        try:
            return Ternary(str(value).upper())
        except ValueError as error:
            raise VLMError(f"VLM returned an invalid context schema: {payload}") from error

    @classmethod
    def _validate_road_response(cls, content: str) -> dict[str, Any]:
        payload = cls._decode_payload(content)
        expected = {"marked_crosswalk", "evidence_summary"}
        if set(payload) != expected:
            raise VLMError(f"VLM returned unexpected road marking keys: {payload}")
        return {
            "marked_crosswalk": cls._ternary(payload["marked_crosswalk"], payload),
            "evidence_summary": str(payload["evidence_summary"]).strip(),
        }

    @classmethod
    def _validate_signal_response(cls, content: str) -> dict[str, Any]:
        payload = cls._decode_payload(content)
        expected = {
            "dedicated_pedestrian_signal_visible",
            "permissive_pedestrian_signal",
            "prohibitive_pedestrian_signal",
            "evidence_summary",
        }
        if set(payload) != expected:
            raise VLMError(f"VLM returned unexpected pedestrian signal keys: {payload}")

        signal_visible = cls._ternary(
            payload["dedicated_pedestrian_signal_visible"], payload
        )
        permissive = cls._ternary(payload["permissive_pedestrian_signal"], payload)
        prohibitive = cls._ternary(payload["prohibitive_pedestrian_signal"], payload)
        if signal_visible is Ternary.NO:
            permissive = Ternary.NO
            prohibitive = Ternary.NO
        elif signal_visible is Ternary.UNCERTAIN:
            permissive = Ternary.UNCERTAIN
            prohibitive = Ternary.UNCERTAIN
        elif permissive is Ternary.YES and prohibitive is Ternary.YES:
            raise VLMError(
                "VLM returned mutually exclusive pedestrian signal states: "
                "permissive and prohibitive cannot both be YES"
            )
        return {
            "permissive_pedestrian_signal": permissive,
            "prohibitive_pedestrian_signal": prohibitive,
            "evidence_summary": str(payload["evidence_summary"]).strip(),
        }

    @classmethod
    def _validate_authorisation_response(cls, content: str) -> dict[str, Any]:
        payload = cls._decode_payload(content)
        expected = {
            "authorised_crossing_sign",
            "crossing_guard_permission",
            "evidence_summary",
        }
        if set(payload) != expected:
            raise VLMError(
                f"VLM returned unexpected crossing authorisation keys: {payload}"
            )
        return {
            "authorised_crossing_sign": cls._ternary(
                payload["authorised_crossing_sign"], payload
            ),
            "crossing_guard_permission": cls._ternary(
                payload["crossing_guard_permission"], payload
            ),
            "evidence_summary": str(payload["evidence_summary"]).strip(),
        }

    @classmethod
    def _validate_zebra_light_response(
        cls, content: str, with_sign: bool = False
    ) -> ContextAssessment:
        payload = cls._decode_payload(content)
        expected = {"zebra_crossing", "traffic_light", "visibility", "evidence_summary"}
        if with_sign:
            expected.add("crossing_sign")
        if set(payload) != expected:
            raise VLMError(f"VLM returned unexpected zebra and light keys: {payload}")
        try:
            visibility = Visibility(str(payload["visibility"]).upper())
        except ValueError as error:
            raise VLMError(f"VLM returned an invalid visibility schema: {payload}") from error
        return ContextAssessment(
            marked_crosswalk=cls._ternary(payload["zebra_crossing"], payload),
            permissive_pedestrian_signal=None,
            authorised_crossing_sign=(
                cls._ternary(payload["crossing_sign"], payload) if with_sign else None
            ),
            crossing_guard_permission=None,
            prohibitive_pedestrian_signal=None,
            visibility=visibility,
            evidence_summary=str(payload["evidence_summary"]).strip(),
            traffic_light=cls._ternary(payload["traffic_light"], payload),
        )

    @classmethod
    def _validate_markings_v3_response(cls, content: str) -> dict[str, Any]:
        payload = cls._decode_payload(content)
        # The optional listing is reasoning scaffolding and is not stored.
        payload.pop("road_markings_seen", None)
        if set(payload) != {"zebra_crossing", "visibility", "evidence_summary"}:
            raise VLMError(f"VLM returned unexpected crossing marking keys: {payload}")
        try:
            visibility = Visibility(str(payload["visibility"]).upper())
        except ValueError as error:
            raise VLMError(f"VLM returned an invalid visibility schema: {payload}") from error
        return {
            "zebra_crossing": cls._ternary(payload["zebra_crossing"], payload),
            "visibility": visibility,
            "evidence_summary": str(payload["evidence_summary"]).strip(),
        }

    @classmethod
    def _validate_controls_v3_response(cls, content: str) -> dict[str, Any]:
        payload = cls._decode_payload(content)
        payload.pop("signals_and_signs_seen", None)
        if set(payload) != {"traffic_light", "crossing_sign", "evidence_summary"}:
            raise VLMError(f"VLM returned unexpected traffic control keys: {payload}")
        return {
            "traffic_light": cls._ternary(payload["traffic_light"], payload),
            "crossing_sign": cls._ternary(payload["crossing_sign"], payload),
            "evidence_summary": str(payload["evidence_summary"]).strip(),
        }

    @classmethod
    def _validate_visibility_response(cls, content: str) -> dict[str, Any]:
        payload = cls._decode_payload(content)
        expected = {"visibility", "evidence_summary"}
        if set(payload) != expected:
            raise VLMError(f"VLM returned unexpected visibility keys: {payload}")
        try:
            visibility = Visibility(str(payload["visibility"]).upper())
        except ValueError as error:
            raise VLMError(f"VLM returned an invalid visibility schema: {payload}") from error
        return {
            "visibility": visibility,
            "evidence_summary": str(payload["evidence_summary"]).strip(),
        }

    @classmethod
    def _validate_response(cls, content: str) -> ContextAssessment:
        """Validate the legacy combined schema used by stored version 3 outputs."""

        payload = cls._decode_payload(content)
        expected_keys = {
            "marked_crosswalk",
            "permissive_pedestrian_signal",
            "authorised_crossing_sign",
            "crossing_guard_permission",
            "prohibitive_pedestrian_signal",
            "visibility",
            "evidence_summary",
        }
        if set(payload) != expected_keys:
            raise VLMError(f"VLM returned unexpected context keys: {payload}")
        try:
            assessment = ContextAssessment(
                marked_crosswalk=cls._ternary(payload["marked_crosswalk"], payload),
                permissive_pedestrian_signal=cls._ternary(
                    payload["permissive_pedestrian_signal"], payload
                ),
                authorised_crossing_sign=cls._ternary(
                    payload["authorised_crossing_sign"], payload
                ),
                crossing_guard_permission=cls._ternary(
                    payload["crossing_guard_permission"], payload
                ),
                prohibitive_pedestrian_signal=cls._ternary(
                    payload["prohibitive_pedestrian_signal"], payload
                ),
                visibility=Visibility(str(payload["visibility"]).upper()),
                evidence_summary=str(payload["evidence_summary"]).strip(),
            )
        except ValueError as error:
            raise VLMError(f"VLM returned an invalid context schema: {payload}") from error
        if (
            assessment.permissive_pedestrian_signal is Ternary.YES
            and assessment.prohibitive_pedestrian_signal is Ternary.YES
        ):
            raise VLMError(
                "VLM returned mutually exclusive pedestrian signal states: "
                "permissive and prohibitive cannot both be YES"
            )
        return assessment
