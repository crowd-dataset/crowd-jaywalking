"""Independent crossing infrastructure check with street scene segmentation.

A Mask2Former model trained on Mapillary Vistas labels every pixel of the evidence
frames. A person can only be claimed as crossing without a zebra crossing or a
traffic light when this check also finds no crosswalk paint near the person's path
or on the road ahead, and no traffic light in view. It runs at full resolution,
because distant crosswalk stripes disappear when a frame is downscaled.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .models import (
    BoundingBox,
    CrossingEvent,
    EvidenceImage,
    InfrastructureMeasurement,
    TrackObservation,
)


# Mapillary Vistas v1.2 class indices used by the Hugging Face checkpoint.
CROSSWALK_CLASSES = (8, 23)  # Crosswalk - Plain, Lane Marking - Crosswalk
TRAFFIC_LIGHT_CLASS = 48
ROAD_AHEAD_TOP = 0.40  # the road ahead is the image below this fraction of its height


def near_path_mask(shape: tuple[int, int], boxes: list[BoundingBox]) -> np.ndarray:
    """Rectangle around the target's foot points, padded by a few body widths."""

    height, width = shape
    feet_x = [((box.x1 + box.x2) / 2) * width for box in boxes]
    feet_y = [box.y2 * height for box in boxes]
    box_width = max(1.0, float(np.median([(box.x2 - box.x1) * width for box in boxes])))
    box_height = max(1.0, float(np.median([(box.y2 - box.y1) * height for box in boxes])))
    x1 = int(max(0, min(feet_x) - 3 * box_width))
    x2 = int(min(width, max(feet_x) + 3 * box_width))
    y1 = int(max(0, min(feet_y) - 0.6 * box_height))
    y2 = int(min(height, max(feet_y) + 0.6 * box_height))
    mask = np.zeros(shape, dtype=bool)
    mask[y1:y2, x1:x2] = True
    return mask


def measure_segmentation(
    segmentations: list[np.ndarray],
    path_boxes: list[BoundingBox],
    thresholds: dict[str, int],
) -> InfrastructureMeasurement:
    """Reduce per frame class maps to the measurement used by the policy."""

    near = road = light = 0
    for seg in segmentations:
        crosswalk = np.isin(seg, CROSSWALK_CLASSES)
        near = max(near, int((crosswalk & near_path_mask(seg.shape, path_boxes)).sum()))
        road = max(road, int(crosswalk[int(ROAD_AHEAD_TOP * seg.shape[0]):].sum()))
        light = max(light, int((seg == TRAFFIC_LIGHT_CLASS).sum()))
    found = (
        near >= thresholds["max_crosswalk_near_path_px"]
        or road >= thresholds["max_crosswalk_road_px"]
        or light >= thresholds["max_traffic_light_px"]
    )
    return InfrastructureMeasurement(near, road, light, len(segmentations), found)


class InfrastructureSegmenter:
    """Load the segmentation model once and measure infrastructure per person."""

    def __init__(self, settings: dict[str, Any]) -> None:
        import torch
        from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation

        self.thresholds = {
            "max_crosswalk_near_path_px": int(settings["max_crosswalk_near_path_px"]),
            "max_crosswalk_road_px": int(settings["max_crosswalk_road_px"]),
            "max_traffic_light_px": int(settings["max_traffic_light_px"]),
        }
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = AutoImageProcessor.from_pretrained(
            settings["model"],
            size={
                "shortest_edge": int(settings["shortest_edge"]),
                "longest_edge": int(settings["longest_edge"]),
            },
        )
        model = Mask2FormerForUniversalSegmentation.from_pretrained(settings["model"]).eval()
        self.half = self.device == "cuda"
        self.model = (model.half() if self.half else model).to(self.device)

    def segment(self, rgb: np.ndarray) -> np.ndarray:
        import torch
        from PIL import Image

        inputs = self.processor(images=Image.fromarray(rgb), return_tensors="pt").to(self.device)
        if self.half:
            inputs["pixel_values"] = inputs["pixel_values"].half()
        with torch.inference_mode():
            outputs = self.model(**inputs)
        return self.processor.post_process_semantic_segmentation(
            outputs, target_sizes=[rgb.shape[:2]]
        )[0].cpu().numpy()

    def measure(
        self,
        video_path: str | Path,
        event: CrossingEvent,
        observations: list[TrackObservation],
        evidence: list[EvidenceImage],
    ) -> InfrastructureMeasurement:
        import cv2

        path_boxes = [
            item.box
            for item in sorted(observations, key=lambda item: item.frame_index)
            if item.class_id == 0 and item.track_id == event.person_id
        ]
        segmentations = []
        capture = cv2.VideoCapture(str(Path(video_path).resolve()))
        try:
            for frame_index in sorted({item.frame_index for item in evidence}):
                capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                ok, bgr = capture.read()
                if ok and bgr is not None:
                    segmentations.append(self.segment(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)))
        finally:
            capture.release()
        if not segmentations:
            # Nothing could be checked: treat as found, so no claim is made.
            return InfrastructureMeasurement(0, 0, 0, 0, True)
        return measure_segmentation(segmentations, path_boxes, self.thresholds)
