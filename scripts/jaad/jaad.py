"""Direct reader for the official JAAD 2.0 XML annotations."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import xml.etree.ElementTree as ET

from scripts.core.models import BoundingBox


VALID_SPLITS = {"train", "val", "test"}


def _attribute_text(element: ET.Element, name: str) -> str | None:
    attribute = element.find(f'./attribute[@name="{name}"]')
    if attribute is None or attribute.text is None:
        return None
    value = attribute.text.strip()
    return value or None


def _normalise_box(element: ET.Element, width: int, height: int) -> BoundingBox:
    width_value = max(float(width), 1.0)
    height_value = max(float(height), 1.0)

    def clipped(value: str | None, divisor: float) -> float:
        return min(1.0, max(0.0, float(value or 0.0) / divisor))

    return BoundingBox(
        x1=clipped(element.get("xtl"), width_value),
        y1=clipped(element.get("ytl"), height_value),
        x2=clipped(element.get("xbr"), width_value),
        y2=clipped(element.get("ybr"), height_value),
    )


def _occlusion_value(element: ET.Element) -> int:
    named = (_attribute_text(element, "occlusion") or "").lower()
    mapped = {"none": 0, "part": 1, "full": 2}
    if named in mapped:
        return mapped[named]
    return int(element.get("occluded", "0"))


def _crossing_value(element: ET.Element) -> bool | None:
    value = (_attribute_text(element, "cross") or "").lower()
    if value == "crossing":
        return True
    if value == "not-crossing":
        return False
    return None


@dataclass(frozen=True)
class JAADPedestrianTrack:
    """One independently annotated JAAD pedestrian track."""

    pedestrian_id: str
    label: str
    frames: tuple[int, ...]
    boxes: dict[int, BoundingBox]
    occlusion: dict[int, int]
    crossing: dict[int, bool | None]
    attributes: dict[str, str]

    @property
    def behaviour_annotated(self) -> bool:
        return any(value is not None for value in self.crossing.values())

    @property
    def is_crossing(self) -> bool:
        return any(value is True for value in self.crossing.values())

    @property
    def visible_frames(self) -> tuple[int, ...]:
        return tuple(frame for frame in self.frames if self.occlusion.get(frame, 0) < 2)

    @property
    def crossing_frames(self) -> tuple[int, ...]:
        return tuple(frame for frame in self.frames if self.crossing.get(frame) is True)

    def crossing_intervals(self) -> tuple[tuple[int, int], ...]:
        """Return contiguous frame intervals labelled as crossing."""

        frames = self.crossing_frames
        if not frames:
            return ()

        intervals: list[tuple[int, int]] = []
        start = frames[0]
        previous = frames[0]
        for frame in frames[1:]:
            if frame != previous + 1:
                intervals.append((start, previous))
                start = frame
            previous = frame
        intervals.append((start, previous))
        return tuple(intervals)


@dataclass(frozen=True)
class JAADVideoAnnotations:
    """All validation annotations required for one JAAD video."""

    video_id: str
    num_frames: int
    width: int
    height: int
    tracks: dict[str, JAADPedestrianTrack]
    traffic: dict[int, dict[str, str]]
    road_type: str

    @property
    def behaviour_tracks(self) -> tuple[JAADPedestrianTrack, ...]:
        return tuple(
            track
            for track in self.tracks.values()
            if track.label == "pedestrian" and track.behaviour_annotated
        )


class JAADDataset:
    """Access the official JAAD repository checkout and downloaded clips."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).resolve()
        self.annotations_dir = self.root / "annotations"
        self.attributes_dir = self.root / "annotations_attributes"
        self.traffic_dir = self.root / "annotations_traffic"
        self.clips_dir = self.root / "JAAD_clips"
        self.split_dir = self.root / "split_ids" / "default"

        required = [self.annotations_dir, self.attributes_dir, self.traffic_dir, self.split_dir]
        missing = [str(path) for path in required if not path.is_dir()]
        if missing:
            raise FileNotFoundError(
                "The JAAD repository is incomplete. Missing: " + ", ".join(missing)
            )

    def video_ids(self, split: str) -> list[str]:
        normalised = str(split).strip().lower()
        if normalised not in VALID_SPLITS:
            raise ValueError("JAAD split must be one of: train, val, test")
        split_path = self.split_dir / f"{normalised}.txt"
        if not split_path.is_file():
            raise FileNotFoundError(f"JAAD split file not found: {split_path}")
        return [line.strip() for line in split_path.read_text(encoding="utf-8").splitlines() if line.strip()]

    def clip_path(self, video_id: str) -> Path:
        path = self.clips_dir / f"{video_id}.mp4"
        if not path.is_file():
            raise FileNotFoundError(
                f"JAAD clip not found: {path}. Run scripts/jaad/download_jaad.ps1 first."
            )
        return path

    def load_video(self, video_id: str) -> JAADVideoAnnotations:
        annotation_path = self.annotations_dir / f"{video_id}.xml"
        if not annotation_path.is_file():
            raise FileNotFoundError(f"JAAD annotation not found: {annotation_path}")

        root = ET.parse(annotation_path).getroot()
        num_frames = int(root.findtext("./meta/task/size", default="0"))
        width = int(root.findtext("./meta/task/original_size/width", default="0"))
        height = int(root.findtext("./meta/task/original_size/height", default="0"))
        if num_frames <= 0 or width <= 0 or height <= 0:
            raise ValueError(f"Invalid JAAD video metadata: {annotation_path}")

        pedestrian_attributes = self._load_attributes(video_id)
        tracks: dict[str, JAADPedestrianTrack] = {}
        for track_element in root.findall("./track"):
            box_elements = track_element.findall("./box")
            if not box_elements:
                continue
            pedestrian_id = _attribute_text(box_elements[0], "id")
            if not pedestrian_id:
                continue

            boxes: dict[int, BoundingBox] = {}
            occlusion: dict[int, int] = {}
            crossing: dict[int, bool | None] = {}
            for box_element in box_elements:
                if box_element.get("outside", "0") == "1":
                    continue
                frame = int(box_element.get("frame", "0"))
                boxes[frame] = _normalise_box(box_element, width, height)
                occlusion[frame] = _occlusion_value(box_element)
                crossing[frame] = _crossing_value(box_element)

            frames = tuple(sorted(boxes))
            if not frames:
                continue
            tracks[pedestrian_id] = JAADPedestrianTrack(
                pedestrian_id=pedestrian_id,
                label=str(track_element.get("label", "")),
                frames=frames,
                boxes=boxes,
                occlusion=occlusion,
                crossing=crossing,
                attributes=pedestrian_attributes.get(pedestrian_id, {}),
            )

        traffic, road_type = self._load_traffic(video_id)
        return JAADVideoAnnotations(
            video_id=video_id,
            num_frames=num_frames,
            width=width,
            height=height,
            tracks=tracks,
            traffic=traffic,
            road_type=road_type,
        )

    def _load_attributes(self, video_id: str) -> dict[str, dict[str, str]]:
        path = self.attributes_dir / f"{video_id}_attributes.xml"
        if not path.is_file():
            return {}
        root = ET.parse(path).getroot()
        result: dict[str, dict[str, str]] = {}
        for pedestrian in root.findall("./pedestrian"):
            pedestrian_id = str(pedestrian.get("id", "")).strip()
            if pedestrian_id:
                result[pedestrian_id] = {
                    str(key): str(value)
                    for key, value in pedestrian.attrib.items()
                    if key not in {"id", "old_id"}
                }
        return result

    def _load_traffic(self, video_id: str) -> tuple[dict[int, dict[str, str]], str]:
        path = self.traffic_dir / f"{video_id}_traffic.xml"
        if not path.is_file():
            return {}, ""
        root = ET.parse(path).getroot()
        road_type = str(root.findtext("./road_type", default=""))
        frames: dict[int, dict[str, str]] = {}
        for frame in root.findall("./frame"):
            frame_id = int(frame.get("id", "0"))
            frames[frame_id] = {
                "ped_crossing": str(frame.get("ped_crossing", "")),
                "ped_sign": str(frame.get("ped_sign", "")),
                "stop_sign": str(frame.get("stop_sign", "")),
                "traffic_light": str(frame.get("traffic_light", "")),
            }
        return frames, road_type
