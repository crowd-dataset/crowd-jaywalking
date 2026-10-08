"""Build the README figures in docs/images from a finished JAAD audit.

Needs the evidence images saved by scripts/jaad/run_jaad_person_audit.py (results folder from
config) and its re-decided claims. The JAAD clips are CC BY 4.0 (Rasouli et al.).

    uv run python .\\scripts\\docs\\make_readme_figures.py
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

import common  # noqa: E402
from scripts.core.config import ProjectConfig  # noqa: E402
from custom_logger import CustomLogger  # noqa: E402
from logmod import logs  # noqa: E402

logs(show_level=common.get_configs("logger_level"), show_color=True)
logger = CustomLogger(__name__)  # use custom logger

OUTPUT = Path(common.root_dir) / "docs" / "images"
WIDTH = 1600
FONT = "C:/Windows/Fonts/arial.ttf"
BOLD = "C:/Windows/Fonts/arialbd.ttf"


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    try:
        return ImageFont.truetype(BOLD if bold else FONT, size)
    except OSError:
        return ImageFont.load_default()


def evidence_folder(run: Path, split: str, video: str, person: int) -> Path:
    return next((run / f"jaad_person_audit_{split}" / "evidence" / video).glob(f"person_{person}_transition_*"))


def frames(folder: Path) -> list[int]:
    return sorted({int(re.search(r"frame_(\d+)_", p.name).group(1)) for p in folder.glob("frame_*_context.jpg")})


def fit(image: Image.Image, width: int) -> Image.Image:
    return image.resize((width, round(image.height * width / image.width)))


def caption_block(lines: list[tuple[str, str]], width: int) -> Image.Image:
    """White strip with (bold label, text) lines; long text wraps under itself."""

    bold, plain = font(26, bold=True), font(26)
    measure = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    layout = []
    for label, text in lines:
        offset = measure.textlength(label, font=bold) + 12
        wrapped, current = [], ""
        for word in text.split():
            trial = f"{current} {word}".strip()
            if measure.textlength(trial, font=plain) > width - 40 - offset and current:
                wrapped.append(current)
                current = word
            else:
                current = trial
        layout.append((label, offset, wrapped + [current]))
    block = Image.new("RGB", (width, 24 + 36 * sum(len(w) for _, _, w in layout)), "white")
    draw = ImageDraw.Draw(block)
    y = 14
    for label, offset, wrapped in layout:
        draw.text((20, y), label, fill=(20, 20, 20), font=bold)
        for line in wrapped:
            draw.text((20 + offset, y), line, fill=(60, 60, 60), font=plain)
            y += 36
    return block


def stack(parts: list[Image.Image]) -> Image.Image:
    out = Image.new("RGB", (max(p.width for p in parts), sum(p.height for p in parts)), "white")
    y = 0
    for part in parts:
        out.paste(part, (0, y))
        y += part.height
    return out


def row(images: list[Image.Image], width: int, gap: int = 8) -> Image.Image:
    tile = (width - gap * (len(images) - 1)) // len(images)
    scaled = [fit(image, tile) for image in images]
    out = Image.new("RGB", (width, max(s.height for s in scaled)), "white")
    for index, image in enumerate(scaled):
        out.paste(image, (index * (tile + gap), 0))
    return out


def row_same_height(images: list[Image.Image], width: int, gap: int = 8) -> Image.Image:
    """Images side by side at one common height, filling ``width``."""

    ratios = [image.width / image.height for image in images]
    height = round((width - gap * (len(images) - 1)) / sum(ratios))
    scaled = [image.resize((round(height * ratio), height)) for image, ratio in zip(images, ratios)]
    out = Image.new("RGB", (width, height), "white")
    x = 0
    for image in scaled:
        out.paste(image, (x, 0))
        x += image.width + gap
    return out


def labelled(image: Image.Image, label: str) -> Image.Image:
    image = image.copy()
    draw = ImageDraw.Draw(image)
    size = max(18, image.width // 28)
    box = draw.textbbox((0, 0), label, font=font(size, bold=True))
    draw.rectangle((0, 0, box[2] + 20, box[3] + 14), fill=(0, 0, 0))
    draw.text((10, 5), label, fill="white", font=font(size, bold=True))
    return image


def trajectory_strip(folder: Path, picks: int = 3) -> Image.Image:
    available = frames(folder)
    chosen = [available[round(i * (len(available) - 1) / (picks - 1))] for i in range(picks)]
    images = [Image.open(folder / f"frame_{frame:06d}_trajectory.jpg").convert("RGB") for frame in chosen]
    return row(images, WIDTH)


def decision(run: Path, split: str, video: str, person: int) -> tuple[dict, dict | None]:
    result = json.loads((run / f"jaad_person_audit_{split}" / "details" / f"{video}.json").read_text(encoding="utf-8"))["result"]
    check = next((c for c in result["crossing_checks"] if c["person_id"] == person), None)
    person_decision = next((d for d in result["person_decisions"] if d["person_id"] == person), None)
    return check, person_decision


def save(image: Image.Image, name: str) -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    image.save(OUTPUT / name, quality=82, optimize=True)
    logger.info("Saved {}", OUTPUT / name)


def claim_figure(run: Path) -> None:
    split, video, person = "test", "video_0093", 5
    folder = evidence_folder(run, split, video, person)
    check, person_decision = decision(run, split, video, person)
    save(stack([
        trajectory_strip(folder),
        caption_block([
            ("Stage 1 (crowd-city):", f"{video}, person {person} passes in front of the camera with the feet on the road: a potential crossing."),
            ("Stage 2 crossing check:", f"{check['answer']}. {check['evidence_summary']}"),
            ("Stage 2 infrastructure:", person_decision["context"]["evidence_summary"]),
            ("Result:", "claim. JAAD: crossing, no pedestrian crossing and no traffic light annotated. Correct."),
        ], WIDTH),
    ]), "claim_example.jpg")


def views_figure(run: Path) -> None:
    folder = evidence_folder(run, "test", "video_0084", 2)
    frame = frames(folder)[len(frames(folder)) // 2]
    name = lambda view: folder / f"frame_{frame:06d}_{view}.jpg"  # noqa: E731
    top = row([
        labelled(Image.open(name("context")).convert("RGB"), "full scene"),
        labelled(Image.open(name("trajectory")).convert("RGB"), "trajectory map"),
    ], WIDTH)
    middle = row_same_height([
        labelled(Image.open(name("focus")).convert("RGB"), "target detail"),
        labelled(Image.open(name("crossing")).convert("RGB"), "crossing close-up"),
        labelled(Image.open(name("lower_road")).convert("RGB"), "road ahead"),
    ], WIDTH)
    tiles = [Image.open(t).convert("RGB") for t in sorted(folder.glob(f"frame_{frame:06d}_tile_*.jpg"))]
    bottom = stack([
        row_same_height([labelled(t, f"control tile {i + 1}") for i, t in enumerate(tiles[:3])], WIDTH),
        Image.new("RGB", (WIDTH, 8), "white"),
        row_same_height([labelled(t, f"control tile {i + 4}") for i, t in enumerate(tiles[3:])], WIDTH),
    ])
    save(stack([top, Image.new("RGB", (WIDTH, 8), "white"), middle, Image.new("RGB", (WIDTH, 8), "white"), bottom,
                caption_block([("One evidence moment.", " Six such moments are sampled around the crossing; "
                                "each VLM call sees four of them.")], WIDTH)]), "evidence_views.jpg")


def rejected_figure(run: Path) -> None:
    split, video, person = "test", "video_0150", 25
    check, _ = decision(run, split, video, person)
    save(stack([
        trajectory_strip(evidence_folder(run, split, video, person)),
        caption_block([
            ("Stage 1 (crowd-city):", f"{video}, person {person} passes in front of the camera: a potential crossing."),
            ("Stage 2 crossing check:", f"{check['answer']}. {check['evidence_summary']}"),
            ("Result:", "no claim. JAAD: a bystander, not crossing."),
        ], WIDTH),
    ]), "rejected_example.jpg")


def infrastructure_figure(run: Path) -> None:
    split, video, person = "test", "video_0046", 1
    check, person_decision = decision(run, split, video, person)
    summary = person_decision["context"]["evidence_summary"]
    save(stack([
        trajectory_strip(evidence_folder(run, split, video, person)),
        caption_block([
            ("Stage 2 crossing check:", f"{check['answer']}. The person crosses in front of the car."),
            ("Stage 2 infrastructure:", summary[: summary.find(".") + 1]),
            ("Result:", "no claim (COMPLIANT). JAAD: pedestrian crossing annotated."),
        ], WIDTH),
    ]), "infrastructure_example.jpg")


def funnel_figure() -> None:
    stages = [
        ("JAAD crossers with no zebra\ncrossing and no traffic light", 111),
        ("proposed by stage 1\n(crowd-city detector)", 40),
        ("confirmed as crossing by\nthe VLM crossing check", 29),
        ("claimed after the VLM\ninfrastructure check", 22),
    ]
    fig, ax = plt.subplots(figsize=(9, 3.6), dpi=150)
    colours = ["#b0b7c3", "#7d9cc8", "#4f7fc0", "#2a9d5c"]
    for index, ((label, value), colour) in enumerate(zip(stages, colours)):
        ax.barh(index, value, color=colour)
        note = " (+1 claim JAAD does not list: video_0092, correct on review)" if index == 3 else ""
        ax.text(value + 1.5, index, f"{value}{note}", va="center", fontsize=10)
    ax.set_yticks(range(len(stages)), [label for label, _ in stages], fontsize=10)
    ax.invert_yaxis()
    ax.set_xlim(0, 125)
    ax.set_xlabel("JAAD pedestrians (train, val, and test splits)")
    ax.set_title("No wrong claims: 23 of 23 correct (22 confirmed by JAAD, 1 by manual review)", fontsize=11)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT / "evaluation_funnel.png")
    logger.info("Saved {}", OUTPUT / "evaluation_funnel.png")


def stage1_chart() -> None:
    """Precision and recall of crowd-city's crossing decision alone, from the saved evaluation."""

    summary = json.loads((ProjectConfig.load().path("results") / "crowd_city_stage1" / "summary.json").read_text(encoding="utf-8"))
    rules = [("road_crossing", "road_crossing (default)", "#2a9d5c"), ("detector", "detector (older, box only)", "#7d9cc8")]
    metrics = [("precision", "Precision\n(accepted tracks that cross)"), ("recall", "Recall\n(all JAAD crossers)"),
               ("eligible_recall", "Recall\n(crossers with no zebra, no light)")]
    fig, ax = plt.subplots(figsize=(9, 3.8), dpi=150)
    width = 0.36
    for offset, (key, label, colour) in enumerate(rules):
        values = [100 * summary[key][metric] for metric, _ in metrics]
        bars = ax.bar([i + (offset - 0.5) * width for i in range(len(metrics))], values, width, color=colour, label=label)
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, value + 1.5, f"{value:.1f}%", ha="center", fontsize=9)
    ax.set_xticks(range(len(metrics)), [label for _, label in metrics], fontsize=9)
    ax.set_ylim(0, 112)
    ax.set_ylabel("%")
    ax.set_title(f"crowd-city's crossing decision on JAAD ({summary['videos']} videos, no VLM)", fontsize=11)
    ax.legend(frameon=False, fontsize=9, loc="upper right", bbox_to_anchor=(1.0, 0.92))
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT / "stage1_crowd_city.png")
    logger.info("Saved {}", OUTPUT / "stage1_crowd_city.png")


def stage1_example_figure() -> None:
    """One track the road_crossing rule accepts and one it rejects, over SegFormer's road mask."""

    import cv2
    import numpy as np

    from scripts.core.tracking import load_observations_csv
    from scripts.crossing.crowd_city_road_rule import CrowdCityRoadCrossingDetector
    from scripts.jaad.jaad_person_audit import JAADPersonAudit

    config = ProjectConfig.load()
    video = "video_0053"
    audit = JAADPersonAudit(config, "test")
    detector = CrowdCityRoadCrossingDetector(config.crowd_city_settings())
    clip = audit.dataset.clip_path(video)
    observations = load_observations_csv(audit.tracks_dir / f"{video}.csv")
    result = detector.detect(observations, 30.0, clip)
    reasons = {c.person_id: (c.track_features or {}).get("rejection") for c in result.classifications}
    joined = detector._joined([o for o in observations if o.confidence >= 0.7], 30.0)
    segmenter = detector.reader._segmenter()
    capture = cv2.VideoCapture(str(clip))

    def panel(event, accepted: bool) -> Image.Image:
        frame_index = (event.transition_start_frame + event.transition_end_frame) // 2
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, image = capture.read()
        if not ok:
            raise RuntimeError(f"Cannot read frame {frame_index} of {clip}")
        small = cv2.cvtColor(cv2.resize(image, (segmenter.input_width, segmenter.input_height), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
        labels, _ = segmenter.segment(small[None])
        road = cv2.resize((labels[0] == 0).astype(np.uint8), (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST) > 0
        shaded = image.copy()
        shaded[road] = (0.62 * shaded[road] + 0.38 * np.array([255, 120, 40])).astype(np.uint8)
        box = next(o.box for o in joined if o.track_id == event.person_id and o.frame_index == frame_index and o.class_id == 0)
        colour = (60, 170, 40) if accepted else (40, 40, 220)
        height, width = image.shape[:2]
        cv2.rectangle(shaded, (int(box.x1 * width), int(box.y1 * height)), (int(box.x2 * width), int(box.y2 * height)), colour, 5)
        return Image.fromarray(cv2.cvtColor(shaded, cv2.COLOR_BGR2RGB))

    accepted = result.valid_events[0]
    rejected = next(e for e in result.rejected_events if reasons.get(e.person_id))
    left = labelled(panel(accepted, True), f"accepted: person {accepted.person_id}")
    right = labelled(panel(rejected, False), f"rejected: person {rejected.person_id}")
    capture.release()
    save(stack([
        row_same_height([left, right], WIDTH),
        caption_block([
            ("Road (blue):", "SegFormer's road surface, read under the feet at sampled frames."),
            ("Accepted:", f"{video}, person {accepted.person_id} passes in front of the camera with the feet on the road at a walking pace."),
            ("Rejected:", f"{video}, person {rejected.person_id} passes the box rules but is dropped: {reasons[rejected.person_id]}."),
        ], WIDTH),
    ]), "stage1_example.jpg")


def main() -> None:
    run = ProjectConfig.load().path("results")
    claim_figure(run)
    views_figure(run)
    rejected_figure(run)
    infrastructure_figure(run)
    funnel_figure()
    stage1_chart()
    stage1_example_figure()


if __name__ == "__main__":
    main()
