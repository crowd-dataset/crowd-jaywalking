"""Run stage 3 as its own step over the saved claims of stages 1 and 2.

Stages 1 and 2 (crowd-city crossing detection, then the VLM crossing and
infrastructure checks) save one details JSON per video and the evidence images of
every claimed person. This step reads those claims, retrieves the jaywalking rule set
of each video's country, and asks a VLM, which may differ from the stage 2 model,
whether the person is jaywalking under that law. It works on JAAD audits
(``<root>/jaad_person_audit_<split>/details``) and CROWD runs (``<root>/details``).
"""

from __future__ import annotations

import csv
import json
import re
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterator

from .jaywalking_law import JaywalkingLawJudge, LawLocation
from .models import EvidenceImage


@dataclass(frozen=True)
class Claim:
    """One person that stages 1 and 2 claim crossed without a zebra crossing or traffic light."""

    source: str
    video_id: str
    person_id: int
    evidence: tuple[EvidenceImage, ...]
    # Location from CROWD mapping metadata; None when the run has none (JAAD).
    location: LawLocation | None

    @property
    def key(self) -> str:
        return f"{self.source}/{self.video_id}/{self.person_id}"


def evidence_images(folder: Path) -> tuple[EvidenceImage, ...]:
    frames = sorted({int(re.search(r"frame_(\d+)_", p.name).group(1)) for p in folder.glob("frame_*_context.jpg")})
    return tuple(
        EvidenceImage(
            frame_index=frame,
            context_path=folder / f"frame_{frame:06d}_context.jpg",
            focus_path=folder / f"frame_{frame:06d}_focus.jpg",
            trajectory_path=folder / f"frame_{frame:06d}_trajectory.jpg",
        )
        for frame in frames
    )


def payload_location(payload: dict[str, Any]) -> LawLocation | None:
    metadata = {str(k).lower(): str(v) for k, v in payload.get("segment", {}).get("metadata", {}).items()}
    country = metadata.get("iso3") or metadata.get("country")
    if not country:
        return None
    return LawLocation(country, metadata.get("state") or None, metadata.get("locality") or metadata.get("city") or None)


def find_claims(results_root: str | Path, evidence_from: str | Path | None = None) -> Iterator[Claim]:
    """Every claimed person under a results folder, with its saved evidence.

    ``evidence_from`` points at another run of the same videos that holds the evidence
    images, for a re-decided copy that keeps only details.
    """

    root = Path(results_root)
    for details in sorted(root.glob("**/details")):
        if not details.is_dir() or "jaywalking_law" in details.parts:
            continue
        run_dir = details.parent
        relative = run_dir.relative_to(root)
        evidence_root = (Path(evidence_from) / relative if evidence_from else run_dir) / "evidence"
        for path in sorted(details.glob("*.json")):
            payload = json.loads(path.read_text(encoding="utf-8"))
            result = payload["result"]
            stem = Path(result["video_path"]).stem
            for decision in result["person_decisions"]:
                if decision["label"] != "JAYWALKING":
                    continue
                event = decision["event"]
                folder = evidence_root / stem / (
                    f"person_{decision['person_id']}_transition_"
                    f"{event['transition_start_frame']}_{event['transition_end_frame']}"
                )
                evidence = evidence_images(folder)
                if not evidence:
                    raise FileNotFoundError(f"No evidence images for {path.stem} person {decision['person_id']}: {folder}")
                yield Claim(
                    source=str(relative).replace("\\", "/") or ".",
                    video_id=payload.get("video_key") or payload.get("video_id") or path.stem,
                    person_id=int(decision["person_id"]),
                    evidence=evidence,
                    location=payload_location(payload),
                )


FIELDS = ("key", "source", "video_id", "person_id", "country", "state", "locality", "label", "reason",
          "verdicts", "verdict_sources", "evidence_summary", "legal_basis", "model")


def run_law_stage(
    judge: JaywalkingLawJudge,
    vlm: Any,
    model_id: str,
    results_root: str | Path,
    countries: list[str] | None = None,
    default_location: LawLocation | None = None,
    evidence_from: str | Path | None = None,
) -> dict[str, Any]:
    """Judge every claim and write ``<root>/jaywalking_law/``; already judged claims are skipped.

    Each claim uses its own location when it has one (CROWD); otherwise each of
    ``countries`` is applied in turn, or ``default_location``.
    """

    output = Path(results_root) / "jaywalking_law"
    output.mkdir(parents=True, exist_ok=True)
    records_path = output / "verdicts.jsonl"
    done = set()
    if records_path.exists():
        for line in records_path.read_text(encoding="utf-8").splitlines():
            record = json.loads(line)
            done.add((record["key"], record["country"]))
    with records_path.open("a", encoding="utf-8") as handle:
        for claim in find_claims(results_root, evidence_from):
            if claim.location is not None:
                locations = [claim.location]
            elif countries:
                locations = [LawLocation(country) for country in countries]
            elif default_location is not None:
                locations = [default_location]
            else:
                locations = [None]
            for location in locations:
                rule = None if location is None else judge.book.get(location.country)
                country = rule.iso if rule else (location.country if location else "")
                if (claim.key, country) in done:
                    continue
                prompts: list[str] = []

                def ask(prompt: str) -> dict[str, Any]:
                    prompts.append(prompt)
                    return vlm.evaluate_law(list(claim.evidence), prompt)

                verdict = judge.judge(location, ask)
                record = {
                    "key": claim.key,
                    "source": claim.source,
                    "video_id": claim.video_id,
                    "person_id": claim.person_id,
                    "country": country,
                    "state": location.state if location else None,
                    "locality": location.locality if location else None,
                    "label": verdict.label.value if verdict else "NO_RULE_SET",
                    "reason": verdict.reason if verdict else "No rule set for this country",
                    "verdicts": verdict.verdicts if verdict else {},
                    "verdict_sources": verdict.verdict_sources if verdict else {},
                    "evidence_summary": verdict.evidence_summary if verdict else "",
                    "legal_basis": verdict.legal_basis if verdict else "",
                    "model": model_id if prompts else "",
                    "prompt": prompts[0] if prompts else "",
                }
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                handle.flush()
                print(f"{claim.key} {country}: {record['label']} {record['verdicts']}", flush=True)

    records = [json.loads(line) for line in records_path.read_text(encoding="utf-8").splitlines()]
    with (output / "verdicts.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, extrasaction="ignore")
        writer.writeheader()
        for record in records:
            writer.writerow({**record, "verdicts": json.dumps(record["verdicts"]),
                             "verdict_sources": json.dumps(record["verdict_sources"])})
    summary = {
        "claims": len({record["key"] for record in records}),
        "rows": len(records),
        "labels": dict(Counter(record["label"] for record in records)),
        "labels_by_country": {
            country: dict(Counter(r["label"] for r in records if r["country"] == country))
            for country in sorted({r["country"] for r in records})
        },
        "rules": judge.book.source,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
