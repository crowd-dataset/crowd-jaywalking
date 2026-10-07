"""Convert Global_Jaywalking_Laws.pdf into the per-country rule file used by stage 3.

Run once whenever the PDF changes:

    uv run python scripts/law/build_jaywalking_rules.py data/Global_Jaywalking_Laws.pdf configs/jaywalking_rules.json

Needs ``pdftotext`` (poppler) on the PATH. Every country entry becomes one retrieval
unit: its numbered conditions, the conditions the research leaves unspecified, and
how the verdicts combine into a label.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path
import common
from custom_logger import CustomLogger
from logmod import logs

logs(show_level=common.get_configs("logger_level"), show_color=True)
logger = CustomLogger(__name__)  # use custom logger

PAGE = re.compile(r"^Page \d+ / \d+$")
CONDITION = re.compile(r"^([A-Z]{3})-([RT]\d+)$")
SECTIONS = (
    "REQUIRED OUTPUT",
    "LEGAL STATUS",
    "FINE STRUCTURE & PENALTIES",
    "KEY LEGISLATIVE DETAILS & SPATIAL NUANCES",
    "PRIMARY VERIFIED SOURCE",
)
STATUS_WORDS = {"ILLEGAL", "REGULATED", "NOT", "RULE", "SET", "NO"}


def pdf_lines(pdf: Path) -> list[str]:
    text = subprocess.run(
        ["pdftotext", "-enc", "UTF-8", str(pdf), "-"], check=True, capture_output=True
    ).stdout.decode("utf-8")
    return [line.strip() for line in text.splitlines()]


def paragraph(lines: list[str], start: int) -> tuple[str, int]:
    """Join the non-empty lines from ``start`` up to the next blank line."""

    while start < len(lines) and not lines[start]:
        start += 1
    parts = []
    while start < len(lines) and lines[start]:
        parts.append(lines[start])
        start += 1
    return " ".join(parts), start


def section(lines: list[str], title: str) -> str:
    if title not in lines:
        return ""
    index = lines.index(title) + 1
    while index < len(lines) and not lines[index]:
        index += 1
    parts = []
    # Stop at the next section title, or at the next country's heading.
    while (
        index < len(lines)
        and lines[index]
        and lines[index] not in SECTIONS
        and not re.match(r"\d{1,2} \S", lines[index])
    ):
        parts.append(lines[index])
        index += 1
    return " ".join(parts)


def parse_entry(header: list[str], body: list[str]) -> dict:
    head = " ".join(header)
    number = int(re.match(r"(\d+)", head).group(1))
    # Status and enforcement badges sit beside the heading, just after the rule set line.
    badges = " ".join([head] + body[:10])
    status = "NOT ILLEGAL" if "NOT ILLEGAL" in badges else ("REGULATED" if "REGULATED" in badges else "ILLEGAL")
    enforcement = next(
        (word for word in ("No Specific Law", "Strict", "Moderate", "Lenient") if word in badges), ""
    )
    required, triggers, unspecified = [], [], []
    iso = None
    index = 0
    while index < len(body):
        match = CONDITION.match(body[index])
        if match:
            iso = match.group(1)
            text, index = paragraph(body, index + 1)
            (required if match.group(2).startswith("R") else triggers).append(
                {"id": body_id(match), "text": text}
            )
            continue
        if body[index].startswith("•") and "NOT SPECIFIED" in " ".join(body[max(0, index - 3): index]) or (
            body[index].startswith("•") and unspecified
        ):
            text, index = paragraph(body, index)
            unspecified.append(text.lstrip("• ").strip())
            continue
        index += 1
    if iso is None:
        codes = [code for code in re.findall(r"\b([A-Z]{3})\b", head) if code not in STATUS_WORDS]
        iso = codes[0]
    name = re.sub(r"\s+", " ", re.sub(r"\b(ILLEGAL|REGULATED|NOT ILLEGAL|No Specific Law|Strict|Moderate|Lenient)\b.*", "", head))
    name = re.sub(rf"^\d+\s+|\b{iso}\b", "", name).strip()
    name = " ".join(name.split())
    decision_text = section(body, "DECISION RULE")
    if "Fixed output" in decision_text:
        decision = "fixed_not_jaywalking"
    elif "permanently UNKNOWN" in decision_text:
        decision = "fixed_insufficient_evidence"
    elif triggers:
        decision = "required_and_any_trigger"
    else:
        decision = "all_required"
    rule_set = next((line.split(":", 1)[1].strip() for line in body if line.startswith("RULE SET:")), "NO STATUTORY OFFENSE")
    return {
        "number": number,
        "iso": iso,
        "name": name,
        "legal_status": status,
        "enforcement": enforcement,
        "rule_set": rule_set,
        "required": required,
        "triggers": triggers,
        "not_specified": unspecified,
        "decision": decision,
        "decision_text": decision_text,
        "legal_basis": section(body, "LEGAL STATUS"),
        "fines": section(body, "FINE STRUCTURE & PENALTIES"),
        "details": section(body, "KEY LEGISLATIVE DETAILS & SPATIAL NUANCES"),
        "source": section(body, "PRIMARY VERIFIED SOURCE"),
    }


def body_id(match: re.Match) -> str:
    return f"{match.group(1)}-{match.group(2)}"


def parse(lines: list[str]) -> list[dict]:
    starts = [i for i, line in enumerate(lines) if line.startswith("RULE SET:") or line == "NO RULE SET — NO STATUTORY OFFENSE"]
    entries = []
    for position, start in enumerate(starts):
        page = max(i for i in range(start) if PAGE.match(lines[i]))
        header = [line for line in lines[page + 1: start] if line]
        # The entry heading starts at its number, after any section title.
        first = max(i for i, line in enumerate(header) if re.match(r"\d{1,2}\s", line))
        header = header[first:]
        end = starts[position + 1] if position + 1 < len(starts) else len(lines)
        # Drop page footers, so a section that runs onto the next page stays whole.
        body = [
            line
            for line in lines[start:end]
            if not PAGE.match(line) and not line.startswith("GLOBAL JAYWALKING LAWS")
        ]
        body = [line for index, line in enumerate(body) if line or (index and body[index - 1])]
        entries.append(parse_entry(header, body))
    return entries


def keep_supplementary(entries: list[dict], output: Path) -> None:
    """Carry the hand-added ``supplementary`` conditions of the old rule file over.

    They are not in the PDF. A rebuild stops when a condition one of them overrides
    no longer exists, so it can be fixed by hand instead of silently dropped.
    """

    if not output.is_file():
        return
    previous = json.loads(output.read_text(encoding="utf-8")).get("countries", {})
    by_iso = {entry["iso"]: entry for entry in entries}
    for iso, old in previous.items():
        items = old.get("supplementary")
        if not items or old.get("added_by_hand"):
            continue
        entry = by_iso.get(iso)
        if entry is None:
            raise SystemExit(f"{iso} has supplementary conditions but is no longer in the PDF")
        ids = {item["id"] for item in entry["required"] + entry["triggers"]}
        missing = sorted(
            {target for item in items for changes in item["overrides"].values() for target in changes} - ids
        )
        if missing:
            raise SystemExit(f"{iso} supplementary conditions override missing conditions: {missing}")
        entry["supplementary"] = items
        logger.info("Kept {} supplementary condition(s) for {}", len(items), iso)


def added_countries(output: Path) -> dict[str, dict]:
    """Rule sets written by hand for countries the PDF does not cover (marked ``added_by_hand``)."""

    if not output.is_file():
        return {}
    previous = json.loads(output.read_text(encoding="utf-8")).get("countries", {})
    return {iso: entry for iso, entry in previous.items() if entry.get("added_by_hand")}


def main() -> None:
    pdf, output = Path(sys.argv[1]), Path(sys.argv[2])
    entries = parse(pdf_lines(pdf))
    if len(entries) != 51 or len({e["iso"] for e in entries}) != 51:
        raise SystemExit(f"Expected 51 distinct countries, parsed {len(entries)}")
    output.parent.mkdir(parents=True, exist_ok=True)
    keep_supplementary(entries, output)
    added = added_countries(output)
    if added:
        clash = sorted(set(added) & {entry["iso"] for entry in entries})
        if clash:
            raise SystemExit(f"Hand-added rule sets are now in the PDF, remove them from the rule file: {clash}")
        logger.info("Kept {} hand-added rule set(s): {}", len(added), ", ".join(added))
    payload = {
        "source": pdf.name,
        "edition": "Edition 3, last updated March 20, 2026",
        "countries": {
            **{entry["iso"]: entry for entry in sorted(entries, key=lambda e: e["number"])},
            **added,
        },
    }
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    logger.info("Wrote {} rule sets to {}", len(entries), output)


if __name__ == "__main__":
    main()
