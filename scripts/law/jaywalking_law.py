"""Stage 3: judge a confirmed crossing against the jaywalking law of its country.

Stages 1 and 2 establish that a person crossed the road in front of the camera car
without a zebra crossing and without a traffic light. Stage 3 retrieves the rule set
of the country the video was recorded in (``configs/jaywalking_rules.json``, built
from Global_Jaywalking_Laws.pdf by ``scripts/law/build_jaywalking_rules.py``), asks the VLM for
a verdict on each numbered condition, and combines the verdicts into a label with
the rule set's own decision rule. The label is computed here, never taken from the
model, so every result is traceable to the per-condition verdicts.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any


class LawLabel(str, Enum):
    JAYWALKING = "JAYWALKING"
    NOT_JAYWALKING = "NOT_JAYWALKING"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"


ANSWERS = ("YES", "NO", "UNKNOWN", "N/A")

# Names and codes CROWD metadata or users may give instead of the ISO 3166 alpha-3 code.
COUNTRY_ALIASES = {
    "US": "USA", "UNITED STATES": "USA", "UNITED STATES OF AMERICA": "USA", "AMERICA": "USA",
    "UK": "GBR", "GB": "GBR", "GREAT BRITAIN": "GBR", "ENGLAND": "GBR", "SCOTLAND": "GBR", "WALES": "GBR",
    "NORTHERN IRELAND": "GBR", "UNITED KINGDOM": "GBR",
    "TURKEY": "TUR", "TURKIYE": "TUR", "TÜRKIYE": "TUR",
    "VIETNAM": "VNM", "VIET NAM": "VNM",
    "KOREA": "KOR", "SOUTH KOREA": "KOR", "REPUBLIC OF KOREA": "KOR", "KOREA, REPUBLIC OF": "KOR",
    "RUSSIAN FEDERATION": "RUS", "IRAN, ISLAMIC REPUBLIC OF": "IRN", "CZECHIA": "CZE",
    "HOLLAND": "NLD", "THE NETHERLANDS": "NLD",
}

# USA-R3 and USA-R4 depend on the state or city, not on the images. The rule set names
# Texas and Florida as keeping an enforceable prohibition, New York City as decriminalised,
# and gives California its own branch; any other place is unresolved and stays UNKNOWN.
US_ENFORCING_STATES = {"TX", "TEXAS", "FL", "FLORIDA"}
US_CALIFORNIA = {"CA", "CALIFORNIA"}
US_DECRIMINALISED_CITIES = {("NY", "NEW YORK"), ("NY", "NEW YORK CITY"), ("NEW YORK", "NEW YORK"), ("NEW YORK", "NEW YORK CITY")}


@dataclass(frozen=True)
class LawLocation:
    """Where the video was recorded, from CROWD mapping metadata or configuration."""

    country: str
    state: str | None = None
    locality: str | None = None


@dataclass(frozen=True)
class RuleSet:
    iso: str
    name: str
    legal_status: str
    enforcement: str
    rule_set: str
    required: tuple[dict[str, str], ...]
    triggers: tuple[dict[str, str], ...]
    not_specified: tuple[str, ...]
    decision: str
    decision_text: str
    legal_basis: str

    @property
    def condition_ids(self) -> list[str]:
        return [item["id"] for item in self.required + self.triggers]


@dataclass(frozen=True)
class LawVerdict:
    """Stage 3 result for one person."""

    country: str
    country_name: str
    label: LawLabel
    reason: str
    verdicts: dict[str, str]
    # Where each verdict came from: "vlm", "pipeline", or "location".
    verdict_sources: dict[str, str] = field(default_factory=dict)
    evidence_summary: str = ""
    legal_basis: str = ""


class RuleBook:
    """All country rule sets, retrieved one country at a time."""

    def __init__(self, path: str | Path) -> None:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        self.source = payload.get("source", "")
        self.rules = {
            iso: RuleSet(
                iso=iso,
                name=entry["name"],
                legal_status=entry["legal_status"],
                enforcement=entry["enforcement"],
                rule_set=entry["rule_set"],
                required=tuple(entry["required"]),
                triggers=tuple(entry["triggers"]),
                not_specified=tuple(entry["not_specified"]),
                decision=entry["decision"],
                decision_text=entry["decision_text"],
                legal_basis=entry["legal_basis"],
            )
            for iso, entry in payload["countries"].items()
        }
        self._names = {rule.name.upper(): iso for iso, rule in self.rules.items()}

    def resolve(self, country: str) -> str | None:
        """ISO alpha-3 code for a country code or name, or None when unknown."""

        key = " ".join(str(country or "").strip().upper().split())
        if not key:
            return None
        if key in self.rules:
            return key
        key = COUNTRY_ALIASES.get(key, key)
        return key if key in self.rules else self._names.get(key)

    def get(self, country: str) -> RuleSet | None:
        iso = self.resolve(country)
        return None if iso is None else self.rules[iso]


def prefilled_verdicts(rule: RuleSet, location: LawLocation) -> dict[str, tuple[str, str]]:
    """Verdicts that do not come from the VLM, as {condition id: (answer, source)}.

    R2 (the person is crossing, or has entered, the roadway) is what stages 1 and 2
    have already established; stage 3 only runs on such people. The United States
    jurisdiction conditions depend on the location alone.
    """

    known: dict[str, tuple[str, str]] = {}
    for item in rule.required:
        if item["id"].endswith("-R2") and "crossing, or has entered, the roadway" in item["text"]:
            known[item["id"]] = ("YES", "pipeline")
    if rule.iso == "USA":
        state = (location.state or "").strip().upper()
        city = (location.locality or "").strip().upper()
        if (state, city) in US_DECRIMINALISED_CITIES:
            known["USA-R3"] = ("NO", "location")
        elif state in US_ENFORCING_STATES or state in US_CALIFORNIA:
            known["USA-R3"] = ("YES", "location")
        else:
            known["USA-R3"] = ("UNKNOWN", "location")
        if state not in US_CALIFORNIA:
            known["USA-R4"] = ("N/A", "location")
    return known


def build_prompt(rule: RuleSet, location: LawLocation, open_ids: list[str]) -> str:
    """The VLM prompt for the conditions it must judge, in the rule set's own words."""

    place = ", ".join(part for part in (location.locality, location.state, rule.name) if part)
    conditions = [item for item in rule.required + rule.triggers if item["id"] in open_ids]
    lines = [
        "You are inspecting one tracked pedestrian in a vehicle dashcam video. The target pedestrian has a RED box. "
        "TRAJECTORY MAP views show the whole scene with the target's foot point path as a YELLOW line ending in an arrow; "
        "FULL SCENE views show the same moments without drawings. The images are chronological.",
        "",
        f"The video was recorded in {place}. Already established: the target crosses the road in front of the camera "
        "car, and no zebra crossing and no traffic light was found near the target's crossing path.",
        "",
        f"Evaluate the following conditions from the {rule.name} jaywalking rule set ({rule.rule_set.lower()}). "
        "For each condition answer YES, NO, or UNKNOWN. Answer UNKNOWN whenever the images do not show the evidence "
        "needed to decide. Do not infer unstated distances or thresholds; judge a stated distance only when it can "
        "be estimated from the images. Answer N/A only where the condition text itself allows it.",
        "",
    ]
    lines += [f"{item['id']}: {item['text']}" for item in conditions]
    if rule.not_specified:
        lines += ["", "Not specified by the rule set, do not infer:"] + [f"- {text}" for text in rule.not_specified]
    example = ", ".join(f'"{condition_id}": "YES|NO|UNKNOWN"' for condition_id in open_ids)
    lines += [
        "",
        "Return one JSON object with exactly these keys:",
        "{",
        f'  "verdicts": {{{example}}},',
        '  "evidence_summary": "one or two short sentences citing condition IDs"',
        "}",
        "",
        "Output JSON only.",
    ]
    return "\n".join(lines)


def parse_verdicts(payload: dict[str, Any], open_ids: list[str]) -> tuple[dict[str, str], str]:
    """Validate a VLM response: one allowed answer for each asked condition."""

    if set(payload) != {"verdicts", "evidence_summary"} or not isinstance(payload["verdicts"], dict):
        raise ValueError(f"Unexpected stage 3 response keys: {payload}")
    verdicts = {str(key).strip().upper(): str(value).strip().upper() for key, value in payload["verdicts"].items()}
    if set(verdicts) != set(open_ids):
        raise ValueError(f"Stage 3 response answers {sorted(verdicts)}, expected {open_ids}")
    bad = {key: value for key, value in verdicts.items() if value not in ANSWERS}
    if bad:
        raise ValueError(f"Stage 3 response has invalid answers: {bad}")
    return verdicts, str(payload["evidence_summary"]).strip()


def decide(rule: RuleSet, verdicts: dict[str, str]) -> tuple[LawLabel, str]:
    """Combine per-condition verdicts with the rule set's decision rule.

    N/A counts as satisfied: it is allowed only where a condition is a branch that
    does not apply to this location.
    """

    if rule.decision == "fixed_not_jaywalking":
        return LawLabel.NOT_JAYWALKING, f"No statutory jaywalking offense in {rule.name}"
    if rule.decision == "fixed_insufficient_evidence":
        return (
            LawLabel.INSUFFICIENT_EVIDENCE,
            f"The rule set for {rule.name} does not specify the operative conditions",
        )
    required = [item["id"] for item in rule.required]
    triggers = [item["id"] for item in rule.triggers]
    failed = [key for key in required if verdicts[key] == "NO"]
    if failed:
        return LawLabel.NOT_JAYWALKING, f"Required condition not met: {', '.join(failed)}"
    if triggers and all(verdicts[key] == "NO" for key in triggers):
        return LawLabel.NOT_JAYWALKING, f"No trigger condition met: {', '.join(triggers)}"
    unknown = [key for key in required if verdicts[key] == "UNKNOWN"]
    if unknown:
        return LawLabel.INSUFFICIENT_EVIDENCE, f"Undecided required condition: {', '.join(unknown)}"
    if triggers:
        met = [key for key in triggers if verdicts[key] == "YES"]
        if not met:
            return LawLabel.INSUFFICIENT_EVIDENCE, f"No trigger condition decided: {', '.join(triggers)}"
        return LawLabel.JAYWALKING, f"All required conditions and trigger {', '.join(met)} met"
    return LawLabel.JAYWALKING, "All required conditions met"


class JaywalkingLawJudge:
    """Retrieve the country's rule set, ask the VLM, and decide the label."""

    def __init__(self, rules_path: str | Path) -> None:
        self.book = RuleBook(rules_path)

    def judge(self, location: LawLocation | None, ask_vlm) -> LawVerdict | None:
        """Stage 3 for one person; ``ask_vlm(prompt)`` returns the parsed JSON payload.

        Returns None when the country has no rule set, so no legal label is given.
        """

        rule = None if location is None else self.book.get(location.country)
        if rule is None:
            return None
        known = prefilled_verdicts(rule, location)
        open_ids = [key for key in rule.condition_ids if key not in known]
        verdicts = {key: answer for key, (answer, _) in known.items()}
        sources = {key: source for key, (_, source) in known.items()}
        summary = ""
        if rule.decision in ("all_required", "required_and_any_trigger") and open_ids:
            answers, summary = parse_verdicts(ask_vlm(build_prompt(rule, location, open_ids)), open_ids)
            texts = {item["id"]: item["text"] for item in rule.required + rule.triggers}
            # N/A only where the condition itself is a branch that allows it.
            answers = {
                key: ("UNKNOWN" if value == "N/A" and "N/A" not in texts[key] else value)
                for key, value in answers.items()
            }
            verdicts.update(answers)
            sources.update({key: "vlm" for key in answers})
        label, reason = decide(rule, verdicts)
        return LawVerdict(
            country=rule.iso,
            country_name=rule.name,
            label=label,
            reason=reason,
            verdicts={key: verdicts[key] for key in rule.condition_ids if key in verdicts},
            verdict_sources=sources,
            evidence_summary=summary,
            legal_basis=rule.legal_basis,
        )
