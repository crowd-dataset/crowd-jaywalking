"""Tests for running stage 3 over the saved claims of stages 1 and 2."""

import json
import re
import tempfile
import unittest
from pathlib import Path

from crowd_jaywalking.jaywalking_law import JaywalkingLawJudge
from crowd_jaywalking.law_stage import find_claims, run_law_stage

RULES = Path(__file__).resolve().parents[1] / "configs" / "jaywalking_rules.json"


def _decision(person_id, label):
    return {"person_id": person_id, "label": label,
            "event": {"transition_start_frame": 10, "transition_end_frame": 20}}


def _write_run(run_dir: Path, name: str, decisions, segment=None):
    (run_dir / "details").mkdir(parents=True, exist_ok=True)
    payload = {"video_id": name, "result": {"video_path": f"/videos/{name}.mp4", "person_decisions": decisions}}
    if segment is not None:
        payload = {"video_key": name, "segment": segment, **payload}
    (run_dir / "details" / f"{name}.json").write_text(json.dumps(payload), encoding="utf-8")
    for decision in decisions:
        folder = run_dir / "evidence" / name / f"person_{decision['person_id']}_transition_10_20"
        folder.mkdir(parents=True, exist_ok=True)
        for frame in (10, 15, 20):
            (folder / f"frame_{frame:06d}_context.jpg").write_bytes(b"")


class _VLM:
    def __init__(self):
        self.calls = 0

    def evaluate_law(self, evidence, prompt):
        self.calls += 1
        ids = re.findall(r"^([A-Z]{3}-[RT]\d+):", prompt, flags=re.M)
        return {"verdicts": {key: "YES" for key in ids}, "evidence_summary": "all met"}


class LawStageTests(unittest.TestCase):
    def test_crowd_claims_use_their_own_location_and_resume(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            segment = {"metadata": {"iso3": "CAN", "state": "ON", "locality": "Toronto"}}
            _write_run(root, "crowd_a", [_decision(1, "JAYWALKING"), _decision(2, "UNCERTAIN")], segment)
            _write_run(root, "crowd_b", [_decision(3, "JAYWALKING")], {"metadata": {"iso3": "GBR"}})
            claims = list(find_claims(root))
            self.assertEqual([(c.video_id, c.person_id) for c in claims], [("crowd_a", 1), ("crowd_b", 3)])
            self.assertEqual(len(claims[0].evidence), 3)

            vlm = _VLM()
            summary = run_law_stage(JaywalkingLawJudge(RULES), vlm, "test-model", root)
            self.assertEqual(summary["labels_by_country"], {"CAN": {"JAYWALKING": 1}, "GBR": {"NOT_JAYWALKING": 1}})
            # The United Kingdom has no offense, so only Canada needed the VLM.
            self.assertEqual(vlm.calls, 1)
            run_law_stage(JaywalkingLawJudge(RULES), vlm, "test-model", root)
            self.assertEqual(vlm.calls, 1)
            record = json.loads((root / "jaywalking_law" / "verdicts.jsonl").read_text(encoding="utf-8").splitlines()[0])
            self.assertIn("CAN-R3", record["prompt"])
            self.assertEqual(record["verdict_sources"]["CAN-R2"], "pipeline")

    def test_jaad_claims_take_the_given_countries(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            _write_run(root / "jaad_person_audit_test", "video_0001", [_decision(4, "JAYWALKING")])
            summary = run_law_stage(JaywalkingLawJudge(RULES), _VLM(), "m", root, countries=["UKR", "AUS"])
            self.assertEqual(summary["rows"], 2)
            self.assertEqual(set(summary["labels_by_country"]), {"UKR", "AUS"})

    def test_claims_without_country_are_recorded_not_judged(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            _write_run(root, "video_0002", [_decision(5, "JAYWALKING")])
            summary = run_law_stage(JaywalkingLawJudge(RULES), _VLM(), "m", root)
            self.assertEqual(summary["labels"], {"NO_RULE_SET": 1})


if __name__ == "__main__":
    unittest.main()
