"""Tests for stage 3, the country specific jaywalking rule sets."""

import unittest
from pathlib import Path

from scripts.law.jaywalking_law import (
    JaywalkingLawJudge,
    LawLabel,
    LawLocation,
    RuleBook,
    build_prompt,
    decide,
    parse_verdicts,
    prefilled_verdicts,
)

RULES = Path(__file__).resolve().parents[1] / "configs" / "jaywalking_rules.json"


class RuleBookTests(unittest.TestCase):
    def setUp(self):
        self.book = RuleBook(RULES)

    def test_all_countries_and_classes_are_present(self):
        self.assertEqual(len(self.book.rules), 51)
        classes = [rule.rule_set for rule in self.book.rules.values()]
        self.assertEqual(
            (classes.count("STATUTORY OFFENSE"), classes.count("CODIFIED DUTY"), classes.count("NO STATUTORY OFFENSE")),
            (36, 11, 4),
        )

    def test_country_codes_and_names_resolve(self):
        for value, iso in (("CAN", "CAN"), ("Canada", "CAN"), ("United States", "USA"), ("UK", "GBR"),
                           ("Turkey", "TUR"), ("Viet Nam", "VNM"), ("south korea", "KOR")):
            self.assertEqual(self.book.resolve(value), iso)
        self.assertIsNone(self.book.resolve("Argentina"))
        self.assertIsNone(self.book.resolve(""))

    def test_rule_texts_carry_their_distances(self):
        self.assertIn("20 metres", self.book.rules["AUS"].required[2]["text"])
        self.assertIn("50 metres", self.book.rules["SGP"].required[2]["text"])
        self.assertEqual([item["id"] for item in self.book.rules["UKR"].triggers], ["UKR-T1", "UKR-T2"])


class DecisionTests(unittest.TestCase):
    def setUp(self):
        self.book = RuleBook(RULES)

    def test_all_required(self):
        aus = self.book.rules["AUS"]
        yes = {key: "YES" for key in aus.condition_ids}
        self.assertEqual(decide(aus, yes)[0], LawLabel.JAYWALKING)
        # No crossing within 20 m: crossing there is lawful in Australia.
        self.assertEqual(decide(aus, {**yes, "AUS-R3": "NO"})[0], LawLabel.NOT_JAYWALKING)
        self.assertEqual(decide(aus, {**yes, "AUS-R3": "UNKNOWN"})[0], LawLabel.INSUFFICIENT_EVIDENCE)
        # A failed condition decides even when another is unknown.
        self.assertEqual(decide(aus, {**yes, "AUS-R1": "NO", "AUS-R3": "UNKNOWN"})[0], LawLabel.NOT_JAYWALKING)

    def test_required_and_any_trigger(self):
        ukr = self.book.rules["UKR"]
        base = {"UKR-R1": "YES", "UKR-R2": "YES"}
        self.assertEqual(decide(ukr, {**base, "UKR-T1": "YES", "UKR-T2": "NO"})[0], LawLabel.JAYWALKING)
        self.assertEqual(decide(ukr, {**base, "UKR-T1": "NO", "UKR-T2": "NO"})[0], LawLabel.NOT_JAYWALKING)
        self.assertEqual(decide(ukr, {**base, "UKR-T1": "UNKNOWN", "UKR-T2": "NO"})[0], LawLabel.INSUFFICIENT_EVIDENCE)

    def test_fixed_rule_sets(self):
        self.assertEqual(decide(self.book.rules["GBR"], {})[0], LawLabel.NOT_JAYWALKING)
        self.assertEqual(decide(self.book.rules["IND"], {})[0], LawLabel.INSUFFICIENT_EVIDENCE)


class JudgeTests(unittest.TestCase):
    def setUp(self):
        self.judge = JaywalkingLawJudge(RULES)
        self.asked = []

    def _vlm(self, answers):
        def ask(prompt):
            self.asked.append(prompt)
            return {"verdicts": answers, "evidence_summary": "test"}
        return ask

    def test_crossing_condition_comes_from_the_pipeline(self):
        verdict = self.judge.judge(LawLocation("Canada", "ON", "Toronto"), self._vlm({"CAN-R1": "YES", "CAN-R3": "YES", "CAN-X1": "NO"}))
        self.assertEqual(verdict.label, LawLabel.JAYWALKING)
        self.assertEqual(verdict.verdict_sources["CAN-R2"], "pipeline")
        self.assertNotIn("CAN-R2:", self.asked[0])
        self.assertIn("CAN-R3:", self.asked[0])
        self.assertIn("Toronto, ON, Canada", self.asked[0])

    def test_no_law_country_never_asks_the_vlm(self):
        verdict = self.judge.judge(LawLocation("GBR"), self._vlm({}))
        self.assertEqual(verdict.label, LawLabel.NOT_JAYWALKING)
        self.assertEqual(self.asked, [])

    def test_country_without_rule_set_gives_no_verdict(self):
        self.assertIsNone(self.judge.judge(LawLocation("ARG"), self._vlm({})))
        self.assertIsNone(self.judge.judge(None, self._vlm({})))

    def test_us_jurisdiction_comes_from_the_location(self):
        answers = {"USA-R1": "YES", "USA-R5": "YES", "USA-X1": "NO"}
        self.assertEqual(self.judge.judge(LawLocation("USA", "TX", "Austin"), self._vlm(answers)).label, LawLabel.JAYWALKING)
        self.assertEqual(self.judge.judge(LawLocation("USA", "NY", "New York"), self._vlm(answers)).label, LawLabel.NOT_JAYWALKING)
        self.assertEqual(self.judge.judge(LawLocation("USA", "OR", "Portland"), self._vlm(answers)).label, LawLabel.INSUFFICIENT_EVIDENCE)
        california = self.judge.judge(LawLocation("USA", "CA", "Los Angeles"), self._vlm({**answers, "USA-R4": "NO"}))
        self.assertEqual(california.label, LawLabel.NOT_JAYWALKING)

    def test_intersection_check_comes_from_the_rule_file(self):
        rule = self.judge.book.rules["USA"]
        self.assertEqual(rule.supplementary_ids, ["USA-X1"])
        texas = LawLocation("USA", "TX", "San Antonio")
        base = {"USA-R1": "YES", "USA-R5": "YES"}
        # At an intersection: an unmarked crosswalk is a designated crossing.
        verdict = self.judge.judge(texas, self._vlm({**base, "USA-X1": "YES"}))
        self.assertEqual((verdict.label, verdict.verdicts["USA-R5"]), (LawLabel.NOT_JAYWALKING, "NO"))
        self.assertEqual(verdict.verdict_sources["USA-R5"], "USA-X1")
        self.assertEqual(verdict.verdicts["USA-X1"], "YES")
        self.assertIn("USA-X1: The pedestrian crosses at a road intersection", self.asked[-1])
        self.assertIn('"USA-X1": "YES|NO|UNKNOWN"', self.asked[-1])
        # Cannot tell: no JAYWALKING label.
        verdict = self.judge.judge(texas, self._vlm({**base, "USA-X1": "UNKNOWN"}))
        self.assertEqual((verdict.label, verdict.verdicts["USA-R5"]), (LawLabel.INSUFFICIENT_EVIDENCE, "UNKNOWN"))
        # Clearly mid-block: the rule set decides as before.
        self.assertEqual(self.judge.judge(texas, self._vlm({**base, "USA-X1": "NO"})).label, LawLabel.JAYWALKING)
        # A NO from the rule set is never overridden.
        verdict = self.judge.judge(texas, self._vlm({"USA-R1": "YES", "USA-R5": "NO", "USA-X1": "UNKNOWN"}))
        self.assertEqual((verdict.label, verdict.verdict_sources["USA-R5"]), (LawLabel.NOT_JAYWALKING, "vlm"))

    def test_countries_without_supplementary_conditions_are_unchanged(self):
        self.judge.judge(LawLocation("AUS"), self._vlm({"AUS-R1": "YES", "AUS-R3": "YES", "AUS-R4": "YES"}))
        self.assertNotIn("Also evaluate", self.asked[-1])
        self.assertNotIn("-X1", self.asked[-1])

    def test_supplementary_conditions_are_validated(self):
        import json
        import tempfile

        payload = json.loads(RULES.read_text(encoding="utf-8"))
        payload["countries"]["USA"]["supplementary"][0]["overrides"] = {"YES": {"USA-R9": "NO"}}
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "rules.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaises(ValueError):
                RuleBook(path)

    def test_unallowed_na_counts_as_unknown(self):
        verdict = self.judge.judge(LawLocation("CAN"), self._vlm({"CAN-R1": "YES", "CAN-R3": "N/A", "CAN-X1": "NO"}))
        self.assertEqual(verdict.verdicts["CAN-R3"], "UNKNOWN")
        self.assertEqual(verdict.label, LawLabel.INSUFFICIENT_EVIDENCE)

    def test_malformed_responses_are_rejected(self):
        with self.assertRaises(ValueError):
            parse_verdicts({"verdicts": {"CAN-R1": "YES"}, "evidence_summary": ""}, ["CAN-R1", "CAN-R3"])
        with self.assertRaises(ValueError):
            parse_verdicts({"verdicts": {"CAN-R1": "MAYBE"}, "evidence_summary": ""}, ["CAN-R1"])

    def test_prompt_lists_unspecified_thresholds(self):
        rule = RuleBook(RULES).rules["AUT"]
        known = prefilled_verdicts(rule, LawLocation("AUT"))
        prompt = build_prompt(rule, LawLocation("AUT"), [key for key in rule.condition_ids if key not in known])
        self.assertIn("do not infer", prompt)
        self.assertIn("short distance", prompt)


if __name__ == "__main__":
    unittest.main()
