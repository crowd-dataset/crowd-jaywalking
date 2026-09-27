"""Tests for the JAAD person level claim audit."""

import unittest

from crowd_jaywalking.jaad import JAADPedestrianTrack, JAADVideoAnnotations
from crowd_jaywalking.jaad_person_audit import (
    CONFIRMED,
    claim_verdict,
    clopper_pearson_lower,
    eligible_crossers,
)
from crowd_jaywalking.models import BoundingBox

BOX = BoundingBox(0.40, 0.40, 0.50, 0.80)
FRAMES = tuple(range(10))


def pedestrian(pid, crossing):
    return JAADPedestrianTrack(
        pedestrian_id=pid,
        label="pedestrian" if crossing is not None else "ped",
        frames=FRAMES,
        boxes={f: BOX for f in FRAMES},
        occlusion={f: 0 for f in FRAMES},
        crossing={f: crossing for f in FRAMES},
        attributes={},
    )


def video(tracks, zebra=False, light=False):
    traffic = {
        f: {"ped_crossing": "1" if zebra else "0", "traffic_light": "red" if light else "n/a"}
        for f in FRAMES
    }
    return JAADVideoAnnotations(
        video_id="video_0001", num_frames=10, width=1920, height=1080,
        tracks={t.pedestrian_id: t for t in tracks}, traffic=traffic, road_type="street",
    )


EVENT = {"start_frame": 0, "end_frame": 9}
TRACK = {f: BOX for f in FRAMES}


class JAADPersonAuditTests(unittest.TestCase):
    def test_confirmed_crossing_without_infrastructure(self):
        annotations = video([pedestrian("0_1_1b", True)])
        self.assertEqual(claim_verdict(annotations, TRACK, EVENT, 0.5, 5), CONFIRMED)
        self.assertEqual(len(eligible_crossers(annotations)), 1)

    def test_zebra_or_light_makes_the_claim_wrong(self):
        for kwargs in ({"zebra": True}, {"light": True}):
            annotations = video([pedestrian("0_1_1b", True)], **kwargs)
            self.assertTrue(claim_verdict(annotations, TRACK, EVENT, 0.5, 5).startswith("wrong"))
            self.assertEqual(eligible_crossers(annotations), [])

    def test_non_crosser_and_bystander(self):
        self.assertTrue(
            claim_verdict(video([pedestrian("0_1_1b", False)]), TRACK, EVENT, 0.5, 5).startswith("wrong")
        )
        self.assertTrue(
            claim_verdict(video([pedestrian("0_1_2", None)]), TRACK, EVENT, 0.5, 5).startswith("unverifiable")
        )
        self.assertTrue(claim_verdict(video([]), TRACK, EVENT, 0.5, 5).startswith("unverifiable"))

    def test_clopper_pearson_lower_bound(self):
        self.assertAlmostEqual(clopper_pearson_lower(26, 26), 0.05 ** (1 / 26))
        self.assertLess(clopper_pearson_lower(25, 26), clopper_pearson_lower(26, 26))
        self.assertEqual(clopper_pearson_lower(0, 0), 0.0)


if __name__ == "__main__":
    unittest.main()
