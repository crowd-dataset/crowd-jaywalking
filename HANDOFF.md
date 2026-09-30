# crowd-jaywalking: handoff for a new session

Read this file first. It records where the project stands on 2026-10-01, what the
user decided, and what is still open. The README describes the code; this file
records the context that is not in the code or git history.

## 1. Goal

Find pedestrians in the CROWD dataset (dashcam videos, 32,570 hours, 238 countries;
`mapping.csv`) who cross the road **in front of the camera car** without a zebra
crossing and without a traffic light, and then decide, per country, whether that is
jaywalking.

* The paper must be able to claim that whenever the system says "crossed without a
  zebra crossing and without a traffic light", the claim is true. Precision of that
  claim is the priority; recall ("get as many correct as possible, we want the data")
  is second.
* Human verification at CROWD scale is impossible, so correctness is measured on JAAD.
* Final runs go on SPIKE-1 (a cluster). Local testing is on one RTX 5090 (32 GB).
  Nothing is sent to SPIKE-1 until the method is confirmed on the 5090.
* When the user labels a claim "yes", it means: the person crossed without a traffic
  light and without a zebra crossing.

## 2. Current design (three stages)

1. **Stage 1: potential crossings from crowd-city.** `crossing_decision_mode:
   "crowd_city"`. `src/crowd_jaywalking/crowd_city_detection.py` is an unchanged copy
   of crowd-city's `utils/crossing/detection.py` (github.com/crowd-dataset/crowd-city,
   main, commit 0b8a164), with only its mapping lookup stubbed; the adapter is
   `crowd_city_crossing.py` (confidence >= 0.7, strip 0.45 to 0.55, crowd-city's
   default parameters). Verified identical to crowd-city on all 323 JAAD videos (232
   accepted tracks, same frame bounds). Bounding boxes only; **no SegFormer** (the user
   removed it: "we just need the knowledge from that code whether the person is
   crossing").
2. **Stage 2: VLM (Qwen/Qwen3-VL-8B-Instruct).**
   * Crossing check, prompt `crossing_vlm_check_version: "v3"`: only YES when the
     person crosses the road the camera car is driving on, in front of the car
     (crossings of side streets or other roads are NO). Only a clear YES continues.
   * Infrastructure check, prompt mode `zebra_light_v5`: zebra crossing, traffic light,
     crossing sign; the VLM never judges the person here.
   * Policy: strict absence, scene scope (any infrastructure seen for any crossing
     person in the clip blocks every claim in that clip).
   * The Mask2Former segmentation veto was **deleted** at the user's request (it
     blocked 11 correct claims to stop 1, confusing snow, slush, and glare with
     crosswalks and counting distant traffic lights).
   * `evidence_infrastructure_span` ("transition" default, or "track") was tried;
     "track" did not improve results, keep "transition".
3. **Stage 3: country jaywalking law**, a separate step run afterwards on the saved
   claims: `run_jaywalking_law.py` (`law_stage.py`, `jaywalking_law.py`). Rules come from
   `data/Global_Jaywalking_Laws.pdf` (51 countries, edition 3, March 2026), converted
   by `build_jaywalking_rules.py` into `configs/jaywalking_rules.json`. The VLM answers
   YES/NO/UNKNOWN per condition ID (for example `CAN-R3`); the label is computed in
   code from the rule set's decision rule. R2 (crossing) is filled from stages 1 and 2;
   US jurisdiction conditions come from state/city. Model: `jaywalking_law_model`
   (default = `vlm_model`). The user will send an updated PDF later; then rebuild the
   JSON and rerun stage 3 only.

## 3. Results on JAAD (current design, YOLO11x tracks)

Config `crowd_city_vlm_v3.config` (git-ignored, see section 6). Audit = every claim is
matched to a JAAD pedestrian; confirmed when JAAD says crossing and JAAD's per frame
`ped_crossing` and `traffic_light` flags are off on its crossing frames.

| JAAD split | Claims correct |
| --- | --- |
| Test | 8 of 8 |
| Train | 13 of 13 |
| Val | 2 of 2 |
| All | 23 of 23 (95% lower bound 87.8%), 23 of 111 eligible crossers found (20.7%) |

* 22 of the 23 are confirmed by JAAD; `video_0092` person 21 JAAD marks as "traffic
  light in view", but the user reviewed it: correct (the person is far from the light).
* Prompt v3 was chosen after the test split had been looked at, so the test split is
  no longer a clean held-out set; the paper must say so.
* Stage 3 check on the same 23 claims: all JAYWALKING under Canada and Ukraine, all
  NOT_JAYWALKING under Australia (no crossing within 20 m, AUS-R3), as the rules intend.

Earlier designs, for comparison (all on JAAD):

| Design | Correct / claims | Found |
| --- | --- | --- |
| Tracking-only strict (first-stage classifier + camera-motion gate + scene rule), YOLO26x | 31 / 32 | 28% |
| Same, YOLO11x | 31 / 34 (32 / 34 after user review of video_0017) | 28% |
| crowd-city + VLM v1 crossing check, with the Mask2Former veto | 9 / 9 | 8% |
| crowd-city + VLM v3 crossing check, no veto (current) | 23 / 23 after review | 21% |

Recall losses, current design, test split: stage 1 proposes 15 of 39 eligible
crossers; the VLM crossing check then rejects about 6 of those ("walks along the
sidewalk").

## 4. Known issues and open work

1. **No CROWD samples yet.** The precomputed CROWD track CSVs
   (`<video_id>_<start>_<fps>.csv`, YOLO11x, conf 0.0, 640 px, BoT-SORT 2 s buffer)
   are not on this machine (crowd-city's config points to `D:/data_folder`, absent).
   Stages 2 and 3 also need the video frames. Waiting for the user to choose:
   (a) copy some track CSVs and matching videos locally, or (b) approve downloading a
   few Toronto (Canada, rule CAN-R3) videos from `files.mobility-squad.com` (credentials
   in `secret`; a header-only size check timed out). **Downloads need explicit approval
   with file names and sizes.**
2. Stage 3 R1 (public road vs private property): the VLM said YES for all 23 JAAD
   claims, including 9 in JAAD "parking lots". Needs a better prompt or a check
   against JAAD `road_type`.
3. "Designated crossing" is treated as a marked crossing or signal; in Canada/US an
   unmarked intersection can be a legal crossing.
4. Red-light crossing offenses are out of scope (scenes with traffic lights are dropped).
5. Gaps in the PDF (a message listing them was drafted for the user's colleague):
   US state table (US = 8,964 h), distances unspecified for 29 countries, no
   conditions for GRC, IND, MEX, NGA, PAK, ZWE, no province/state tables (CAN, AUS,
   MEX), unclear definitions (unmarked intersection, private vs public parking,
   bridges/underpasses), missing situations (motorways, pedestrian zones, giving way,
   diagonal crossing, buses, children), 187 CROWD countries missing (12.4% of hours;
   largest: Myanmar, UAE, Argentina, Cambodia, Czechia), sources are database names
   only, one global "last updated" date.
6. Speed benchmark for SPIKE-1 sizing not done.

## 5. How to run

```powershell
uv sync
uv run python -m unittest discover -s tests          # 128 tests pass
# Stages 1 and 2 on a JAAD split (writes results/<results>/jaad_person_audit_<split>/)
$env:CROWD_JAYWALKING_CONFIG = ".\crowd_city_vlm_v3.config"
$env:CROWD_JAYWALKING_JAAD_SPLIT = "test"
uv run python -u .\run_jaad_person_audit.py
# Stage 3 on the saved claims (JAAD needs --country; CROWD takes it from mapping.csv)
uv run python .\run_jaywalking_law.py results\jaad_crowd_city_vlm_v3 --country CAN
# Rebuild the rule file when the PDF changes (needs pdftotext)
uv run python .\build_jaywalking_rules.py data\Global_Jaywalking_Laws.pdf configs\jaywalking_rules.json
```

The no-veto results in `results/jaad_crowd_city_vlm_v3_noveto` were produced by
re-deciding saved VLM answers, so its evidence lives in `results/jaad_crowd_city_vlm_v3`
(use `--evidence-from results\jaad_crowd_city_vlm_v3`). The current code has no veto,
so a fresh run gives the no-veto result directly.

## 6. Local-only files (git-ignored; copy them by hand to another machine)

* `*.config` except `strict.config`. The current one is `crowd_city_vlm_v3.config`
  = `strict.config` plus: `results: results/jaad_crowd_city_vlm_v3`, `tracking_model:
  yolo11x.pt`, `min_confidence: 0.0`, `iou: 0.7`, `jaad_benchmark_results:
  results/jaad_yolo11x`, classifier/gate paths `results/jaad_crossing_classifier_yolo11x`
  and `results/jaad_crossing_gate_yolo11x`, `crossing_decision_mode: crowd_city`,
  `crossing_vlm_check: true`, `crossing_vlm_check_version: v3`. Other settings in
  strict.config: `vlm_prompt_mode: zebra_light_v5`, `vlm_max_pixels: 250880`,
  `vlm_task_max_frames: 4`, `strict_absence: true`.
* `data/`: JAAD, `annotations*.csv`, `evaluation_manifest.csv`, `claim_review.csv`
  (the user's verdicts on claims), `label_corrections.csv`, `Global_Jaywalking_Laws.pdf`.
  The user corrected video_0183 and video_0184 to Yes/JAYWALKING (backups in
  `data/backups/`).
* `results/`: all runs, trained classifier/gate artifacts, `jaad_yolo11x` tracks and
  `camera_motion`.
* `secret`: CROWD file server credentials. Never print or share.

## 7. Git

* Branch `main`. `7c11ad0` added the crowd-city detector, the v3 prompt, and removed the veto;
  the next commit adds stage 3 (country law), this file, and the README "Stage 3" section.
* Commit or push only when the user asks. `gh` is logged in as a non-collaborator
  account; do not switch accounts.

## 8. Working rules the user set or confirmed

* Never change ground-truth labels without an explicit instruction.
* Ask before downloading anything (state file, source, size).
* The VLM must only see crossing snippets, never full footage.
* Crossing is decided by tracking (crowd-city) plus the VLM crossing check; the VLM
  infrastructure prompt must not judge the person.
* Tune on JAAD train and val; say clearly when the test split has been used.

## 9. Environment notes

* Windows 11; Python via `uv`; CUDA torch 2.11 (cu128); Qwen3-VL-8B needs
  `vlm_max_pixels` 250880 to stay inside 32 GB of VRAM.
* Shell gotcha: in Bash heredocs a double backslash can be collapsed (`\\b` became a
  backspace in the README once); write files with the Write tool or use `chr(92)`.
* Files mix CRLF and LF; keep each file's existing line endings.
* `pdftotext` (poppler) is available in Git Bash (`/mingw64/bin`).
* The crowd-city repo is at `..\crowd-city` (its local clone was behind GitHub; the
  latest main was exported to a scratch folder for reading, not pulled).
