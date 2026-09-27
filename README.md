# Crowd Jaywalking Clean Pipeline

This repository implements a clean first version of person specific jaywalking detection.

The system separates three questions:

1. Did a tracked person genuinely cross the road corridor?
2. What crossing infrastructure or traffic control is visibly present for that person?
3. Does the observable context satisfy the fixed jaywalking definition?

The visual language model never directly decides the final label. It reports observable context, and deterministic code applies the policy.

Two committed configuration profiles trade recall for precision:

| Profile | File | Purpose | JAAD precision of positive claims | Recall |
| --- | --- | --- | --- | --- |
| Standard | `default.config` | Larger sets, with a measured error rate | about 86 to 93% per person | about 50% |
| Strict | `strict.config` | Claims that must be correct: every positive is confirmed by independent checks | 26 of 26 on JAAD (9 of 9 held out) | about 23% |

See [Strict mode](#strict-mode) for how the strict figure was measured and what it does and does not show.

## Operational definition

A valid crossing person is classified as `JAYWALKING` when they cross the road with no zebra crossing (marked crosswalk) and no traffic light, for vehicles or pedestrians, at the crossing place.

The positive label is meant to be precise rather than complete. A `JAYWALKING` person is passed to later jaywalking analysis, so any doubt about crossing infrastructure produces `UNCERTAIN` instead of a positive.

The policy is configured by five settings:

| Setting | `default.config` | Legacy value (used when the key is absent) | Meaning |
| --- | --- | --- | --- |
| `permission_cues` | `["marked_crosswalk", "permissive_pedestrian_signal", "traffic_light"]` | all four cues, which adds `authorised_crossing_sign` and `crossing_guard_permission` | Visible cues that make a crossing compliant |
| `context_scope` | `"scene"` | `"person"` | `scene` counts a cue seen for any crossing person in the clip for every crossing person in that clip. `person` uses only that person's own evidence |
| `partial_visibility_uncertain` | `false` | `true` | Whether partial visibility with no visible permission gives `UNCERTAIN` instead of `JAYWALKING` |
| `prohibitive_signal_overrides_crosswalk` | `false` | `true` | Whether a visible red or do not walk signal for the person gives `JAYWALKING` even at a crosswalk. It is off because a red light means a traffic light is present |
| `strict_absence` | `true` | `false` | A positive requires all five infrastructure fields (zebra, green light, red light, crossing sign, crossing guard) to be `NO` for every crossing person in the clip. Any `YES` or `UNCERTAIN` that grants no permission gives `UNCERTAIN` |

`scene` scope matches how the video ground truth is defined. JAAD's own annotations show that every annotated clip contains a crossing pedestrian. On the development split, "a JAAD crosser is present and JAAD marks no pedestrian crossing in the clip" agrees with the human video label 94% of the time. So the video label is essentially a scene level question. Per person evidence is strict: the VLM must see the stripes intersect that one person's sampled path, which often fails for people crossing beside the stripes or seen at an angle.

The video label is `JAYWALKING` when at least one valid crossing person is classified as jaywalking. If no valid crossing is found, the video label is `COMPLIANT`. Insufficient visual evidence produces `UNCERTAIN`.

## Architecture

1. YOLO26x detects objects and BoT SORT assigns track IDs.
2. The frozen JAAD supervised classifier (the first stage) scores every usable person track using only inference safe tracking and motion features.
3. A high precision crossing gate must also accept every first stage crossing. The first stage was trained only on JAAD pedestrians with behaviour labels, but it scores every track, so on its own it accepts many bystanders. See [Crossing gate](#crossing-gate).
4. Optionally, the VLM confirms that each candidate really walks across the carriageway (`crossing_vlm_check`), and first stage rejections that the gate strongly trusts can be rescued by that confirmation (`crossing_rescue_min_first_stage`, `crossing_rescue_min_gate`). Both are on in the strict profile.
5. The rule detector is retained as an audit feature and can be selected as an explicit fallback, but it does not override classifier decisions in classifier mode.
6. Six chronological moments centred on the crossing transition are sampled for every accepted person. Each produces a full scene, target detail, trajectory map, road area, a full resolution crossing close-up around the target's feet, the road ahead at full width, and six near native resolution tiles of the upper scene where signals and signs are.
7. A Hugging Face VLM reports the observable context. The default prompt mode `zebra_light_v3` makes two calls: markings within about 20 metres of the path, from the close-up and road ahead views, and traffic lights and crossing signs, from the tiles. Older modes remain selectable with `vlm_prompt_mode`.
8. Optionally, an independent street scene segmentation model (Mask2Former trained on Mapillary Vistas) measures crosswalk paint near the path and on the road ahead, and traffic lights in view, at full resolution. When `infrastructure_segmentation_model` is set, any such finding vetoes a positive claim.
9. A deterministic policy produces the person and video labels.

The configured tracker is BoT SORT with ReID enabled and the exact parameters in `configs/botsort.yaml`.

The committed v1.5 configuration models the ego corridor as a trapezoid that widens towards the camera. Its left and right boundaries are interpolated at the pedestrian bounding box bottom edge. These generic values are an initial geometry and must be calibrated using training and validation data before they are treated as final dataset parameters.

## Repository structure

```text
.
├── default.config            # standard profile
├── strict.config             # strict profile for claims that must be correct
├── config                    # optional local configuration, ignored by Git
├── default.secret            # template for the ignored CROWD download credentials
├── configs/
│   └── botsort.yaml
├── data/
│   ├── annotations.csv
│   ├── annotations_with_splits.csv
│   ├── JAAD/                 # created by download_jaad.ps1
│   └── videos/
├── mapping.csv               # CROWD source video to segment mapping
├── download_jaad.ps1
├── prepare_splits.py
├── run_one_video.py
├── run_evaluation.py
├── diagnose_evaluation.py
├── run_jaad_crossing_benchmark.py
├── train_jaad_crossing_classifier.py
├── train_crossing_gate.py
├── evaluate_jaad_crossing_classifier.py
├── prepare_jaad_context.py
├── run_jaad_context_benchmark.py
├── compare_jaad_context_models.py
├── run_crowd_analysis.py
├── run_jaad_person_audit.py  # runs a profile on a JAAD split and audits every claimed person
├── jaad_front_crossing_annotator.html
├── jaad_observable_context_annotator.html
├── src/crowd_jaywalking/
│   ├── config.py
│   ├── crossing.py                  # rule based crossing detector and corridor geometry
│   ├── crossing_classifier.py       # JAAD classifier training, evaluation, and loading
│   ├── crossing_gate.py             # high precision gate against bystanders
│   ├── model_crossing.py            # applies the frozen classifier to tracks
│   ├── track_features.py            # inference safe track features
│   ├── crowd_analysis.py
│   ├── crowd_source.py              # CROWD mapping, download, and segment extraction
│   ├── evidence.py
│   ├── evaluation.py
│   ├── jaad.py                      # official JAAD annotation loader
│   ├── jaad_benchmark.py
│   ├── jaad_context.py              # context audit sample and evidence
│   ├── jaad_context_evaluation.py
│   ├── jaad_person_audit.py         # person level claim audit against JAAD labels
│   ├── models.py
│   ├── pipeline.py
│   ├── policy.py
│   ├── segmentation.py              # independent crosswalk and traffic light check
│   ├── tracking.py
│   ├── vlm.py
│   └── vlm_comparison.py
└── tests/
```

## Windows setup

Run all commands from the repository root in PowerShell.

### 1. Install dependencies

```powershell
uv sync
```

Verify that the source package was installed correctly:

```powershell
uv run python -c "import crowd_jaywalking; print(crowd_jaywalking.__file__)"
```

The VLM is loaded directly by Transformers. Ollama is not used and does not need to be installed or started.

On Windows and Linux, `pyproject.toml` installs PyTorch from the CUDA 12.8 wheel index, the minimum for RTX 50 series GPUs. Confirm the GPU is used before running anything long:

```powershell
uv run python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

A CPU build of PyTorch still works but is about 25 times slower for the VLM. `bitsandbytes` is installed so that larger VLMs can be loaded in 4-bit by setting `"vlm_quantization": "4bit"`. The committed profiles leave it unset.

The images sent to the VLM are capped at `vlm_max_pixels` (250880) so that one call stays within 32 GB of VRAM together with the segmentation model. Calls that overflow VRAM on Windows fall back to shared system memory and become tens of times slower, so lower this value on smaller GPUs rather than raising it.

### 2. Create the local configuration

The project follows the same configuration convention as the CROWD repository. `default.config` is the committed template and `config` is the machine specific file without an extension. The supplied project archive includes both, while Git ignores `config`.

If you need to recreate the local file, run:

```powershell
Copy-Item .\default.config .\config
```

Edit `config` for local paths and evaluation settings. If `config` does not exist, the code automatically uses `default.config`. Every setting is a top level entry, matching the flat CROWD configuration style:

```text
{
  "data": ["data"],
  "videos": ["data/videos"],
  "source_annotations": "annotations.csv",
  "annotations": "annotations_with_splits.csv",
  "tracking_model": "yolo26x.pt",
  "bbox_tracker": "configs/botsort.yaml",
  "min_confidence": 0.25,
  "vlm_model": "Qwen/Qwen3-VL-8B-Instruct",
  "vlm_comparison_models": [
    "Qwen/Qwen3-VL-8B-Instruct",
    "google/gemma-4-12B-it"
  ]
}
```

To use another configuration file for one run:

```powershell
$env:CROWD_JAYWALKING_CONFIG = ".\config.validation"
uv run python .\run_evaluation.py
```

### 3. Optional Hugging Face cache location

By default, Hugging Face uses its normal user cache. To put the model weights on another drive, set `HF_HOME` before running the pipeline:

```powershell
$env:HF_HOME = "D:\huggingface-cache"
```

You can also set `vlm_cache_dir` in `config` to an absolute path. Leave it as `null` to use the normal Hugging Face cache.

The default VLM is `Qwen/Qwen3-VL-8B-Instruct`. On the automatic JAAD context benchmark it was more precise than `google/gemma-4-12B-it` and `Qwen/Qwen2.5-VL-7B-Instruct` with the same prompt. The strict profile also downloads `facebook/mask2former-swin-large-mapillary-vistas-semantic` (about 830 MB) on first use; check that the Mapillary Vistas licence fits your use. Their weights are large, so allow substantial disk space and GPU or system memory. `device_map: "auto"` lets Accelerate place each model on the available GPU and CPU resources. The first inference downloads the weights from Hugging Face. YOLO26x weights are also downloaded automatically on first use by Ultralytics.

### 4. Prepare the dataset

Place the JAAD videos in:

```text
data\videos
```

Place the human annotations at:

```text
data\annotations.csv
```

The supported source format is:

```csv
video_id,filename,label
video_0002,video_0002.mp4,No
video_0003,video_0003.mp4,Yes
video_0007,video_0007.mp4,Not Sure
```

`Yes` means `JAYWALKING`, `No` means `COMPLIANT`, and other labels are excluded.

### 5. Create frozen stratified splits

```powershell
uv run python .\prepare_splits.py
```

The default split is 60% development, 20% validation, and 20% locked test, stratified independently for `Yes` and `No` labels with seed 42.

Use only the development split for initial threshold and prompt work. Use validation to compare proposed configurations. Change to `locked_test` only after the method is frozen.

### 6. Run unit tests

```powershell
uv run python -m unittest discover -s tests -v
```

The unit tests do not download model weights.

### 7. Smoke test one video

```powershell
$env:CROWD_JAYWALKING_VIDEO = ".\data\videos\video_0002.mp4"
uv run python .\run_one_video.py
```

On the first run, leave the terminal open while the model weights download. Inspect the JSON result and the generated evidence images under:

```text
results\jaad_development_v2\smoke
```

Confirm that the red box follows the intended person and that full scene images preserve crosswalk and signal context.

The evidence interval contains the detected crossing transition plus 0.50 seconds of context on each side by default. Change `evidence_context_seconds` if the transition needs more or less surrounding context.

### 8. Evaluate the configured split

```powershell
uv run python .\run_evaluation.py
```

The default configuration evaluates `development`. Results are saved after every video, so an interrupted run can resume safely.

### 9. Diagnose evaluation errors

```powershell
uv run python .\diagnose_evaluation.py
```

This reads the `details/*.json` files already written to `results` and runs no inference. It needs no GPU and no model weights. It reports:

* The distribution of every VLM context field and policy reason.
* The per video outcomes.
* How accuracy changes with the number of accepted crossings per video.
* A crossing threshold sweep recomputed from the saved classifier scores.
* Policy counterfactuals: counting `UNCERTAIN` as either label, and requiring two or more jaywalking people per video.
* Policy variants: the saved context re-decided by the real policy under different `permission_cues`, `context_scope`, and `partial_visibility_uncertain` settings. The first row reproduces the legacy policy, which checks that the replay is faithful.

Use it only on development results. Selecting a threshold or policy change from validation or locked test output leaks those splits.

### 10. Compare the context models and prompt modes

Set `jaad_context_split` to `train` (the default) or `val`. The comparison refuses to select a model on `test`. The evidence includes a target trajectory, a road-path crop, and overlapping high-resolution traffic-control crops. Regenerate the evidence before running the model. Existing manual labels are preserved:

```powershell
uv run python -u .\prepare_jaad_context.py
```

Complete the context fields in `results\jaad_context_audit_v5\<split>\context_annotations.csv`. `jaad_observable_context_annotator.html` can be used for this. Then run every configured model and prompt mode on those exact same events:

```powershell
uv run python -u .\compare_jaad_context_models.py
```

Every combination of `vlm_comparison_models` and `vlm_comparison_prompt_modes` is evaluated, and the models are loaded one at a time. To score a single model and prompt mode, use `run_jaad_context_benchmark.py`.

Selection is by the macro F1 of the derived `permission_present` observable. This value combines only the four visible permission cues and is not a legal or jaywalking label. Ties are broken in this order:

1. Mean macro F1 over fields containing at least two ground truth classes.
2. The weakest field's F1.
3. Mean accuracy.
4. A preference for `baseline_v3`.

The selection is recorded in:

```text
results\jaad_vlm_comparison_v5\<split>\selected_model.json
```

Before the final locked context evaluation and CROWD analysis, copy `selected_model` to `vlm_model` and `selected_prompt_mode` to `vlm_prompt_mode` in `config`.

## Changing evaluation splits

Change both settings before evaluating another split:

```json
"evaluation_split": "validation",
"results": "results/jaad_validation_v1"
```

For the final locked evaluation:

```json
"evaluation_split": "locked_test",
"results": "results/jaad_locked_test_v1"
```

Never reuse a results directory after changing a prompt or configuration. The evaluator checks a configuration fingerprint and refuses to mix incompatible results.

## Outputs

Each evaluation directory contains:

```text
run_manifest.json
per_video_results.csv
summary.json
details/<video_id>.json
evidence/<video>/<person_event>/*.jpg
```

`summary.json` reports:

* Overall accuracy, where `UNCERTAIN` counts as incorrect
* Decided only accuracy
* Coverage
* Precision
* Recall
* Specificity
* F1 score
* TP, TN, FP, FN, and uncertain counts

## Initial parameters

The crossing boundaries and filters in `default.config` are starting values adapted from CROWD city, not validated JAAD parameters. In particular, the `0.45` to `0.55` road corridor assumes centred forward facing footage.

Tune parameters only with development data. Record every configuration and compare it on validation data before selecting a final version.

## Current limitations

* Ground truth remains video level. It does not identify which person caused a positive label.
* The first stage represents the road by a configurable image corridor. Semantic segmentation is used only by the optional infrastructure veto.
* One crossing event is produced per continuous person track segment.
* The fixed central corridor can miss crossings that occur entirely away from the centre of the image; inspect development videos with no candidate before changing this assumption.
* VLM context quality depends on infrastructure being visible in the sampled frames.
* The `focused_v5` prompt mode makes four inference calls per event and `zebra_light_v3` makes two, plus one for the optional crossing check.
* The video label is `JAYWALKING` if any one person is classified as jaywalking, so a single person level false positive flips the video. Busy scenes with many accepted crossings are most exposed.
* `scene` scope assumes one clip shows one crossing location. A person who jaywalks well away from a zebra crossing visible elsewhere in the same clip is labelled `COMPLIANT`. Use `person` scope when individual attribution matters more than the video label.
* Only 6 of the 138 annotated JAAD clips contain a traffic light, so the traffic light part of the definition is only lightly tested on JAAD.
* The strict profile keeps about 23% of JAAD crossers without a zebra crossing or traffic light. Relaxing any of its checks brought back errors that JAAD can verify.
* Some JAAD labels contradict each other or the images: per pedestrian attributes can say "designated, signalised" where the per frame scene annotations and the imagery show neither, and some visible crosswalks are not annotated.
* A malformed or schema-violating VLM response stops the run with `VLMError` rather than inventing a label.
* JAAD test performance estimates crossing detection, not the final jaywalking policy on CROWD.
* CROWD context predictions still require a stratified manual audit before population estimates are reported.

## Run the frozen method on CROWD

Keep the frozen classifier and the exact validated YOLO26 plus BoT SORT setup:

```json
"crossing_decision_mode": "classifier",
"crossing_classifier_model": "results/jaad_crossing_classifier_v1/crossing_classifier.joblib",
"crossing_classifier_fallback_to_rules": false,
"crossing_classifier_min_track_frames": 5,
"crossing_gate_model": "results/jaad_crossing_gate_v1/crossing_gate.joblib",
"crowd_results": "results/crowd_jaywalking_v2"
```

The runner reads the segments listed in `mapping`, which is `mapping.csv` by default. For each source video, it first looks in the `videos` directories. If the video is not there, it downloads it from `ftp_server` into `crowd_download_dir`. Downloading needs credentials in a local `secret` file, which Git ignores. Create it from the template:

```powershell
Copy-Item .\default.secret .\secret
```

Fill in `ftp_username` and `ftp_password`, and `ftp_token` if required. Alternatively, set `CROWD_JAYWALKING_SECRET` to another secret file path. Set `crowd_max_segments` to a positive number to process only a subset. Then run:

```powershell
uv run python -u .\run_crowd_analysis.py
```

To produce claims that must be correct, run the strict profile instead:

```powershell
$env:CROWD_JAYWALKING_CONFIG = ".\strict.config"
uv run python -u .\run_crowd_analysis.py
```

Every person in `per_person_results.csv` also carries `gate_precision_tier`, so a standard profile run can be filtered afterwards to the strictest crossing tier without new inference.

Do not supply legacy detector CSVs to the frozen classifier. It was validated on features from YOLO26 plus the configured BoT SORT tracker, so the runner deliberately creates those tracks again from the video clips.

The CROWD output directory contains:

```text
run_manifest.json
summary.json
per_video_results.csv
per_person_results.csv
audit_sample.csv
errors.csv
details/<video_key>.json
tracking/<video_key>.csv
evidence/<video>/<person_event>/*.jpg
```

`audit_sample.csv` deterministically samples classifier boundary cases, high confidence crossings, boundary non-crossings, and random non-crossings. Complete `human_crossing` and `human_jaywalking` only for the sampled CROWD cases after inference. This audit is separate from JAAD tuning and does not alter the frozen model.

## Official JAAD validation

Download the official JAAD 2.0 annotations and all 346 clips into `data\JAAD`:

```powershell
.\download_jaad.ps1
```

Each stage is then validated separately against the official labels:

```powershell
uv run python -u .\run_jaad_crossing_benchmark.py        # tracking and rule crossing detection on jaad_benchmark_split
uv run python -u .\train_jaad_crossing_classifier.py     # fit and select the classifier from the saved train and val benchmarks
uv run python -u .\evaluate_jaad_crossing_classifier.py  # one-off evaluation of the frozen classifier on the saved test benchmark
uv run python -u .\train_crossing_gate.py                # fit the crossing gate on train and val tracks, check it on test
uv run python -u .\run_jaad_context_benchmark.py         # VLM context fields against the manual context annotations
uv run python -u .\run_jaad_person_audit.py              # run a profile on a JAAD split and audit every claimed person
```

Run the crossing benchmark once each for `jaad_benchmark_split` set to `train`, `val` and `test` before training. The trainer and the evaluator read `<jaad_benchmark_results>\<split>\per_pedestrian.csv`.

### Frozen crossing classifier benchmark

The frozen logistic regression crossing classifier with threshold `0.57` was evaluated once on the official JAAD test split. This measures only whether the first stage identifies crossing pedestrians. It is not the accuracy of the final jaywalking decision.

| Metric | Result |
| --- | ---: |
| Videos | 117 |
| Annotated pedestrians | 276 |
| Track match recall | 98.55% |
| Crossing accuracy | 79.35% |
| Crossing precision | 87.71% |
| Crossing recall | 81.77% |
| Crossing specificity | 73.81% |
| Crossing F1 | 84.64% |
| Balanced accuracy | 77.79% |

The confusion matrix was TP 157, TN 62, FP 22, and FN 35. These are locked test results and must not be used for further threshold selection.

### Crossing gate

The positive output must be trustworthy without human checking, because CROWD is far too large to verify by hand. The gate therefore measures itself automatically against JAAD.

Every track that the first stage accepts in the saved JAAD train, val and test benchmarks is labelled by matching it to JAAD's own annotations:

* A pedestrian whose annotated crossing overlaps the track is the only positive.
* JAAD bystanders without behaviour labels, unannotated detections, labelled non crossers, and crossings at another time are all negatives. These tracks cannot be verified, so counting them as wrong makes the reported precision a lower bound.
* Videos in the project's locked test split are excluded, so the end to end locked evaluation is untouched.

A gradient boosting gate uses the same inference safe track features as the first stage. It is fitted on JAAD train and val. Grouped, out of fold predictions over those videos set one threshold per precision tier in `crossing_gate_precision_tiers`, each accepting at least `crossing_gate_min_accepted` tracks. JAAD's official test videos are not used for fitting and serve once as the independent check.

The pipeline accepts people at the `crossing_gate_min_precision` tier (0.90), sends them to the VLM, and records `gate_precision_tier`: the strictest tier each person meets. One CROWD run can therefore be filtered afterwards to the most confident subset or to the largest one, without new inference.

Worst case precision, with every unverifiable track counted as wrong:

| Tier | Train+val, out of fold | JAAD test check | Real crossers kept (test) |
| --- | ---: | ---: | ---: |
| First stage only | 31.5% | 32.6% | 100% |
| 98% | 98.1% | 92.1% (117 of 127) | 70.9% |
| 95% | 95.5% | 90.0% (126 of 140) | 76.4% |
| 90% | 90.2% | 86.1% (136 of 158) | 82.4% |

The JAAD test check is lower than cross validation predicts, and not only because of unverifiable bystanders: among JAAD verifiable tracks, crossing precision on test is 92 to 94%. The crossing stage is therefore the main remaining source of error, before the VLM and the strict policy filter further.

Replaying the frozen gate and the strict policy on the saved locked test run:

| Tier | People flagged | Confirmed no zebra, no light | Borderline (`video_0122`, JAAD zebra, human label jaywalking) | Unverifiable | Video TP / FP / FN |
| --- | ---: | ---: | ---: | ---: | --- |
| 98% | 7 | 6 | 1 | 0 | 7 / 0 / 3 |
| 95% | 8 | 7 | 1 | 0 | 8 / 0 / 2 |
| 90% | 9 | 7 | 1 | 1 | 8 / 0 / 2 |

For the most reliable CROWD data, keep people with `gate_precision_tier` of at least 0.95. Of the 12 JAAD labelled crossers in the locked test jaywalking videos, 8 are found at the 95% tier. Three of the four misses are rejected by the first stage. Letting the gate see tracks below the first stage threshold gave no stable gain in cross validation, so the first stage threshold is unchanged.

These are small samples. They show high precision, not 100% precision.

Set `crossing_gate_model` to the trained artifact to enable the gate. When the key is absent, the gate is disabled and legacy configurations keep their fingerprints. A configured gate file that is missing stops the run rather than being skipped.

## Strict mode

The strict profile exists for one statement: whenever it claims that a person crossed without a zebra crossing and without a traffic light, the claim is true. A person is claimed only when every independent check agrees:

1. The crossing gate places the track in the 95% tier, or a first stage rejection is rescued (first stage score of at least 0.25 and gate score of at least 0.98).
2. The VLM confirms that the person walks across the carriageway.
3. The VLM reports no zebra crossing, traffic light, or crossing sign, and every other crossing person in the clip reports the same.
4. Segmentation finds fewer than 50 crosswalk pixels near the person's path, fewer than 1000 on the road ahead, and fewer than 100 traffic light pixels in view.

Any doubt produces `UNCERTAIN`, never a claim.

### How the precision figure was measured

Reproduce it with:

```powershell
$env:CROWD_JAYWALKING_CONFIG = ".\strict.config"
$env:CROWD_JAYWALKING_JAAD_SPLIT = "test"
uv run python -u .\run_jaad_person_audit.py
```

The audit runs the profile over the saved tracks of one JAAD split and matches every claimed person to a JAAD pedestrian. A claim is confirmed when that pedestrian has a JAAD crossing label and JAAD's per frame scene annotations mark no pedestrian crossing and no traffic light on its crossing frames. Bystanders without behaviour labels and unannotated detections cannot be verified and would be reported separately; none were claimed.

| JAAD split | Claims | Confirmed | 95% lower bound on precision | Eligible crossers | Recall |
| --- | ---: | ---: | ---: | ---: | ---: |
| Test (117 videos, held out) | 9 | 9 | 71.7% | 39 | 23.1% |
| Train and val (192 videos) | 17 | 17 | 83.8% | 72 | 23.6% |
| All | 26 | 26 | 89.1% | 111 | 23.4% |

The lower bound is the one sided Clopper-Pearson bound. Held out means the gate was fitted, the prompt was chosen, and every segmentation threshold was fixed on JAAD train and val data only. The project's locked test videos were excluded from all tuning and are part of the JAAD test split. Train and val are tuning data, so their row shows consistency rather than an independent estimate.

Two of the nine held out claims (`video_0093`, `video_0101`, one residential street) carry the JAAD pedestrian attribute "designated, signalised" although JAAD's per frame annotations and the images show neither a crossing nor a signal. Under the stricter attribute based definition, 7 of the 9 held out claims are confirmed.

### Where eligible crossers are lost

On train and val, before rescue, eligible crossers were lost at: the first stage 24%, the VLM crossing check 18%, the segmentation veto 18%, the gate tier 10%, and the VLM context check 10%. Each relaxation that was tried recovered crossers only by admitting errors that JAAD can verify: looser crossing check wording, a 90% gate tier, alternative segmentation thresholds, a first stage retrained with bystanders and ground surface features, and a 27B VLM for the crossing check (Qwen3.5-27B in 4-bit, which kept fewer crossers than Qwen3-VL-8B). Rescue was the one change that raised recall without errors.

### Front crossing audit

Open `jaad_front_crossing_annotator.html` in a browser, select the JAAD video folder, and label whether any pedestrian crosses directly through the ego vehicle's forward driving path. The tool stores progress locally in the browser and exports a CSV file.

On macOS:

```bash
open jaad_front_crossing_annotator.html
```

On Windows PowerShell:

```powershell
Start-Process .\jaad_front_crossing_annotator.html
```
