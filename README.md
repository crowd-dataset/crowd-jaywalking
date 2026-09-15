# Crowd Jaywalking Clean Pipeline

This repository implements a clean first version of person specific jaywalking detection.

The system separates three questions:

1. Did a tracked person genuinely cross the road corridor?
2. What crossing infrastructure or traffic control is visibly present for that person?
3. Does the observable context satisfy the fixed jaywalking definition?

The visual language model never directly decides the final label. It reports observable context, and deterministic code applies the policy.

## Operational definition

A valid crossing person is classified as `JAYWALKING` when no marked crosswalk, permissive pedestrian signal, authorised crossing sign, or crossing guard permission is visible.

The current policy assumes that a visible red or do not walk pedestrian signal overrides a marked crosswalk. Change `prohibitive_signal_overrides_crosswalk` in `config` if this policy decision changes.

The video label is `JAYWALKING` when at least one valid crossing person is classified as jaywalking. If no valid crossing is found, the video label is `COMPLIANT`. Insufficient visual evidence produces `UNCERTAIN`.

## Architecture

1. YOLO26x detects objects and BoT SORT assigns track IDs.
2. The frozen JAAD supervised classifier scores every usable person track using only inference safe tracking and motion features.
3. The rule detector is retained as an audit feature and can be selected as an explicit fallback, but it does not override classifier decisions in classifier mode.
4. The selected crossing interval is used to locate evidence for the target person.
5. Six chronological moments centred on the crossing transition are sampled for every accepted person.
6. Each moment produces a full scene, target detail, road path, and two overlapping traffic control views. The target has a red box and its footpoint trajectory is yellow.
7. A validated Hugging Face VLM answers three focused questions: road markings, traffic controls, and evidence visibility.
8. A deterministic policy produces the person and video labels.

The configured tracker is BoT SORT with ReID enabled and the exact parameters in `configs/botsort.yaml`.

The committed v1.5 configuration models the ego corridor as a trapezoid that widens towards the camera. Its left and right boundaries are interpolated at the pedestrian bounding box bottom edge. These generic values are an initial geometry and must be calibrated using training and validation data before they are treated as final dataset parameters.

## Repository structure

```text
.
├── default.config
├── config                    # optional local configuration, ignored by Git
├── configs/
│   └── botsort.yaml
├── data/
│   ├── annotations.csv
│   ├── annotations_with_splits.csv
│   └── videos/
├── prepare_splits.py
├── prepare_jaad_context.py
├── compare_jaad_context_models.py
├── run_one_video.py
├── run_evaluation.py
├── run_crowd_analysis.py
├── src/crowd_jaywalking/
│   ├── config.py
│   ├── crossing.py
│   ├── crossing_classifier.py
│   ├── crowd_analysis.py
│   ├── evidence.py
│   ├── evaluation.py
│   ├── models.py
│   ├── pipeline.py
│   ├── policy.py
│   ├── tracking.py
│   ├── track_features.py
│   └── vlm.py
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

The default candidate VLM is `Qwen/Qwen3-VL-8B-Instruct`. Validation compares it with `google/gemma-4-12B-it`, loading one model at a time. Their weights are large, so allow substantial disk space and GPU or system memory. `device_map: "auto"` lets Accelerate place each model on the available GPU and CPU resources. The first inference downloads the weights from Hugging Face. YOLO26x weights are also downloaded automatically on first use by Ultralytics.

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

### 9. Compare the two context models

Keep `jaad_context_split` set to `val`. Version 4 adds a target trajectory, a road-path crop, and overlapping high-resolution traffic-control crops. Regenerate the evidence before running the model; existing manual labels are preserved:

```powershell
uv run python -u .\prepare_jaad_context.py
```

Complete the context fields in `results\jaad_context_audit\val\context_annotations.csv`, then run both VLMs on those exact same events:

```powershell
uv run python -u .\compare_jaad_context_models.py
```

The models are loaded sequentially. Selection uses the macro F1 of the derived `permission_present` observable first. This value combines only the four visible permission cues and is not a legal or jaywalking label. Ties use mean macro F1 over fields containing at least two ground truth classes, then the weakest field F1 and mean accuracy. The script refuses to select a model on the `test` split. The selected model is recorded in:

```text
results\jaad_vlm_comparison_v4\val\selected_model.json
```

Copy its `selected_model` value to `vlm_model` in `config` before the final locked context evaluation and CROWD analysis.

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
* The physical road is represented by a configurable image corridor rather than semantic road segmentation.
* One crossing event is produced per continuous person track segment.
* The fixed central corridor can miss crossings that occur entirely away from the centre of the image; inspect development videos with no candidate before changing this assumption.
* VLM context quality depends on infrastructure being visible in the sampled frames.
* Version 4 makes three focused inference calls per event. This is slower than the earlier combined prompt and can exceed memory on machines without a suitable GPU.
* JAAD test performance estimates crossing detection, not the final jaywalking policy on CROWD.
* CROWD context predictions still require a stratified manual audit before population estimates are reported.

## Run the frozen method on CROWD

Keep the frozen classifier and the exact validated YOLO26 plus BoT SORT setup:

```json
"crossing_decision_mode": "classifier",
"crossing_classifier_model": "results/jaad_crossing_classifier_v1/crossing_classifier.joblib",
"crossing_classifier_fallback_to_rules": false,
"crossing_classifier_min_track_frames": 5,
"crowd_results": "results/crowd_jaywalking_v1"
```

Set `videos` to one or more directories containing local CROWD video clips, then run:

```powershell
uv run python -u .\run_crowd_analysis.py
```

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

Use [JAAD_VALIDATION.md](JAAD_VALIDATION.md) to download all 346 official clips and validate person tracking, crossing detection, VLM context fields, and the deterministic jaywalking policy as separate stages.

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
