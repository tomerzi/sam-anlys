# Football Action Coach — Project Scope & Roadmap

This document captures the **current codebase**, the **target product**, and a **phased plan** to get from here to a system that ingests video and outputs **graded pose quality** (e.g. good / ok / bad) with a path toward coaching-style feedback.

---

## 1. Current codebase (inventory)

| Path | Responsibility |
|------|----------------|
| `main.py` | CLI: `preprocess` (one labeled video), `train` (LOO-CV + final model), `infer` (new video). |
| `config.py` | Paths, YOLO/ByteTrack/SAM2/ViTPose, windowing, XGBoost hyperparameters, device. |
| `pipeline/detector.py` | YOLO person detect → ByteTrack → mouse click to lock `tracker_id` → padded crops per frame. |
| `pipeline/segmentor.py` | SAM 2 (optional): box prompt = full crop → binary mask → masked crop + disk. |
| `pipeline/pose_estimator.py` | mmpose ViTPose top-down on masked crop → COCO-17 `(x, y, score)` → `keypoints.json`. |
| `pipeline/feature_engineer.py` | Per-frame 59-D vector (normalized xy + conf + 8 angles), sliding windows, 708-D aggregates, augmentation. |
| `pipeline/classifier.py` | `StandardScaler` + `XGBClassifier`, LOO-CV by **video**, `scale_pos_weight`, save/load joblib. |
| `utils/visualization.py` | Skeleton draw, prediction overlay, horizontal strip for windows. |
| `requirements.txt` | Core deps; SAM2 / mmpose noted as manual install + checkpoints. |

**Data flow today**

1. **Preprocess:** video → crops → masked crops → per-frame keypoints + `meta.json` (`label` 0/1/2, `rubric_version` 2, `video_id`).
2. **Train:** load all `data/<id>/`, augment sequences (7×), build windows, LOO-CV, fit final XGBoost on all windows, save `models/`.
3. **Infer:** same vision pipeline → windows → predict binary good/bad + probability strip.

**Explicit limitations (as implemented)**

- **Rubric v2** supports `0=bad`, `1=ok`, `2=good` (CLI + `meta.json`); legacy meta without `rubric_version` still maps old `0/1` bad/good to `0/2` on train.
- **No joint-level “what to change”** output; only class + prob + viz — **Phase D**.

**Phase A hardening (done)**

- Inference artifacts use `config.outputs_dir / "infer" / <video_stem>/` (cross-platform).
- Horizontal flip aug mirrors `x' = W - x` using padded-crop width from `bbox`, then LR swap.
- **Time axis:** lost-track frames produce no crop row; `build_window_features` windows index the **saved sequence** (see `detector.py` + `feature_engineer.py` docstrings).

---

## 2. Target scope (product)

**Inputs**

- One or more videos of a football action (single primary player after click-select + track).

**Outputs**

1. **Pose estimation** — stable COCO-17 (or agreed schema) per frame, with optional mask/refinement.
2. **Quality grade** — at least **three levels** (e.g. *good / ok / bad*) or a **continuous score** mapped to bands; aligned to how coaches actually bin quality.
3. **(Stretch)** **Action-specific coaching hints** — largest deviations vs a “good” reference for the same action type (angles / normalized pose), not only a black-box label.

**Non-goals for v1 (unless you expand scope)**

- Multi-player automatic coaching without selection.
- Full 3D biomechanics (inverse dynamics); stick to 2D pose + heuristics/ML unless you add sensors.

---

## 3. Architecture direction (senior ML view)

| Layer | Purpose |
|-------|---------|
| **Vision** | Detect → track → (optional) segment → top-down pose. Keep modular; consider making SAM optional default-on for crowd, off for speed tests. |
| **Representation** | Per-frame normalized pose + temporal context (windows or short sequence model). Current 708-D window stats are a solid **baseline**; later optional upgrade: lightweight temporal encoder on keypoint sequences. |
| **Grading** | Start with **ordinal or multiclass** XGBoost (or separate binary heads + calibration). Map probabilities to **good / ok / bad** with documented thresholds. |
| **Explainability (v2)** | Compare window-level angle / pose summaries to **per-action** “good” centiles or SHAP on **named** features if feature names are wired through. |

**Labeling strategy (critical for multi-tier grades)**

- Define **rubric** in writing (what is good vs ok vs bad for *your* action set).
- Collect enough clips per band; avoid severe class imbalance without `sample_weight` or stratified splits.

---

## 4. Phased roadmap (execute in order)

Work **one phase at a time**; each phase should end with something runnable and testable.

### Phase A — Hardening the existing pipeline ✅

| ID | Task | Status |
|----|------|--------|
| A.1 | Cross-platform output dirs (`config.outputs_dir`, infer under `outputs/infer/<stem>/`). | Done |
| A.2 | Horizontal flip: `x' = W - x` + LR swap (`feature_engineer.horizontal_flip_keypoints`). | Done |
| A.3 | Smoke test + pointers in `main.py` module docstring; layout/checkpoints below. | Done |
| A.4 | Document missing-frame / window-index policy in code + this file. | Done |

### Phase B — Multi-level grading (core product goal) ✅

| ID | Task | Status |
|----|------|--------|
| B.1 | `meta.json`: `label` in `{0,1,2}`, `rubric_version: 2`; CLI `--label`; legacy meta without version maps `1→good(2)` on train. | Done |
| B.2 | Classifier: binary XGB if 2 rubric classes, else `multi:softprob`; save `classes_rubric.pkl`. | Done |
| B.3 | Infer: BAD / OK / GOOD names + per-class probabilities; strip uses headline colors. | Done |
| B.4 | LOO-CV: macro-F1, confusion matrix, binary AUC when exactly two classes. | Done |
| B.5 | **Calibration** (optional): temperature / isotonic — not implemented. | Optional |

### Phase C — Grading quality & robustness

| ID | Task | Done when |
|----|------|-----------|
| C.1 | **Action type** field in `meta.json` (e.g. `shot`, `pass`) so future models can be per-action or multi-task. | Schema ready; single action still works. |
| C.2 | Ablation: train **with vs without** SAM2 and/or without angles — measure impact. | Informed default in `config.py`. |
| C.3 | Consider **minimum window count** and class-aware augmentation if data is scarce. | Fewer silent skips in `train`. |

### Phase D — Coaching / “what to change” (stretch)

| ID | Task | Done when |
|----|------|-----------|
| D.1 | For each infer window, compute **named angle / pose deltas** vs a stored **good reference** (per action): mean vector or quantiles from training “good” only. | Text or JSON: top-k joints/angles + direction. |
| D.2 | Optional: **SHAP** on a version of features with **stable names** (per-angle window means) for debugging, not necessarily user-facing copy. | Developers can trace model focus. |
| D.3 | Optional API / batch script: video path in → JSON + rendered video out. | Suitable for demo or integration. |

### Phase E — Packaging (optional, later)

| ID | Task | Done when |
|----|------|-----------|
| E.1 | `pyproject.toml` or pinned `requirements-lock`, Docker image with GPU base. | Reproducible deploy. |
| E.2 | Minimal web or CLI “report” HTML with timeline of grades. | Easier demos. |

---

## 5. Suggested task order (first sprint)

1. ~~**A.1** + **A.2**~~ (done).
2. ~~**B.1**–**B.4**~~ (done); optional **B.5** calibration if you need calibrated probabilities.
3. **D.1** once you want user-visible “what to change” (prototype deltas vs good reference).

---

## 6. Risks & mitigations

| Risk | Mitigation |
|------|------------|
| Small dataset + high capacity model | Strong regularization, video-level CV, limit augmentation bugs, collect more labeled windows. |
| Label noise | Double-annotate a subset; clear rubric. |
| 2D pose ambiguity (depth) | Grade relative cues that are visible in broadcast footage; avoid over-claiming in coaching text. |
| Dependency weight (SAM2, mmpose) | Document install; optional `use_sam2=False` path for CI/dev without full stack. |

---

## 7. How we work from here

- Treat each **Phase row** as a **ticket**: implement, run on sample data, then tick or adjust this doc.
- After each mergeable chunk, update **“Done when”** checkboxes or add a short **Changelog** section at the bottom of this file.

---

## 8. Folder layout, checkpoints, smoke test

**Directories (runtime)**

| Path | Contents |
|------|----------|
| `data/<video_id>/` | One preprocessed clip: `meta.json`, `masks/`, `keypoints/keypoints.json`. |
| `models/` | `xgb_model.pkl`, `scaler.pkl` after `train`. |
| `outputs/infer/<video_stem>/` | Last `infer` run: `masks/`, `keypoints/`, `predictions_strip.jpg`. |
| `checkpoints/` | SAM2 + ViTPose weights (not in git; see `requirements.txt`). |

**Checkpoints (manual)**

- SAM 2: install `sam2` from GitHub; weight e.g. `sam2_hiera_large.pt` — see comments in `requirements.txt`.
- ViTPose-H: mmpose config + `vitpose_huge.pth` — URLs in `requirements.txt`.

**Smoke test**

1. From directory `football_action_coach/`, with deps and checkpoints installed.
2. Preprocess ≥2 videos (different `--id`) or place two valid `data/<id>/` trees.
3. `python main.py train`
4. `python main.py infer --video <path>.mp4` — confirm `outputs/infer/<stem>/predictions_strip.jpg`.

---

## Changelog

| Date | Note |
|------|------|
| Initial | Plan created from repo grep/read of `football_action_coach`. |
| Phase A | A.1–A.4: `outputs_dir` infer path, flip aug fix, docs + window/frame policy. |
| Phase B | B.1–B.4: rubric v2 (0/1/2), multiclass XGB, infer distribution + viz, LOO metrics. |
