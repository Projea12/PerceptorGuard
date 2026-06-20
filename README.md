# PerceptorGuard

A **perception evaluation harness** for YOLO detection models — slice-based metrics, CI regression gating, failure triage, and reproducible synthetic fixtures. Built to demonstrate that evaluation is an engineering discipline, not a one-liner.

---

## Quick start

```bash
# 1. Install
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]" && pip install scikit-learn pyyaml jinja2

# 2. Generate a 20-scene dataset and run eval
python scripts/generate.py  --count 20 --seed 42 --out artifacts/dataset
python scripts/run_eval.py  --dataset artifacts/dataset --out artifacts/eval

# 3. Triage failures and generate report
python scripts/triage.py    --matches artifacts/eval/matches.csv
python scripts/generate_report.py \
    --dataset  artifacts/dataset \
    --eval     artifacts/eval \
    --baseline artifacts/baseline \
    --triage   artifacts/triage
# → open artifacts/report/report.html
```

**Planted-regression demo** (gate goes red, then green):
```bash
python scripts/save_baseline.py          # promote current eval to baseline
python scripts/demo_regression.py        # PASS → FAIL → PASS
```

---

## System overview

```
┌───────────────────────────────────────────────────────────────────┐
│  Scenario generator (PyBullet DIRECT)                             │
│  6 challenge profiles × 2-tier object catalog (easy + hard)      │
│  → manifest.csv + per-scene frame.png + ground_truth.json        │
└──────────────────────────┬────────────────────────────────────────┘
                           │
┌──────────────────────────▼────────────────────────────────────────┐
│  Eval runner                                                      │
│  YOLO inference at conf=0.01 (full distribution for AP)          │
│  Greedy class-aware bipartite matching  (IoU ≥ 0.5)             │
│  → matches.csv  (tidy: one row per TP / FP / FN)                │
└──────────────────────────┬────────────────────────────────────────┘
                           │
          ┌────────────────┴──────────────────┐
          │                                   │
┌─────────▼──────────┐             ┌──────────▼──────────────────┐
│  Metrics engine    │             │  Failure triage              │
│  mAP (11-pt VOC)  │             │  missed / localization /     │
│  P/R/F1 @ 0.25    │             │  wrong_class / false_pos     │
│  slice tables:     │             │  KMeans cluster analysis     │
│  profile / dist /  │             │  → failures_classified.csv  │
│  lighting / clutter│             │  → cluster_summary.csv      │
│  tier / class      │             └──────────────────────────────┘
└─────────┬──────────┘
          │
┌─────────▼──────────────────────────────────────────────────────┐
│  Regression gate                                               │
│  Compare current metrics vs baseline (artifacts/baseline/)     │
│  46 checks: mAP, AP/recall per class, recall/FP per profile   │
│  Exit 1 on any regression beyond configured slack             │
└─────────┬──────────────────────────────────────────────────────┘
          │
┌─────────▼──────────────────────────────────────────────────────┐
│  Report renderer (Jinja2)                                      │
│  Self-contained HTML + Markdown                                │
│  Slice tables, failure gallery, gate diff table                │
│  Optional: W&B / MLflow experiment tracking                    │
└────────────────────────────────────────────────────────────────┘
```

**CI split** — deliberate engineering decision:
- **PR gate** (`.github/workflows/ci_gate.yml`): 20 scenes, seed=42, ~2 min. Catches regressions before they land in `main`.
- **Nightly** (`.github/workflows/nightly.yml`): 100 scenes, auto-promotes baseline on success. Authoritative quality bar.

---

## Key design decisions

### 1. Evaluation as a first-class engineering subsystem

Most perception teams treat evaluation as an afterthought: run a metric script, log the number, move on. The script is throw-away code; the number is a spreadsheet cell; there is no regression gate.

PerceptorGuard treats the eval pipeline with the same engineering discipline as the model itself:

| Property | Ad-hoc eval script | PerceptorGuard |
|----------|-------------------|----------------|
| Dataset | "whatever images we had" | Versioned, reproducible, committed |
| Metrics | Overall mAP only | Per-slice: profile, distance, lighting, clutter, tier, class |
| Failures | "recall is low" | Named failure modes with KMeans-clustered conditions |
| Regressions | Discovered in staging | Caught at PR time, 46 checks, named slice + delta |
| Reproducibility | "I think we used these settings" | Seed-fixed fixtures, LFS-tracked weights |

The eval harness is the **durable investment**. Models come and go; the harness lets you compare them honestly.

### 2. Model-agnostic interface

The harness has exactly one coupling point to YOLO: `EvalRunner._predict()` (12 lines in `runner/eval_runner.py`). The rest of the pipeline — matching, metrics, gating, triage, reporting — operates on `Detection` and `GroundTruth` Pydantic schemas.

Swapping YOLOv8n for YOLOv8x, a RT-DETR, or a custom model requires changing exactly that one function. You can A/B test models through the same harness and compare them on the same reproducible dataset without any eval-code changes.

This is the same principle I apply to RAG and agentic systems: the evaluation framework must be agnostic to the implementation choice, or you end up with evaluation that only works for the thing you already have.

### 3. CI gate blast-radius argument

Catching a regression has very different costs depending on where it surfaces:

| Where caught | Cost |
|-------------|------|
| At PR (CI gate) | 2 min CI compute |
| In staging after merge | Deploy + rollback + engineer-hours |
| In production | Incident response + user-trust loss |

The gate runs 46 checks per PR. Each check is a named (slice, metric) pair with an explicit floor: `floor = baseline_value − slack`. If any check fires, the gate exits 1 and names the regressed slice and delta. Engineers know exactly what broke and by how much — not "mAP went down a bit."

The 20-scene CI fast path is a **deliberate tradeoff**: 20 scenes is noisy enough that you'll miss subtle regressions, but it catches real structural breaks (IoU threshold bug, conf threshold change, class mapping error) in 2 minutes. The nightly full 100-scene run is the authoritative measurement. This split is explained in both workflow files.

### 4. Two-tier object catalog

```
Easy  (COCO-recognizable): cup, bottle, bowl, teddy bear, sports ball
Hard  (off-vocabulary):    cube, duck, lego, domino
```

The split gives you two signals simultaneously:
- **Easy tier** measures how much signal you can extract from a COCO-pretrained backbone. Any recall at all indicates some domain-transfer.
- **Hard tier** measures true zero-shot generalisation. Near-zero AP here is expected and honest — it's the documented result, not a bug.

Running only easy classes would give you false confidence. Running only hard classes would give you nothing to gate on. The mix is deliberate.

### 5. Sub-threshold IoU enrichment for FN rows

The matcher records `best_iou_any_class` and `best_pred_class_at_overlap` for every FN row. This enables the failure classifier to distinguish:

- `missed_detection` — no prediction overlapped the GT (IoU < 0.1 from any box)
- `localization_error` — right class, right location, IoU just below threshold
- `wrong_class` — something overlaps (IoU ≥ 0.1) but with the wrong class label

These are **different engineering problems** requiring different interventions. Knowing you have 123 `wrong_class` failures in near-distance crowded scenes (the actual finding) is actionable. Knowing "FN = 355" is not.

---

## Findings from the pilot run (YOLOv8n, 100 scenes)

```
Overall: mAP=0.1%  Precision=0%  Recall=0%
TP=0  FP=27  FN=355  (at op-point conf≥0.25)
```

The headline finding is a **complete domain gap**: YOLOv8n, pretrained on COCO, achieves 0% recall on PyBullet synthetic renders at any operating threshold. This is expected and honest.

**The only live signal: sports ball, AP=0.9%** — the soccerball URDF has a realistic texture that partially overlaps the COCO training distribution. Two sub-threshold predictions match GT at IoU≥0.5 during the full-distribution AP sweep.

**Failure mode breakdown (771 failures):**
1. `false_positive` (54%) — 416 hallucinations. The model confidently detects objects that aren't there (chairs, people, cars) because PyBullet renders look like partial-context frames from real images.
2. `missed_detection` (30%) — 232 GTs with zero model signal. Worst in crowded + far profiles; domino and lego lead by class.
3. `wrong_class` (16%) — 123 GTs where a prediction overlaps but carries the wrong label. Cube accounts for 34% of wrong-class failures — the model recognises a rectangular object but can't resolve the class.

**Cluster insight**: KMeans (k=5) on [camera_distance, ambient_light, num_objects, failure_mode, tier, profile] identifies two pure FP clusters (100% false positives at mid-distance and near-distance), a mixed missed+wrong-class cluster under crowded conditions, and a distinct dark-scene cluster with different FP characteristics.

**What this means for next steps:**
- The gap is at the distribution level, not the architecture level. Domain randomization of textures + sim-to-real transfer is the right intervention, not model scaling.
- The FP flood is a calibration problem. A classification head fine-tuned on synthetic negatives would reduce it dramatically.
- The wrong-class failures on cube/domino suggest that geometric shape features are present but class-label resolution requires task-specific training.

---

## Planted-regression demo

The gate verifiably catches regression and names the failing slice:

```
$ python scripts/demo_regression.py

DEMO STEP 1 — Gate on current (good) metrics
→ PASSED — all 46 checks within threshold

DEMO STEP 2 — Plant regression: zero sports-ball AP
(simulates iou_threshold cranked to 0.9)
→ FAILED — 1 regression(s) detected
  ✗ class:sports ball / ap
      baseline=0.0091  current=0.0000  floor=0.0041  delta=-0.0091

DEMO STEP 3 — Restore
→ PASSED — all 46 checks within threshold

Result: PASS → FAIL → PASS  ✓  (gate behaves correctly)
```

To trigger this in real CI, open a PR that changes `--iou 0.9` in the eval runner. The CI workflow runs, the sports-ball AP drops from 0.9% to 0%, the gate exits 1, and the PR check fails — naming the regressed slice and delta. Revert → green.

---

## Cross-domain principle: eval as infrastructure

The architectural insight that generalises across ML domains:

> **Evaluation should be a first-class subsystem, not an afterthought. It must be model-agnostic, slice-aware, and wired into CI.**

I've applied the same principle in three domains:

| Domain | What gets evaluated | The harness checks |
|--------|--------------------|--------------------|
| **Perception (this project)** | YOLO detector | Per-slice mAP, FP rate, localization quality |
| **RAG systems** | Retriever + LLM | Retrieval recall, answer faithfulness, citation precision |
| **Agentic systems** | Tool-use agent | Task completion rate, tool selection accuracy, latency |

In each case:
- The harness is decoupled from the implementation (model-agnostic interface)
- Metrics are sliced by the conditions that matter (difficulty, distance, topic domain, query type)
- A baseline is stored and regressions are caught before they reach production
- Failures are named, not just counted

The model changes; the eval discipline doesn't. This is the engineering investment that compounds.

---

## What I'd do differently

**1. Domain randomization before domain gap is "interesting"**
PyBullet renders are too synthetic. Before drawing any production conclusions, I'd add texture randomization (PBR materials, HDRI backgrounds), noise augmentation, and random object scales. The 0% recall result is expected — but a more realistic synthetic distribution would push that to a meaningful non-zero baseline worth gating on.

**2. Real-image holdout**
The synthetic-to-real gap is documented but not measured. A small (50-100 image) real-world validation set, with the same object categories, would quantify the gap. This is the honest thing to do before claiming the eval harness is production-relevant.

**3. Active learning loop**
The triage output (failure mode distribution, KMeans clusters) should feed back into the scenario generator: generate more scenes matching the hardest cluster conditions. Right now triage is a report; it should be a signal that drives the next data generation run.

**4. Temporal and latency eval**
`match_scene` operates on single frames. Real robot perception needs tracking across frames, trajectory prediction, and FPS budgeting. None of that is here.

**5. Confidence calibration**
The op-point threshold is set at conf≥0.25 by convention. A calibration sweep (precision-recall curve analysis by condition) would let you set per-slice thresholds that reflect the actual operating point you need — not a fixed global number.

---

## Repository structure

```
perceptorguard/
├── scenarios/          Pydantic schemas, parameterized scenario generator
│   ├── schemas.py      BoundingBox, Detection, GroundTruth, Scenario, ObjectSpec
│   └── generator.py    6 profiles × 2-tier catalog, occluder placement
├── runner/             Inference + matching + GT extraction
│   ├── scene_runner.py PyBullet DIRECT renderer, AABB→screen projection
│   ├── eval_runner.py  YOLO inference loop, bin assignment
│   └── matcher.py      Greedy bipartite match; FN rows carry sub-threshold IoU
├── metrics/            Metrics, triage, reporting
│   ├── engine.py       11-pt VOC AP, operating-point P/R/F1, sliced tables
│   ├── failure_classifier.py  missed / localization / wrong_class / fp
│   ├── cluster_analyzer.py    KMeans on scenario feature vector
│   ├── triage_reporter.py     Ranked failure-mode summary
│   └── reporter.py     ASCII console report
├── gates/              Regression gate
│   ├── thresholds.py   GateThresholds dataclass, YAML-backed
│   ├── comparator.py   46-check comparison: mAP, AP, recall, FP per slice
│   └── gate_runner.py  Print report, return bool, exit 1 on failure
├── reports/            Report rendering
│   ├── annotator.py    Annotate failure scenes (GT boxes, missed/detected)
│   ├── renderer.py     Jinja2 → HTML + Markdown
│   ├── tracker.py      W&B + MLflow optional integration
│   └── templates/      report.html.j2, report.md.j2
├── scripts/
│   ├── generate.py     Dataset generation CLI
│   ├── run_eval.py     Eval CLI (--model, --iou, --imgsz)
│   ├── triage.py       Failure triage CLI
│   ├── run_gate.py     Gate CLI (exit 0/1)
│   ├── save_baseline.py  Promote eval → baseline
│   ├── generate_report.py  Report generation CLI
│   └── demo_regression.py  Planted-regression demo
├── tests/              74 unit tests (all pass)
│   ├── test_matcher.py, test_metrics.py
│   ├── test_triage.py, test_gates.py
│   └── verify_chunk2.py  (55 GT-pipeline invariant tests)
├── assets/             Custom URDFs (bottle.urdf, bowl.urdf)
├── configs/
│   └── gate_thresholds.yml  Tunable slack per metric
├── artifacts/
│   ├── baseline/       100-scene golden reference (committed)
│   ├── ci_baseline/    20-scene CI reference (committed, seed=42)
│   ├── ci_dataset/     Reproducible 20-scene fixture (committed, LFS)
│   ├── eval/           Full 100-scene eval output
│   └── triage/         Failure classification + cluster summary
└── .github/
    ├── workflows/ci_gate.yml     PR: 20 scenes, ~2 min
    └── workflows/nightly.yml     Scheduled: 100 scenes, auto-promote baseline
```

---

## Running the full test suite

```bash
pytest tests/ -v
# → 74 passed
```

Tests are hermetic — no model inference, no disk artifacts required. The 55 `verify_chunk2.py` tests validate the GT pipeline (AABB projection, occlusion geometry, multi-object placement) against hardcoded fixtures. The 12 `test_matcher.py` tests cover IoU edge cases and greedy matching invariants. The 15 `test_gates.py` tests cover threshold loading, regression detection (mAP drop, FP spike, recall drop), and gate-runner pass/fail return values.
