# PerceptorGuard — Quick Start

You have a detection model. You have labeled images. In ten minutes you will have sliced metrics, a named failure breakdown, and an HTML report.

---

## 1. Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install perceptorguard
```

Requires Python ≥ 3.11. No GPU needed.

---

## 2. What you need

| File | Format | Required |
|------|--------|----------|
| Ground truth labels | COCO annotations JSON | Yes |
| Model predictions | COCO results JSON | Yes |
| Images | Any directory of JPG/PNG | No — but enables lighting slice |
| Metadata CSV | One row per image, any columns | No — but enables custom slices |

**COCO annotations JSON** — the standard format from CVAT, Label Studio, Roboflow, and most labeling tools. If your tool exports "COCO format", this is it.

**COCO results JSON** — the array your model writes when you run `model.predict(..., save_json=True)` with Ultralytics, or the output of `coco_eval` with most frameworks. It looks like:
```json
[{"image_id": 1, "category_id": 3, "bbox": [x, y, w, h], "score": 0.91}, ...]
```

---

## 3. Run eval on your data

```bash
perceptorguard eval \
  --gt       path/to/annotations.json \
  --preds    path/to/results.json \
  --out      artifacts/eval
```

With optional enrichment:
```bash
perceptorguard eval \
  --gt       path/to/annotations.json \
  --preds    path/to/results.json \
  --images   path/to/images/ \
  --metadata path/to/metadata.csv \
  --out      artifacts/eval
```

What happens on first run:
1. PerceptorGuard reads your class names and your model's class names.
2. If they don't match exactly, it auto-suggests a mapping and writes `configs/class_map.yml`. **Review this file** — edit it if any mapping is wrong, then re-run.
3. Eval runs. Results land in `artifacts/eval/`.

---

## 4. Read the console output

```
Overall  mAP=0.42  P=0.61  R=0.55  TP=1204  FP=312  FN=489

Slice: object_size
  small    mAP=0.18  R=0.31  FP=187
  medium   mAP=0.51  R=0.63  FP=88
  large    mAP=0.71  R=0.79  FP=37

Slice: clutter
  sparse   mAP=0.58  R=0.71
  moderate mAP=0.39  R=0.52
  crowded  mAP=0.21  R=0.34

Slice: lighting        (requires --images)
  dark     mAP=0.19  R=0.28
  normal   mAP=0.47  R=0.59
  bright   mAP=0.53  R=0.66

Failure breakdown:
  missed_detection   312  (42%)  — no model signal at all
  wrong_class        189  (26%)  — right location, wrong label
  localization_error 144  (19%)  — right class, IoU just below threshold
  false_positive     312  (13%)  — hallucinations with no GT match
```

The numbers above are illustrative. Your numbers will be different and honest.

---

## 5. Generate the HTML report

```bash
perceptorguard report \
  --eval  artifacts/eval \
  --out   artifacts/report

open artifacts/report/report.html
```

The report is a single self-contained HTML file — no server, no account. Share it by attaching it to a Slack message or Jira ticket.

---

## 6. Save a baseline and gate future runs

```bash
# Promote today's eval to baseline
perceptorguard gate --save-baseline --current artifacts/eval

# On the next eval run, gate checks 46 named (slice, metric) pairs
perceptorguard gate \
  --baseline artifacts/baseline \
  --current  artifacts/eval
```

Exit 0 = all checks pass. Exit 1 = at least one regression, named:

```
FAILED — 1 regression(s) detected
  ✗ clutter:crowded / recall
      baseline=0.34  current=0.21  floor=0.32  delta=-0.13
```

Wire this into CI (`if [ $? -ne 0 ]; then exit 1; fi`) and regressions are caught before they merge.

---

## Optional: metadata CSV for custom slices

If your team tracks additional context per image, add a CSV:

```csv
filename,   weather, sensor,  location
img001.jpg, rain,    camera,  urban
img002.jpg, sunny,   lidar,   highway
img003.jpg, fog,     camera,  suburban
```

Pass it with `--metadata path/to/metadata.csv`. Every column becomes a slice dimension automatically — no config, no code.

```
Slice: weather
  rain     mAP=0.28  R=0.39
  sunny    mAP=0.51  R=0.63
  fog      mAP=0.14  R=0.22   ← this is your problem
```

---

## Optional: class map

On first run, PerceptorGuard writes `configs/class_map.yml`:

```yaml
# Maps your dataset class names to the names your model outputs.
# null means no model equivalent — those GTs will be excluded.
mappings:
  vehicle: car        # auto-suggested
  pedestrian: person  # auto-suggested
  forklift: null      # no COCO equivalent — excluded
```

Edit this file directly. The next run uses your corrections. It is never overwritten automatically.

---

## Try it now with the bundled fixture

The repo ships with a 20-scene synthetic fixture so you can see the full pipeline without any data:

```bash
# Requires pip install 'perceptorguard[synthetic]'
perceptorguard eval \
  --dataset artifacts/ci_dataset \
  --out     artifacts/eval

perceptorguard report \
  --eval artifacts/eval \
  --out  artifacts/report

open artifacts/report/report.html
```

---

## Getting help

- Issues: https://github.com/your-org/perceptorguard/issues
- The gate threshold slacks are tunable in `configs/gate_thresholds.yml`
- Run any command with `--help` for full options
