# GSRD: Granularity + Shift Robust Detection

GSRD is a reproducibility-first research pipeline for **open-vocabulary object detection** under:
1. **Vocabulary granularity changes** (coarse / standard / fine / mixed), and
2. **Natural distribution shifts** (ID COCO val vs OOD BDD100K val domains).

The detector is treated as a **fixed pretrained black box**. No detector retraining is required.

## What This Tests

This codebase evaluates three claims:
1. Open-vocabulary detectors can be highly sensitive to how users phrase and granularize class vocabularies.
2. Domain shift and vocabulary granularity interact non-trivially (the same detector can degrade differently per granularity).
3. A training-free post-hoc risk-control layer can enforce user-chosen risk targets by calibrating a single threshold knob on a small held-out calibration split.

## Key Features

- End-to-end staged pipeline:
  - data prep
  - vocabulary generation
  - detector inference with cache/shards/resume
  - evaluation (COCO-style metrics + granularity/shift analyses + CIs)
  - post-hoc risk calibration (flat + hierarchy-aware)
  - paper-ready reporting (plots/tables/final summary)
- ECCV-grade statistics:
  - 95% CI over vocabulary-list variation per granularity
  - uncertainty-aware ID→OOD drop estimates
  - explicit shift×granularity interaction effects vs standard vocab
- Novelty modules:
  - counterfactual vocabulary stress tests with distractor terms
  - worst-group domain-conditional risk control (`strategy=groupwise` / `strategy=all`)
  - novelty summary metrics (GSI + vocabulary instability + tail robustness)
- Cluster-friendly design (MIT Engaging): shard inference across job arrays, merge artifacts later.
- Versioned stable prediction format and run manifests with hardware + seed + config + git commit.
- Multiple vocab lists per granularity to avoid cherry-picking.

## Datasets

- **ID**: COCO 2017 validation (`instances_val2017.json`)
- **OOD (natural shifts)**: BDD100K validation detection split, converted to COCO format for consistent evaluation.

BDD100K per-image attributes (`timeofday`, `weather`, `scene`) are preserved for domain breakdown plots.

## Installation

```bash
cd /path/to/GSRD
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

For development:

```bash
pip install -e .[dev]
```

## Detector Adapters

Built-in adapters:
- `grounding_dino` (default enabled; strong baseline)
- `owlvit` (optional)
- `mock` (for smoke tests)

The adapter interface is in `src/gsrd/detectors/base.py`.

## Quick Smoke Test (local, tiny)

This runs a synthetic toy dataset with the `mock` detector and generates at least one plot/table.

```bash
gsrd run --config configs/smoke.yaml --stage data,vocab,inference,merge,eval,risk,report
```

Expected artifacts:
- `artifacts/outputs/report/final_report.md`
- `artifacts/outputs/report/figures/*.png`
- `artifacts/outputs/report/tables/*.csv`

## Bigger Local Validation (recommended before cluster)

This runs a larger synthetic benchmark with multiple vocab lists per granularity,
counterfactual stress vocabularies, and full report generation while keeping local
runtime practical.

```bash
gsrd run --config configs/local_bigger.yaml \
  --stage data,vocab,inference,merge,eval,risk,report \
  --set paths.data_root=data/local_bigger \
  --set paths.cache_root=artifacts/local_bigger/cache \
  --set paths.outputs_root=artifacts/local_bigger/outputs \
  --set paths.logs_root=artifacts/local_bigger/logs \
  --set evaluation.bootstrap_samples=12
```

The synthetic benchmark includes five classes (`car`, `truck`, `person`, `rider`, `traffic light`)
with style/domain variation (`timeofday`, `weather`, `scene`) so that:
- granularity robustness is non-trivial,
- counterfactual vocab perturbations are measurable,
- hierarchy/groupwise risk behavior is observable.

## Full COCO-val Run (ID)

```bash
gsrd data prepare --config configs/default.yaml --set datasets.bdd100k_val=false
gsrd vocab generate --config configs/default.yaml --dataset coco_val2017
gsrd inference run --config configs/default.yaml --dataset coco_val2017 --detector grounding_dino
gsrd inference merge --config configs/default.yaml --dataset coco_val2017 --detector grounding_dino
gsrd eval run --config configs/default.yaml --dataset coco_val2017 --detector grounding_dino
gsrd risk run --config configs/default.yaml --dataset coco_val2017 --detector grounding_dino
gsrd report generate --config configs/default.yaml --dataset coco_val2017 --detector grounding_dino
```

## Full OOD Benchmark Run (BDD100K)

```bash
gsrd data prepare --config configs/default.yaml --set datasets.coco_val=false
gsrd vocab generate --config configs/default.yaml --dataset bdd100k_val
gsrd inference run --config configs/default.yaml --dataset bdd100k_val --detector grounding_dino
gsrd inference merge --config configs/default.yaml --dataset bdd100k_val --detector grounding_dino
gsrd eval run --config configs/default.yaml --dataset bdd100k_val --detector grounding_dino
gsrd risk run --config configs/default.yaml --dataset bdd100k_val --detector grounding_dino
gsrd report generate --config configs/default.yaml --dataset bdd100k_val --detector grounding_dino
```

## One-Command Full Pipeline

```bash
gsrd run --config configs/default.yaml --stage data,vocab,inference,merge,eval,risk,report
```

## MIT Engaging Cluster Usage

### 1) Prepare data + vocab once

```bash
gsrd data prepare --config configs/engaging_cluster.yaml
gsrd vocab generate --config configs/engaging_cluster.yaml --dataset coco_val2017 --dataset bdd100k_val
```

### 2) Launch inference job array (sharded)

```bash
sbatch scripts/slurm/run_inference_array.sbatch configs/engaging_cluster.yaml coco_val2017 grounding_dino
sbatch scripts/slurm/run_inference_array.sbatch configs/engaging_cluster.yaml bdd100k_val grounding_dino
```

### 3) Merge + evaluate + risk + report

```bash
sbatch scripts/slurm/merge_and_eval.sbatch configs/engaging_cluster.yaml coco_val2017 grounding_dino
sbatch scripts/slurm/merge_and_eval.sbatch configs/engaging_cluster.yaml bdd100k_val grounding_dino
```

## Reproducibility and Resume Behavior

- Inference writes shard artifacts under `artifacts/cache/predictions/...`.
- If `inference.skip_existing=true`, reruns skip existing shards.
- `gsrd inference merge` merges shards into `merged.jsonl.gz`.
- Run manifests are written in `artifacts/outputs/runs/<timestamp>/run_manifest.json`.
- Compute summary is written in `artifacts/outputs/compute_summary.json`.

## Vocabulary Generation and Cherry-Picking Avoidance

For each dataset, GSRD generates **multiple vocabulary lists per granularity** (`lists_per_granularity`):
- `standard`: canonical labels
- `coarse`: merged semantic groups
- `fine`: lexical refinements per class
- `mixed`: a controlled blend of coarse and fine terms
- `counterfactual`: lexical perturbations + distractor terms to stress-test open-vocab robustness

Evaluation reports distributions/means over all lists, not a single curated list.

## Risk Definition and Monotonicity

### Control knob
A single score threshold `t` controls abstention:
- keep detections with score >= `t`
- higher `t` => lower coverage

### Flat risk
`risk_flat(t) = incorrect_kept / kept`, where a kept detection is correct only if:
- IoU with best GT >= 0.5, and
- predicted class-set is an exact singleton matching the GT class.

### Hierarchy-aware risk
`risk_hier(t)` uses the same IoU criterion, but label correctness allows class-set overlap with GT canonical class.
This is more robust when vocab includes coarse terms or mixed granularity.

### Worst-group groupwise risk (novelty)
`risk_groupwise(t)` calibrates one threshold against the **worst domain group** (e.g., `timeofday`) using group-level UCBs. This reduces shift-induced failure modes hidden by global averages.

### Calibration guarantee (plain English)
Given a calibration split and target risk `alpha`, we choose the **lowest** threshold whose upper confidence bound on calibration risk is <= `alpha`. By default, we use an exact one-sided **Clopper-Pearson** binomial bound (`ucb_method=clopper_pearson`) for stronger finite-sample conservativeness. This yields an interpretable high-probability guarantee under exchangeability.

## Reporting Outputs (ECCV-ready)

Generated under `artifacts/outputs/report/`:
- Figures:
  - `granularity_sensitivity.png`
  - `risk_vs_coverage.png`
  - `id_to_ood_degradation.png`
  - `shift_granularity_interaction_heatmap.png`
  - `id_to_ood_domain_breakdown.png`
  - each figure is also exported as `.pdf`
- Tables (CSV + LaTeX):
  - performance by granularity
  - worst-case drop across granularities
  - calibration effectiveness (achieved risk vs alpha, coverage)
  - risk summary (target-met rate, guarantee margin)
  - shift interaction summary with CI
  - novelty metrics table (GSI, instability CV, worst-10% vocab behavior)
- Final summary:
  - `final_report.md`

## How to Add a New Detector Adapter

1. Implement `OpenVocabDetector` in a new file under `src/gsrd/detectors/`.
2. Return detections as `(term, score, bbox_xyxy)`.
3. Register the adapter in `src/gsrd/detectors/factory.py`.
4. Add detector config to `configs/*.yaml`.

No evaluation/risk/reporting changes are needed if the adapter conforms to the interface.

## Testing

```bash
pytest
```

## Notes

- BDD100K download endpoints can occasionally be slow; rerun `gsrd data prepare` to resume.
- For strict reproducibility, keep seeds fixed and avoid changing detector/model revisions mid-study.
