# Analysis scripts

Everything needed to go from a trained metric-learning checkpoint to the
figures and numbers in the paper:

```
images ──► extract_embeddings.py ──► .npz ──┬──► evaluate_tracking.py   (longitudinal accuracy, ablation)
                                            └──► evaluate_enrolment.py  (retention sweep)
```

`common.py` (IO, label parsing, plot styling) and `tracker.py` (the anchor-bank
tracker itself) are libraries rather than entry points — import them if you want
to drive the method from your own code.

The two evaluation scripts only need the `.npz` files, so **anyone can
reproduce the figures from the released embeddings without a GPU, without the
image data, and without PyTorch installed.**

## Install

```bash
# evaluation only — reproduces every figure from the released .npz files
pip install -r analysis/requirements.txt

# + PyTorch and OpenCV, if you also want to re-extract embeddings from images
pip install -r analysis/requirements-extract.txt \
    --extra-index-url https://download.pytorch.org/whl/cu121
```

Versions are pinned to the environment the released results were produced in
(CUDA 12.1). The evaluation scripts are not fussy — they have been verified
under both numpy 1.x and 2.x; see the comments in `requirements.txt` for the
looser floors. `tqdm` is optional and only drives the progress bar.

## Try it without downloading anything

```bash
python analysis/make_synthetic_npz.py --out /tmp/fake.npz
python analysis/evaluate_tracking.py --npz "Demo=/tmp/fake.npz" --out-dir /tmp/demo
```

If that produces a plot, your environment is good.

---

## 1. `extract_embeddings.py`

Runs a trained checkpoint over a folder of crops and saves the embeddings.

Expected layout — one sub-directory per identity:

```
<data-root>/
    G19_238/  cam1__20230412_1032.jpg  ...
    G19_241/
    G20_239/
```

```bash
python analysis/extract_embeddings.py \
    --checkpoint runs/aabb_v19/fold_0/best_model_state.pkl \
    --data-root  data/bbox_cropped/open_set \
    --out        results/embeddings/aabb_nomuck_openset.npz
```

Useful flags: `--device cpu|cuda|mps`, `--batch-size`, `--num-workers`,
`--pattern 'G*'` (which identity folders to include), `--date-regex` if your
filenames don't follow the `..__<date>_..` convention, `--repo-root` if
`analysis/` doesn't sit next to `models/`.

Two things worth knowing:

- **No mean/std normalisation is applied.** The network was trained on raw
  `[0, 1]` tensors after a letterbox resize; adding ImageNet statistics here
  will quietly degrade every downstream number.
- The classification head is unused at inference. The training-time class count
  is read back out of the checkpoint so the weights load cleanly; override with
  `--num-train-classes` if that fails.

## 2. `evaluate_tracking.py`

Daily identification accuracy over time: K-means → multi-anchor voting →
confidence gate → anchor bank. This replaces the two near-duplicate tracking
scripts — the day-to-day baseline is now `--baseline`, and the anchor-bank
sweep is `--ablation`.

```bash
python analysis/evaluate_tracking.py \
    --config analysis/configs/experiments.example.json \
    --out-dir results/tracking \
    --baseline "Ins. Segm." --ablation "Ins. Segm."
```

Writes `daily_accuracy.csv`, `summary.csv`,
`longitudinal_tracking_comparison.png`, and (with `--ablation`)
`anchor_bank_ablation.csv` / `.png`.

Tracker knobs: `--bank-size` (5), `--min-votes` (2), `--link-conf-max` (0.35).

**Day-0 initialisation.** By default day 0 is aligned to ground truth so the
tracked identities carry readable names; every later day is fully
unsupervised. `--init first` removes the ground-truth touch entirely, at the
cost of accuracy no longer being measurable against the true labels — worth
running once if a reviewer asks.

`--extra-series accuracies.csv` overlays numbers computed elsewhere onto the
same figure — in particular the `accuracies.csv` written by
`daily_finetune.py`, which is how the *tuned* series in the paper is produced.
Columns: `Experiment`, `Accuracy (%)`, and either `Date` or `Day Index`. Rows
whose `class` column is not `OVERALL` should be filtered out first.

## 3. `evaluate_enrolment.py`

How accurate is automatic enrolment if you keep only the samples closest to
each cluster centroid? Sweeps the retention percentage and reports the
precision/coverage trade-off.

```bash
python analysis/evaluate_enrolment.py \
    --config analysis/configs/experiments.example.json \
    --out-dir results/enrolment
```

Writes `enrolment_accuracy.csv`, `enrolment_daily.csv`, `summary.txt`, and
`comparison_accuracy_by_percent.png`.

K defaults to the number of distinct identities in the file (12 for the
combined G19 + G20 open-set evaluation — matching happens jointly across both
pens, not per pen). Clustering here runs in the **raw** embedding space, which
is what the published numbers use; `--normalise` switches to the unit sphere
and will shift the results.

---

## Configuring experiments

Either a JSON config:

```json
{ "experiments": { "AABB": "results/embeddings/aabb.npz" } }
```

or repeated CLI flags: `--npz "AABB=results/embeddings/aabb.npz"`. Both scripts
accept either or both. Names matching the paper's modalities (`Ins. Segm.`,
`OBB`, `AABB`, their `(ND)` variants, `Baseline`) keep their published colours;
anything else gets a colour automatically, so custom names won't break the
figures. Missing files are skipped with a warning, and two names pointing at
the same file are flagged rather than silently plotted as two identical curves.

## Embedding `.npz` format

| array | shape | contents |
|---|---|---|
| `embeddings` | `(N, D)` float32 | one embedding per crop |
| `labels` | `(N,)` | `"<identity>_<date>"` |
| `date` | `(N,)` | capture date |
| `fnames` | `(N,)` | source image path |

`--id-parts` controls how many underscore-separated tokens form the identity
(default 2, i.e. `G19_238` from `G19_238_20230412`). Extraction also writes a
`<name>.meta.json` sidecar recording the checkpoint, arguments, and package
versions, so a released `.npz` can be traced back to how it was made.

## Reproducing the released results

1. Download the checkpoints and the crop dataset (see the top-level README).
2. Run `extract_embeddings.py` once per modality/ablation.
3. Point a config at the resulting `.npz` files.
4. Run both evaluation scripts.

Or skip 1–2 and use the released `.npz` files directly.