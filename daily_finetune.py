#!/usr/bin/env python
"""
Self-supervised daily fine-tuning ("continual learning") pipeline.

This is the experiment reported as the *tuned* variant in the paper: instead of
freezing one model and tracking with it for weeks, the model is re-trained every
day on pseudo-labels that it generated itself, with no human input after day 0.

For each day, in order:

  1.  Embed every crop of that day with the current model.
  2.  Cluster the day's embeddings with K-means (K = number of identities), and
      keep only the `--core-ratio` of each cluster closest to its centroid.
  3.  Link today's cluster centroids to the running "master" centroid per
      identity with the Hungarian algorithm. Links whose cost exceeds
      `--cluster-conf` are rejected (that identity goes unclaimed for the day).
  4.  Log accuracy against the ground-truth folder labels. These labels are
      *only* used for measuring - never for training (except during an optional
      supervised warm start, `--warm-start-days`).
  5.  Harvest the images whose distance to the linked master centroid is below
      `--image-conf` into a growing pseudo-labelled dataset (80/20 train/test).
  6.  Fine-tune from `--base-model` on everything harvested so far.
  7.  Recompute the master centroids with the new model, and purge stored images
      that have drifted further than `--purge-conf` from their centroid. Images
      from day 0 are never purged - they are the only trustworthy anchor.

State is checkpointed after every day, so the run can be resumed after a crash.

Expected input layout - one folder per identity, filenames containing the
capture date (see `--date-sep`):

    <source-dir>/
        G19_238/  cam1__20230412_1032.png ...
        G19_241/
        ...

Example
-------
python daily_finetune.py \
    --source-dir data/bbox_cropped/open_set \
    --work-dir   results/daily_finetune \
    --base-model output/inssegm/fold_0/best_model_state.pkl \
    --base-model-classes 137 \
    --epochs 3

Outputs, inside `--work-dir`:
    accuracies.csv        per-day, per-identity accuracy (+ an OVERALL row)
    tracking_overlay.csv  the same per-day overall accuracy, in the column
                          layout `evaluate_tracking.py --extra-series` reads
    checkpoint.pkl        resumable pipeline state
    images/train|test/    the accumulated pseudo-labelled dataset
    splits/               the generated folds file passed to train.py
    models/               one fine-tuned checkpoint per day

`tracking_overlay.csv` can be overlaid onto the tracking figure with
`analysis/evaluate_tracking.py --extra-series`.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import pickle
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.cluster import KMeans, HDBSCAN
from sklearn.metrics import pairwise_distances
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

from models.embeddings import resnet50  # noqa: E402
from utilities.ioutils import loadResizeImage  # noqa: E402
from utilities.utils import getDevice  # noqa: E402

IMAGE_EXTS = ("*.png", "*.jpg", "*.jpeg")
ACC_CSV_HEADER = ["day_index", "date", "class", "correct", "total", "accuracy"]

# Second, narrower CSV written in the exact shape that
# `analysis/evaluate_tracking.py --extra-series` expects, so the daily
# fine-tuning curve can be overlaid on the tracking figure without reshaping.
OVERLAY_CSV_HEADER = ["Experiment", "Day Index", "Date", "Accuracy (%)"]


# ==========================================
# EMBEDDING EXTRACTION
# ==========================================
def init_model(model_path, num_classes, embedding_size, device):
    model = resnet50(
        pretrained=True,
        num_classes=num_classes,
        ckpt_path=model_path,
        embedding_size=embedding_size,
    )
    return model.to(device).eval()


def extract_embeddings(model_path, image_paths, num_classes, args, device):
    """Load a checkpoint, embed every image, return L2-normalised embeddings."""
    print(f"\n--- Embedding {len(image_paths)} images with "
          f"{os.path.basename(model_path)} (classes={num_classes}) ---")
    model = init_model(model_path, num_classes, args.embedding_size, device)

    outputs = []
    with torch.no_grad():
        for start in tqdm(range(0, len(image_paths), args.batch_size), desc="Inference"):
            chunk = image_paths[start:start + args.batch_size]

            # Same letterbox preprocessing and [0, 1] scaling as training,
            # with augmentation switched off
            batch = np.stack([
                loadResizeImage(p, (args.image_size, args.image_size), augment=False)
                for p in chunk
            ])
            tensor = torch.from_numpy(batch.transpose(0, 3, 1, 2)).float().to(device) / 255.0

            emb, _ = model(tensor)
            outputs.append(emb.cpu().numpy())

    return normalise_l2(np.concatenate(outputs, axis=0))


# ==========================================
# HELPERS
# ==========================================
def normalise_l2(vectors):
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.maximum(norms, 1e-10)


def date_from_filename(filename, sep="__"):
    """Date token following the final `sep` (e.g. "..__20230412_1032.png")."""
    parts = filename.split(sep)
    if len(parts) >= 2:
        candidate = parts[-1].split("_")[0]
        if candidate.isdigit():
            return candidate

    # Fallback: any 8-digit token anywhere in the name
    for token in os.path.splitext(filename)[0].replace(sep, "_").split("_"):
        if len(token) == 8 and token.isdigit():
            return token
    return None


def images_by_date(source_dir, date_sep, ignore_classes):
    print(f"Scanning {source_dir} for images...")
    date_to_files = {}

    files = []
    for pattern in IMAGE_EXTS:
        files.extend(glob.glob(os.path.join(source_dir, "**", pattern), recursive=True))

    for f in sorted(files):
        date = date_from_filename(os.path.basename(f), sep=date_sep)
        if date is None:
            continue
        true_class = os.path.basename(os.path.dirname(f))
        if true_class in ignore_classes:
            continue
        date_to_files.setdefault(date, []).append((f, true_class))

    if not date_to_files:
        raise SystemExit(f"No dated images found under {source_dir} (check --date-sep)")
    return date_to_files


def write_splits_json(work_dir, known_classes):
    splits_dir = os.path.join(work_dir, "splits")
    os.makedirs(splits_dir, exist_ok=True)
    json_path = os.path.join(splits_dir, "pseudo_labels.json")
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump({"0": {"known": sorted(known_classes), "unknown": []}}, fh, indent=4)
    return json_path


def core_set(embeddings, core_ratio):
    """Indices of the `core_ratio` fraction of rows closest to the mean."""
    centre = np.mean(embeddings, axis=0).reshape(1, -1)
    dists = pairwise_distances(embeddings, centre).flatten()
    return dists <= np.percentile(dists, core_ratio * 100)


def safe_copy(src_path, true_lbl, pred_id, dist_val, dest_dir):
    """Copy a harvested crop, recording the ground truth, prediction and distance
    in the filename so that harvesting errors can be audited afterwards."""
    orig_name = os.path.basename(src_path)
    name_only, ext = os.path.splitext(orig_name)
    counter = 0
    while True:
        suffix = "" if counter == 0 else f"_{counter}"
        new_name = f"GT-{true_lbl}@PRED-{pred_id}@CONF-{dist_val:.3f}@{name_only}{suffix}{ext}"
        dest_path = os.path.join(dest_dir, new_name)
        if not os.path.exists(dest_path):
            shutil.copy2(src_path, dest_path)
            return
        counter += 1


# ==========================================
# CHECKPOINT + LOGGING
# ==========================================
def save_checkpoint(path, state):
    with open(path, "wb") as fh:
        pickle.dump(state, fh)
    print(f"  [checkpoint] saved -> {path}")


def load_checkpoint(path):
    if os.path.exists(path):
        with open(path, "rb") as fh:
            state = pickle.load(fh)
        print(f"  [checkpoint] resumed after day index {state['last_completed_day_index']}")
        return state
    return None


def init_accuracy_csv(path):
    if not os.path.exists(path):
        with open(path, "w", newline="", encoding="utf-8") as fh:
            csv.DictWriter(fh, fieldnames=ACC_CSV_HEADER).writeheader()


def init_overlay_csv(path):
    if not os.path.exists(path):
        with open(path, "w", newline="", encoding="utf-8") as fh:
            csv.DictWriter(fh, fieldnames=OVERLAY_CSV_HEADER).writeheader()


def log_overlay_csv(path, series_name, day_index, date, overall):
    """Append this day's overall accuracy in --extra-series format."""
    with open(path, "a", newline="", encoding="utf-8") as fh:
        csv.DictWriter(fh, fieldnames=OVERLAY_CSV_HEADER).writerow({
            "Experiment": series_name,
            "Day Index": day_index,
            "Date": date,
            "Accuracy (%)": round(overall * 100, 4),
        })


def log_accuracy_csv(path, day_index, date, class_accuracies):
    """Append one row per identity plus an OVERALL row for this day."""
    rows = []
    total_correct = total_total = 0

    for cls, (correct, total) in sorted(class_accuracies.items()):
        rows.append({
            "day_index": day_index,
            "date": date,
            "class": cls,
            "correct": correct,
            "total": total,
            "accuracy": round(correct / total if total else 0.0, 6),
        })
        total_correct += correct
        total_total += total

    overall = total_correct / total_total if total_total else 0.0
    rows.append({
        "day_index": day_index,
        "date": date,
        "class": "OVERALL",
        "correct": total_correct,
        "total": total_total,
        "accuracy": round(overall, 6),
    })

    with open(path, "a", newline="", encoding="utf-8") as fh:
        csv.DictWriter(fh, fieldnames=ACC_CSV_HEADER).writerows(rows)

    print(f"\nDay {date} (index {day_index}) - mean accuracy {overall:.1%}")
    for r in rows[:-1]:
        print(f"  {r['class']:15s}  {r['accuracy']:.1%}  ({r['correct']}/{r['total']})")
    print("-" * 40)

    return overall


# ==========================================
# MAIN LOOP
# ==========================================
def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Self-supervised daily fine-tuning for pig re-identification.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--source-dir", required=True,
                   help="Directory of crops, one sub-directory per identity")
    p.add_argument("--work-dir", required=True,
                   help="Where the harvested dataset, models and logs are written")
    p.add_argument("--base-model", required=True,
                   help="Checkpoint every day's fine-tuning starts from")
    p.add_argument("--base-model-classes", type=int, required=True,
                   help="Number of classes --base-model was trained with")

    # Fine-tuning hyperparameters, forwarded to train.py
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--learning-rate", type=float, default=1e-4)
    p.add_argument("--finetune-layers", default="layer4")
    p.add_argument("--train-batch-size", type=int, default=64)
    p.add_argument("--triplet-selection", default="SemihardNegative")
    p.add_argument("--triplet-margin", type=float, default=0.5)
    p.add_argument("--loss-function", default="OnlineTripletSoftmaxLoss")
    p.add_argument("--model", default="TripletResnetSoftmax")
    p.add_argument("--use-arcface", action="store_true", default=False)
    p.add_argument("--muck-dir", default=None,
                   help="Noise fields for muck augmentation during fine-tuning")

    # Inference settings
    p.add_argument("--embedding-size", type=int, default=128)
    p.add_argument("--image-size", type=int, default=224,
                   help="Input resolution, used for both inference and fine-tuning")
    p.add_argument("--batch-size", type=int, default=64, help="Inference batch size")
    p.add_argument("--num-workers", type=int, default=4,
                   help="DataLoader workers for the fine-tuning subprocess. Use 0 on "
                        "Windows, where nested subprocesses with worker processes can "
                        "abort the run")

    # Pipeline behaviour
    p.add_argument("--start-day", type=int, default=0, help="Index of the first day to process")
    p.add_argument("--warm-start-days", type=int, default=0,
                   help="Number of initial days that use ground-truth labels instead of clustering")
    p.add_argument("--ignore-classes", nargs="*", default=[],
                   help="Identity folder names to exclude entirely")
    p.add_argument("--use-hdbscan", action="store_true", default=False,
                   help="Cluster with HDBSCAN (falling back to K-means) instead of K-means")
    p.add_argument("--core-ratio", type=float, default=0.70,
                   help="Fraction of each cluster kept before the distance checks")
    p.add_argument("--cluster-conf", type=float, default=0.60,
                   help="Maximum Hungarian cost for a cluster->identity link to be accepted")
    p.add_argument("--image-conf", type=float, default=0.50,
                   help="Maximum centroid distance for an image to be harvested")
    p.add_argument("--purge-conf", type=float, default=0.65,
                   help="Maximum centroid distance for a stored image to be kept")
    p.add_argument("--date-sep", default="__",
                   help="Separator preceding the date token in filenames")
    p.add_argument("--series-name", default="Ins. Segm. (tuned)",
                   help="Name this run is given in tracking_overlay.csv, used as the "
                        "series label by analysis/evaluate_tracking.py --extra-series")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args(argv)


def finetune(args, json_path, out_path):
    """Fine-tune from the base model on everything harvested so far."""
    cmd = [
        sys.executable, str(REPO_ROOT / "train.py"),
        f"--finetune_model={args.base_model}",
        f"--model_num_classes={args.base_model_classes}",
        f"--model={args.model}",
        f"--num_epochs={args.epochs}",
        f"--folds_file={json_path}",
        f"--out_path={out_path}",
        f"--dataset_root={args.work_dir}",
        f"--finetune_layers={args.finetune_layers}",
        f"--learning_rate={args.learning_rate}",
        f"--batch_size={args.train_batch_size}",
        f"--embedding_size={args.embedding_size}",
        f"--img_rows={args.image_size}",
        f"--img_cols={args.image_size}",
        f"--num_workers={args.num_workers}",
        f"--triplet_selection={args.triplet_selection}",
        f"--triplet_margin={args.triplet_margin}",
        f"--loss_function={args.loss_function}",
        f"--seed={args.seed}",
    ]
    if args.use_arcface:
        cmd.append("--use_arcface")
    if args.muck_dir:
        cmd.append(f"--muck_dir={args.muck_dir}")
    else:
        cmd.append("--disable_muck")

    try:
        subprocess.run(cmd, check=True, cwd=str(REPO_ROOT))
    except subprocess.CalledProcessError as exc:
        raise SystemExit(f"Fine-tuning failed: {exc}")


def main(argv=None) -> None:
    args = parse_args(argv)
    np.random.seed(args.seed)
    device = getDevice()

    work_dir = args.work_dir
    train_dir = os.path.join(work_dir, "images", "train")
    test_dir = os.path.join(work_dir, "images", "test")
    models_dir = os.path.join(work_dir, "models")
    for d in (train_dir, test_dir, models_dir):
        os.makedirs(d, exist_ok=True)

    checkpoint_path = os.path.join(work_dir, "checkpoint.pkl")
    acc_csv_path = os.path.join(work_dir, "accuracies.csv")
    overlay_csv_path = os.path.join(work_dir, "tracking_overlay.csv")
    init_accuracy_csv(acc_csv_path)
    init_overlay_csv(overlay_csv_path)

    # ---- scan source ----
    date_to_files = images_by_date(args.source_dir, args.date_sep, set(args.ignore_classes))
    eval_dates = sorted(date_to_files.keys())[args.start_day:]
    day_0_date = eval_dates[0]
    print(f"Pipeline | {len(eval_dates)} days | anchor day = {day_0_date}")

    # ---- resume from checkpoint if available ----
    checkpoint = load_checkpoint(checkpoint_path)
    if checkpoint:
        current_model_path = checkpoint["current_model_path"]
        current_model_classes = checkpoint["current_model_classes"]
        master_centroids = checkpoint["master_centroids"]
        resume_from = checkpoint["last_completed_day_index"] + 1
    else:
        current_model_path = args.base_model
        current_model_classes = args.base_model_classes
        master_centroids = {}
        resume_from = 0

    # ==========================================
    # DAY LOOP
    # ==========================================
    for day_index, current_date in enumerate(eval_dates):
        if day_index < resume_from:
            continue

        print(f"\n{'='*60}\nPROCESSING DAY {day_index}: {current_date}\n{'='*60}")

        day_data = date_to_files[current_date]
        day_filepaths = [item[0] for item in day_data]
        day_true_labels = np.array([item[1] for item in day_data])
        unique_classes = np.unique(day_true_labels)
        k = len(unique_classes)

        # ---- 1. embed the day ----
        day_emb = extract_embeddings(current_model_path, day_filepaths,
                                     current_model_classes, args, device)

        assigned_ids = {}
        core_filepaths = {}
        core_truelabels = {}
        core_embs = {}

        if day_index < args.warm_start_days:
            # ---- supervised warm start: identities come from the folder names ----
            print(f"  -> SUPERVISED WARM START (day {day_index+1}/{args.warm_start_days})")
            clusters = np.zeros(len(day_true_labels), dtype=int)

            for idx, cls in enumerate(unique_classes):
                cls_mask = day_true_labels == cls
                c_embs = day_emb[cls_mask]
                c_paths = np.array(day_filepaths)[cls_mask]
                c_labels = day_true_labels[cls_mask]
                mask = core_set(c_embs, args.core_ratio)

                master_centroids[cls] = normalise_l2(np.mean(c_embs[mask], axis=0).reshape(1, -1))[0]

                assigned_ids[idx] = cls
                core_filepaths[idx] = c_paths[mask]
                core_truelabels[idx] = c_labels[mask]
                core_embs[idx] = c_embs[mask]
                clusters[cls_mask] = idx

        else:
            # ---- 2. cluster ----
            if args.use_hdbscan:
                min_size = max(5, len(day_filepaths) // (k * 3))
                clusters = HDBSCAN(min_cluster_size=min_size).fit_predict(day_emb)
                cluster_ids = [c for c in np.unique(clusters) if c != -1]
                if len(cluster_ids) < k * 0.5:
                    print(f"  [!] HDBSCAN found {len(cluster_ids)} clusters for {k} "
                          f"identities - falling back to K-means")
                    clusters = KMeans(n_clusters=k, random_state=args.seed, n_init=10).fit_predict(day_emb)
                    cluster_ids = list(range(k))
            else:
                clusters = KMeans(n_clusters=k, random_state=args.seed, n_init=10).fit_predict(day_emb)
                cluster_ids = list(range(k))

            # Core-set filtering
            core_centroids = []
            valid_clusters = []

            for c in cluster_ids:
                c_mask = clusters == c
                if not np.any(c_mask):
                    continue
                c_embs = day_emb[c_mask]
                mask = core_set(c_embs, args.core_ratio)

                core_centroids.append(np.mean(c_embs[mask], axis=0))
                core_filepaths[c] = np.array(day_filepaths)[c_mask][mask]
                core_truelabels[c] = day_true_labels[c_mask][mask]
                core_embs[c] = c_embs[mask]
                valid_clusters.append(c)

            core_centroids = normalise_l2(np.array(core_centroids))

            # ---- 3. link to the master centroids ----
            if not master_centroids:
                raise SystemExit("No master centroids: run with --warm-start-days >= 1 "
                                 "so that day 0 establishes the identities.")

            master_keys = list(master_centroids.keys())
            master_matrix = np.array([master_centroids[mk] for mk in master_keys])
            cost_matrix = pairwise_distances(master_matrix, core_centroids, metric="cosine")
            row_ind, col_ind = linear_sum_assignment(cost_matrix)

            for r, c_idx in zip(row_ind, col_ind):
                cost = cost_matrix[r, c_idx]
                cluster = valid_clusters[c_idx]
                if cost <= args.cluster_conf:
                    assigned_ids[cluster] = master_keys[r]
                else:
                    print(f"  [REJECTED] cluster {cluster} -> {master_keys[r]} "
                          f"cost={cost:.3f} > {args.cluster_conf}")

        # ---- 4. accuracy logging (ground truth used for measurement only) ----
        day_predictions = np.array(["Unknown"] * len(day_true_labels), dtype=object)
        for c, pred_id in assigned_ids.items():
            day_predictions[clusters == c] = pred_id

        class_accuracies = {
            cls: (int(np.sum(day_predictions[day_true_labels == cls] == cls)),
                  int(np.sum(day_true_labels == cls)))
            for cls in unique_classes
        }
        overall = log_accuracy_csv(acc_csv_path, day_index, current_date, class_accuracies)
        log_overlay_csv(overlay_csv_path, args.series_name, day_index, current_date, overall)

        # ---- 5. harvest confident images ----
        print("\nHarvesting confident images...")
        accepted = rejected = 0

        for c, pred_id in assigned_ids.items():
            master_cent = master_centroids[pred_id].reshape(1, -1)
            class_train_dir = os.path.join(train_dir, pred_id)
            class_test_dir = os.path.join(test_dir, pred_id)
            os.makedirs(class_train_dir, exist_ok=True)
            os.makedirs(class_test_dir, exist_ok=True)

            valid = []
            for path, true_lbl, emb in zip(core_filepaths[c], core_truelabels[c], core_embs[c]):
                dist = pairwise_distances(emb.reshape(1, -1), master_cent, metric="cosine")[0][0]
                if dist <= args.image_conf:
                    valid.append((path, true_lbl, dist))
                else:
                    rejected += 1

            accepted += len(valid)
            np.random.shuffle(valid)
            split_idx = int(len(valid) * 0.8)

            for path, lbl, dist in valid[:split_idx]:
                safe_copy(path, lbl, pred_id, dist, class_train_dir)
            for path, lbl, dist in valid[split_idx:]:
                safe_copy(path, lbl, pred_id, dist, class_test_dir)

        print(f"  Accepted: {accepted} | Rejected: {rejected}")

        # ---- 6. fine-tune from the base model on everything harvested so far ----
        json_path = write_splits_json(work_dir, list(master_centroids.keys()))
        out_path = os.path.join(models_dir, f"ft_{args.finetune_layers}_{current_date}_ep{args.epochs}")

        print(f"\n>>> Fine-tuning from the base model on data up to {current_date} >>>")
        finetune(args, json_path, out_path)

        current_model_path = os.path.join(out_path, "fold_0", "best_model_state.pkl")
        current_model_classes = len(master_centroids)

        # ---- 7. refresh centroids and purge drifted images in a single pass ----
        print("\nRefreshing centroids + purging with the new model...")
        accumulated = []
        for pattern in IMAGE_EXTS:
            accumulated.extend(glob.glob(os.path.join(work_dir, "images", "**", pattern), recursive=True))

        if accumulated:
            all_embs = extract_embeddings(current_model_path, accumulated,
                                          current_model_classes, args, device)

            # Pass A: rebuild the master centroids
            accum = {}
            for path, emb in zip(accumulated, all_embs):
                accum.setdefault(os.path.basename(os.path.dirname(path)), []).append(emb)
            for cls_name, embs in accum.items():
                master_centroids[cls_name] = normalise_l2(np.mean(embs, axis=0).reshape(1, -1))[0]

            # Pass B: purge images that have drifted from the fresh centroids
            protected = 0
            to_delete = []
            for path, emb in zip(accumulated, all_embs):
                # Day-0 images are the only ground-truth-aligned anchor we have
                if day_0_date in os.path.basename(path):
                    protected += 1
                    continue

                master_cent = master_centroids.get(os.path.basename(os.path.dirname(path)))
                if master_cent is None:
                    continue
                dist = pairwise_distances(emb.reshape(1, -1), master_cent.reshape(1, -1), metric="cosine")[0][0]
                if dist > args.purge_conf:
                    to_delete.append(path)

            for path in to_delete:
                os.remove(path)

            print(f"  Purged: {len(to_delete)} | Protected (anchor day): {protected}")

        # ---- checkpoint after every completed day ----
        save_checkpoint(checkpoint_path, {
            "last_completed_day_index": day_index,
            "current_model_path": current_model_path,
            "current_model_classes": current_model_classes,
            "master_centroids": master_centroids,
        })

    print(f"\nDaily fine-tuning complete. Per-day accuracies: {acc_csv_path}")


if __name__ == "__main__":
    main()
