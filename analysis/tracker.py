"""
Longitudinal identity tracker: K-means -> multi-anchor voting -> confidence
gate -> anchor bank.

This is the library version of the tracking loop. `evaluate_tracking.py` is the
command-line front end; import `run_tracker` directly if you want to embed the
tracker in your own experiment.

Method summary
--------------
For each day:
  1. Cluster that day's embeddings with K-means (K = number of animals).
  2. Match today's cluster centroids against every anchor in the bank with the
     Hungarian algorithm; each anchor casts one vote per cluster.
  3. A cluster is assigned an identity if it collects >= `min_votes` votes.
  4. If too many clusters are undecided (> K/3), fall back to a plain Hungarian
     match against the most recent anchor only.
  5. If the mean matching cost is below `link_conf_max`, today's centroids are
     admitted to the bank (FIFO, capped at `bank_size`).

Note on initialisation: day 0 is aligned to ground truth so that identity
labels are human-readable from the start. Every subsequent day is fully
unsupervised. Set `init="first"` to seed the bank with arbitrary cluster
labels instead, which removes the ground-truth touch entirely at the cost of
accuracy no longer being comparable to the labelled identities.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import (  # noqa: E402
    cluster_centroids,
    cosine_distance,
    kmeans_cluster,
    l2_normalise,
    stack_embeddings,
)


class AnchorBank:
    """FIFO bank of per-identity centroids from recent confident days."""

    def __init__(self, max_size: int = 5):
        self.max_size = max_size
        self.bank: list[dict[str, np.ndarray]] = []
        self.dates: list[str] = []

    def add(self, pig_centroids: dict[str, np.ndarray], date: str) -> None:
        self.bank.append(pig_centroids)
        self.dates.append(date)
        if len(self.bank) > self.max_size:
            self.bank.pop(0)
            self.dates.pop(0)

    def vote(
        self, today_cents: np.ndarray, pig_list: list[str], min_votes: int = 2
    ) -> tuple[dict[int, str | None], float]:
        """
        Each anchor independently Hungarian-matches its identities to today's
        clusters. Returns {cluster_index: identity or None} and the mean
        matching cost across all anchors (used as the confidence gate).
        """
        k = len(today_cents)
        votes = np.zeros((k, len(pig_list)), dtype=int)
        all_costs: list[float] = []

        for anchor in self.bank:
            available = [p for p in pig_list if p in anchor]
            if not available:
                continue
            anchor_cents = np.array([anchor[p] for p in available])
            cost = cosine_distance(anchor_cents, today_cents)
            rows, cols = linear_sum_assignment(cost)
            for r, c in zip(rows, cols):
                all_costs.append(float(cost[r, c]))
                if c < k:
                    votes[c, pig_list.index(available[r])] += 1

        assignment: dict[int, str | None] = {}
        for c in range(k):
            best = int(votes[c].argmax())
            assignment[c] = pig_list[best] if votes[c, best] >= min_votes else None

        mean_cost = float(np.mean(all_costs)) if all_costs else 1.0
        return assignment, mean_cost

    def __len__(self) -> int:
        return len(self.bank)


def _greedy_fill(
    uncertain: list[int],
    cluster_map: dict[int, str],
    today_cents: np.ndarray,
    latest_anchor: dict[str, np.ndarray],
    pig_list: list[str],
) -> dict[int, str]:
    """Assign the clusters that failed the vote by nearest remaining anchor."""
    assigned = set(cluster_map.values())
    for c in uncertain:
        remaining = [p for p in pig_list if p not in assigned] or list(pig_list)
        candidates = [p for p in remaining if p in latest_anchor] or list(latest_anchor)
        if not candidates:
            continue
        anchor_cents = np.array([latest_anchor[p] for p in candidates])
        sims = (today_cents[c : c + 1] @ anchor_cents.T).ravel()
        best = candidates[int(sims.argmax())]
        cluster_map[c] = best
        assigned.add(best)
    return cluster_map


def run_tracker(
    df: pd.DataFrame,
    dates: list[str] | None = None,
    bank_size: int = 5,
    min_votes: int = 2,
    link_conf_max: float = 0.35,
    init: str = "gt",
    seed: int = 42,
    n_init: int = 50,
    progress: bool = True,
) -> pd.DataFrame:
    """
    Run the tracker over a DataFrame from `common.load_embeddings`.

    Returns one row per day: date, day_idx, accuracy_pct, status, n_images,
    mean_cost, bank_size_used.
    """
    if dates is None:
        dates = sorted(df["date"].unique())

    pig_list = sorted(df["true_id"].unique())
    k = len(pig_list)
    bank = AnchorBank(max_size=bank_size)
    rows = []

    iterator = enumerate(dates)
    if progress:
        try:
            from tqdm import tqdm

            iterator = tqdm(iterator, total=len(dates), desc="Tracking days", leave=False)
        except ImportError:
            pass

    for day_idx, date in iterator:
        day_df = df[df["date"] == date]
        x = l2_normalise(stack_embeddings(day_df))
        y_true = day_df["true_id"].values

        if len(x) < k:
            print(f"  ! {date}: only {len(x)} images for {k} identities — skipping day")
            continue

        pred_k = kmeans_cluster(x, k, n_init=n_init, seed=seed)
        today_cents = cluster_centroids(x, pred_k, k)

        if day_idx == 0 or len(bank) == 0:
            if init == "gt":
                idx_of = {p: i for i, p in enumerate(pig_list)}
                overlap = np.zeros((k, k))
                for t, c in zip(y_true, pred_k):
                    overlap[idx_of[t], c] += 1
                rows_i, cols_i = linear_sum_assignment(overlap.max() - overlap)
                cluster_map = {cols_i[i]: pig_list[rows_i[i]] for i in range(len(rows_i))}
            else:  # arbitrary but stable seeding
                cluster_map = {c: pig_list[c] for c in range(k)}
            bank.add({v: today_cents[c] for c, v in cluster_map.items()}, date)
            status, mean_cost = "GT-INIT" if init == "gt" else "SELF-INIT", 0.0
        else:
            assignment, mean_cost = bank.vote(today_cents, pig_list, min_votes=min_votes)
            uncertain = [c for c, p in assignment.items() if p is None]

            if len(uncertain) > k // 3:
                latest = bank.bank[-1]
                available = [p for p in pig_list if p in latest]
                anchor_cents = np.array([latest[p] for p in available])
                rows_i, cols_i = linear_sum_assignment(cosine_distance(anchor_cents, today_cents))
                cluster_map = {cols_i[i]: available[rows_i[i]] for i in range(len(rows_i))}
                status = "FALLBACK"
            else:
                cluster_map = {c: p for c, p in assignment.items() if p is not None}
                cluster_map = _greedy_fill(
                    uncertain, cluster_map, today_cents, bank.bank[-1], pig_list
                )
                status = "VOTED" if not uncertain else "PARTIAL-VOTE"

            if mean_cost < link_conf_max:
                confident = {
                    cluster_map[c]: today_cents[c] for c in range(k) if c in cluster_map
                }
                if confident:
                    bank.add(confident, date)

        pred_ids = np.array([cluster_map.get(c, "Unknown") for c in pred_k])
        rows.append(
            {
                "date": date,
                "day_idx": day_idx,
                "accuracy_pct": 100.0 * float((pred_ids == y_true).mean()),
                "status": status,
                "n_images": int(len(day_df)),
                "mean_cost": round(float(mean_cost), 4),
                "bank_size_used": len(bank),
            }
        )

    return pd.DataFrame(rows)
