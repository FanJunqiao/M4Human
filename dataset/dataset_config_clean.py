"""
Dataset split configuration.
All hardcoded subject/action lists and ratios are now read from config.yaml.
The split logic (split_indices, save/load helpers) is unchanged.
"""

import random
import pickle
import gzip
import yaml
from pathlib import Path
from typing import Any, Dict, List
from collections import defaultdict


# ---------------------------------------------------------------------------
# Config loader
# ---------------------------------------------------------------------------

def _load_dataset_cfg(config_path=None) -> Dict:
    """Load the 'dataset' section from config.yaml."""
    if config_path is None:
        config_path = Path(__file__).parent.parent / "config.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)["dataset"]


# ---------------------------------------------------------------------------
# Scale builder  (replaces the old hardcoded info_init)
# ---------------------------------------------------------------------------

def info_init(scale_id: str = "p1", config_path=None) -> Dict:
    """
    Build the scale dict for all three split schemes (s1/s2/s3).
    Reads subject lists, action lists, and ratios from config.yaml.
    """
    ds = _load_dataset_cfg(config_path)
    ratio = ds["scale_ratios"][scale_id]

    all_sub = ds["all_subjects"]
    all_act = ds["all_actions"]

    # --- cross-subject (s2) ---
    cs_test = ds["cross_subject"]["test"]
    cs_val  = ds["cross_subject"]["val"]
    cs_train = [s for s in all_sub if s not in cs_test and s not in cs_val]

    # --- cross-action (s3) ---
    ca_test = ds["cross_action"]["test"]
    ca_val  = ds["cross_action"]["val"]
    ca_train = [a for a in all_act if a not in ca_test and a not in ca_val]

    # --- random (s1) ---
    r = ds["random"]
    r_train = r["train_ratio"] * ratio
    r_val   = r["val_ratio"]   * ratio
    r_test  = r["test_ratio"]  * ratio

    cross_sub_split = {
        "train": {"sub": cs_train, "act": all_act, "ratio": ratio},
        "val":   {"sub": cs_val,   "act": all_act, "ratio": ratio},
        "test":  {"sub": cs_test,  "act": all_act, "ratio": ratio},
        "all":   [all_sub, all_act],
    }
    cross_act_split = {
        "train": {"sub": all_sub, "act": ca_train, "ratio": ratio},
        "val":   {"sub": all_sub, "act": ca_val,   "ratio": ratio},
        "test":  {"sub": all_sub, "act": ca_test,  "ratio": ratio},
        "all":   [all_sub, all_act],
    }
    random_split = {
        "train": {"sub": all_sub, "act": all_act, "ratio": r_train},
        "val":   {"sub": all_sub, "act": all_act, "ratio": r_val},
        "test":  {"sub": all_sub, "act": all_act, "ratio": r_test},
        "all":   [all_sub, all_act],
    }

    return {"s1": random_split, "s2": cross_sub_split, "s3": cross_act_split}


# ---------------------------------------------------------------------------
# Split logic (unchanged from original)
# ---------------------------------------------------------------------------

def split_ok_pool(pool, out_train, out_val, out_test):
    to_set = lambda L: {tuple(x) for x in L}
    return to_set(pool) == (to_set(out_train) | to_set(out_val) | to_set(out_test))


def split_indices(
    index_list: List[List[int]],
    scale_cfg: Dict[str, Dict[str, Dict[str, Any]]],
    scheme: str = "s1",
    seed: int = 42,
) -> Dict[str, List]:
    """
    Partition index_list into train/val/test splits.
    index_list items: [sub_id, act_id, frame_id]
    """
    rng = random.Random(seed)
    cfg = scale_cfg[scheme]

    all_sub, all_act = set(cfg["all"][0]), set(cfg["all"][1])
    filtered = [[s, a, f] for s, a, f in index_list if s in all_sub and a in all_act]

    subs_train = set(cfg["train"]["sub"]);  r_train = float(cfg["train"]["ratio"])
    acts_train = set(cfg["train"]["act"])
    subs_val   = set(cfg["val"]["sub"]);    r_val   = float(cfg["val"]["ratio"])
    acts_val   = set(cfg["val"]["act"])
    subs_test  = set(cfg["test"]["sub"]);   r_test  = float(cfg["test"]["ratio"])
    acts_test  = set(cfg["test"]["act"])

    # Group frames by (sub, act)
    frames_by_pair = defaultdict(list)
    for s, a, f in filtered:
        frames_by_pair[(s, a)].append(f)
    for k in frames_by_pair:
        frames_by_pair[k].sort()

    def pick_block(sorted_frames: List[int], k: int, start=None):
        """Random consecutive block of length k; returns (picked, remaining, start)."""
        n = len(sorted_frames)
        if start is None:
            start = rng.randint(0, n - k)
        return sorted_frames[start:start + k], \
               sorted_frames[:start] + sorted_frames[start + k:], \
               start

    out_train, out_val, out_test = [], [], []

    for (s, a), frames in frames_by_pair.items():
        remain = frames
        n = len(frames)
        ratio_used = 0.0

        if s in subs_test and a in acts_test:
            n_test = max(1, int(n * r_test))
            test_block, remain, _ = pick_block(remain, n_test)
            out_test.extend([[s, a, f] for f in test_block])
            ratio_used += r_test

        if s in subs_train and a in acts_train:
            n_train = max(1, int(n * r_train))
            train_block, remain, _ = pick_block(remain, n_train)
            out_train.extend([[s, a, f] for f in train_block])
            ratio_used += r_train

        if s in subs_val and a in acts_val:
            n_val = max(1, int(n * r_val))
            val_block, remain, _ = pick_block(remain, n_val)
            out_val.extend([[s, a, f] for f in val_block])
            ratio_used += r_val

        # Any leftover frames go to val
        if ratio_used == 1.0:
            out_val.extend([[s, a, f] for f in remain])

    return {"train": out_train, "val": out_val, "test": out_test}


# ---------------------------------------------------------------------------
# Index file helpers (unchanged)
# ---------------------------------------------------------------------------

def save_idx_to_file(
    in_indicator_list,
    path: str = "indeces.pkl.gz",
    all_scales: List[str] = ["p1", "p2", "p3"],
    all_splits: List[str] = ["s1", "s2", "s3"],
):
    print(f"Save indices to {path}.")
    dataset_split = {}
    for sk in all_scales:
        dataset_split[sk] = {}
        for s in all_splits:
            scale_cfg = info_init(scale_id=sk)
            dataset_split[sk][s] = split_indices(in_indicator_list, scale_cfg=scale_cfg, scheme=s)
    with gzip.open(path, "wb") as f:
        pickle.dump(dataset_split, f, -1)


def load_idx_to_file(path: str = "indeces.pkl.gz"):
    print(f"Load indices from {path}.")
    with gzip.open(path, "rb") as f:
        return pickle.load(f)