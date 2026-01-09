def info_init(p_id = "p1"):
    
    
    
    # protocol 1
    if p_id == "p3":
        ratio = 0.25
    elif p_id == "p2":
        ratio = 0.5
    else:
        ratio = 1.0
    all_sub = list(range(1, 21))
    cross_sub_test = [19, 5, 8, 10]
    cross_sub_val = [17] # 1
    cross_sub_train = [i for i in all_sub if (i not in cross_sub_test and i not in cross_sub_val)]
    
    all_act = list(range(1, 51))
    cross_act_test = [ 2, 11, 23, 28, 29
                    ,33,
                    38, 43, 47, 50]
    cross_act_val = [24, 10, 32] # 22
    cross_act_train = [i for i in all_act if (i not in cross_act_test and i not in cross_act_val)]


    cross_sub_split = {"train": {"sub": cross_sub_train, "act":all_act, "ratio": ratio},
                    "val": {"sub": cross_sub_val, "act":all_act, "ratio": ratio},
                    "test": {"sub": cross_sub_test, "act":all_act, "ratio": ratio},
                    "all": [all_sub, all_act]
                    }

    cross_act_split = {"train": {"sub": all_sub, "act": cross_act_train, "ratio": ratio},
                    "val": {"sub":  all_sub, "act":cross_act_val, "ratio": ratio},
                    "test": {"sub":  all_sub, "act":cross_act_test, "ratio": ratio},
                    "all": [all_sub, all_act]
                    }

    random_split = {"train": {"sub": all_sub, "act": all_act, "ratio": 0.75 * ratio},
                    "val": {"sub":  all_sub, "act":all_act, "ratio": 0.05 * ratio},
                    "test": {"sub":  all_sub, "act":all_act, "ratio": 0.2 * ratio},
                    "all": [all_sub, all_act]
                    }
    protocol = {"s1": random_split, "s2": cross_sub_split, "s3": cross_act_split}

    
    return protocol



    
    
#     return protocol
import random
from typing import List, Tuple, Union, Dict, Any
from collections import Counter, defaultdict

def split_ok_pool(pool, out_train, out_val, out_test):
    to_set = lambda L: {tuple(x) for x in L}
    return to_set(pool) == (to_set(out_train) | to_set(out_val) | to_set(out_test))

# index_list item: [sub_id, act_id, frame_id]
def split_indices(
    index_list: List[List[int]],
    protocol: Dict[str, Dict[str, Dict[str, Any]]],
    scheme: str = "s1",
    seed: int = 42,
):
    rng = random.Random(seed)
    cfg = protocol[scheme]
    
    # ----- 0) Global filter by cfg["all"] -----
    all_sub, all_act = cfg["all"]  # [all_sub_list, all_act_list]
    all_sub, all_act = set(all_sub), set(all_act)
    filtered = [[s,a,f] for (s,a,f) in index_list if s in all_sub and a in all_act]

    # val & test share the same sub/act (per your setup)
    subs_train = set(cfg["train"]["sub"])
    acts_train = set(cfg["train"]["act"])
    r_train = float(cfg["train"]["ratio"])
    subs_val = set(cfg["val"]["sub"])
    acts_val = set(cfg["val"]["act"])
    r_val  = float(cfg["val"]["ratio"])
    subs_test = set(cfg["test"]["sub"])
    acts_test = set(cfg["test"]["act"])
    r_test = float(cfg["test"]["ratio"])
    
    

    # ----- 1) Group filtered frames by (sub, act) -----
    frames_by_pair = defaultdict(list)  # (s,a) -> sorted [frame_id,...]
    for s, a, f in filtered:
        frames_by_pair[(s, a)].append(f)
    for k in frames_by_pair:
        frames_by_pair[k].sort()

    def pick_random_consecutive_block(sorted_frames: List[int], k: int, start = None):
        """Pick a random consecutive block of length k from sorted_frames.
        Return (picked_list, remaining_list). Assumes k <= len(sorted_frames)."""
        n = len(sorted_frames)
        if start == None:
            start = rng.randint(0, n - k)  # inclusive
        picked = sorted_frames[start:start + k]
        remaining = sorted_frames[:start] + sorted_frames[start + k:]
        return picked, remaining, start

    out_train, out_val, out_test = [], [], []
    start_train, start_val, start_test = [], [], []


    # ----- 2) Allocate per (sub, act): TEST -> VAL -> TRAIN -----
    for (s, a), frames in frames_by_pair.items():
        remain = frames
        n = len(frames)
        ratio = 0
        if (s in subs_test) and (a in acts_test):
                            
            # TEST (random consecutive)
            n_test  = max(1, int(n * r_test))
            ratio += r_test
            assert n_test <= n
            
            test_block, remain, start = pick_random_consecutive_block(remain, n_test, start=None)
            out_test.extend([[s, a, f] for f in test_block])
            start_test.append([s, a, start])
            
        if (s in subs_train) and (a in acts_train):
            
            # Train (random consecutive)
            n_train = max(1, int(n * r_train))
            ratio += r_train
            assert n_train <= n
            
            train_block, remain, start = pick_random_consecutive_block(remain, n_train, start=None)
            out_train.extend([[s, a, f] for f in train_block])
            start_train.append([s, a, start])

        if (s in subs_val) and (a in acts_val):
            
            # Train (random consecutive)
            n_val = max(1, int(n * r_val))
            ratio += r_val
            assert n_val <= n
            
            val_block, remain, start = pick_random_consecutive_block(remain, n_val, start=None)
            out_val.extend([[s, a, f] for f in val_block])
            start_val.append([s, a, start])
        
        if ratio == 1:
            out_val.extend([[s, a, f] for f in remain])
        
    if protocol == "p1":
        assert split_ok_pool(filtered, out_train, out_val, out_test), "Split union != filtered pool"
            
                        

    # ----- 3) Quick sanity print + union check against filtered pool -----

    return {"train": out_train, "val": out_val, "test": out_test}


import pickle, gzip
def save_idx_to_file(in_indicator_list, path="indeces.pkl.gz", all_protocols = ["p1", "p2", "p3"], all_splits = ["s1", "s2", "s3"]):
    print(f"Save indices to {path} indices splits.")
    dataset_split = {}
    for pk in all_protocols:
        dataset_split[pk] = {}
        for s in all_splits:
            protocol = info_init(p_id=pk)
            split_lists = split_indices(in_indicator_list, protocol= protocol, scheme=s)
            dataset_split[pk][s] = split_lists
            
    with gzip.open(path, "wb") as f:
        pickle.dump(dataset_split, f, protocol=pickle.HIGHEST_PROTOCOL)
        
def load_idx_to_file(path="indeces.pkl.gz"):
    print(f"Load indices from pre-saved {path} indices splits.")
    with gzip.open(path, "rb") as f:
        return pickle.load(f)

            