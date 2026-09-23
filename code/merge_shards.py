"""Merge sharded pricing datasets into the single file the trainer loads.

pricing_data.py --out lets several processes generate disjoint seed ranges at once (CP-SAT
scales badly inside one solve, so shards with one thread each are far faster than one shard
with eight). This concatenates them back:

    python merge_shards.py --split train
    python merge_shards.py --split val
"""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True)
    ap.add_argument("--pattern", default=None, help="defaults to pricing_<split>_s*.pkl")
    args = ap.parse_args()

    pattern = args.pattern or f"pricing_{args.split}_s*.pkl"
    shards = sorted(DATA_DIR.glob(pattern), key=lambda q: int(q.stem.split("_s")[-1]))
    if not shards:
        raise SystemExit(f"no shards matching {pattern} in {DATA_DIR}")

    out, seen = [], set()
    for sh in shards:
        with open(sh, "rb") as f:
            data = pickle.load(f)
        dup = [d for d in data if d["cfg"]["seed"] in seen]
        if dup:
            raise SystemExit(f"{sh.name}: seeds {[d['cfg']['seed'] for d in dup]} already in another shard")
        seen.update(d["cfg"]["seed"] for d in data)
        out.extend(data)
        n_prob = sum(len(recs) for d in data for s in d["snaps"] for recs in s["exact"])
        print(f"  {sh.name}: {len(data)} instances, {n_prob} pricing problems")

    dest = DATA_DIR / f"pricing_{args.split}.pkl"
    with open(dest, "wb") as f:
        pickle.dump(out, f)
    total = sum(len(recs) for d in out for s in d["snaps"] for recs in s["exact"])
    print(f"-> {dest.name}: {len(out)} instances, {total} pricing problems")


if __name__ == "__main__":
    main()
