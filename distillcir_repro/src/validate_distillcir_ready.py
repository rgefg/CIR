import argparse
import glob
import json
from pathlib import Path

import braceexpand
import numpy as np


def expand(pattern):
    out = []
    for item in braceexpand.braceexpand(pattern):
        if "*" in item or "?" in item:
            out.extend(glob.glob(item))
        else:
            out.append(item)
    return sorted(
        item
        for item in out
        if item.startswith(("http://", "https://", "pipe:")) or Path(item).exists()
    )


def main():
    parser = argparse.ArgumentParser("DistillCIR preflight checker")
    parser.add_argument("--cc3m-cir-jsonl", required=True)
    parser.add_argument("--wds-shards", required=True)
    parser.add_argument("--pic2word-pretrained", required=True)
    parser.add_argument("--teacher-cache", required=True)
    args = parser.parse_args()

    jsonl = Path(args.cc3m_cir_jsonl)
    pic2word = Path(args.pic2word_pretrained)
    teacher = Path(args.teacher_cache)
    shards = expand(args.wds_shards)

    print(f"jsonl: {jsonl} exists={jsonl.exists()}")
    print(f"pic2word: {pic2word} exists={pic2word.exists()}")
    print(f"wds shards: {len(shards)}")
    if shards:
        print(f"first shard: {shards[0]}")
        print(f"last shard:  {shards[-1]}")

    required = [teacher / "meta.json", teacher / "ids.txt", teacher / "embeddings.npy"]
    for path in required:
        print(f"teacher {path.name}: exists={path.exists()}")
    if all(path.exists() for path in required):
        meta = json.loads((teacher / "meta.json").read_text(encoding="utf-8"))
        arr = np.load(teacher / "embeddings.npy", mmap_mode="r")
        num_ids = sum(1 for _ in (teacher / "ids.txt").open("r", encoding="utf-8"))
        print(f"teacher meta: {meta}")
        print(f"teacher embeddings shape={arr.shape} ids={num_ids}")
        if arr.shape[0] != num_ids:
            raise SystemExit("teacher cache row/id mismatch")

    missing = []
    if not jsonl.exists():
        missing.append(str(jsonl))
    if not pic2word.exists():
        missing.append(str(pic2word))
    if not shards:
        missing.append(args.wds_shards)
    missing.extend(str(path) for path in required if not path.exists())
    if missing:
        raise SystemExit("missing required files:\n" + "\n".join(missing))
    print("DistillCIR preflight OK")


if __name__ == "__main__":
    main()
