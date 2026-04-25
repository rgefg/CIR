import argparse
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser("Merge rank-local teacher embedding shards")
    parser.add_argument("--shard-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    shard_dir = Path(args.shard_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    emb_files = sorted(shard_dir.glob("embeddings.rank*.npy"))
    if not emb_files:
        raise FileNotFoundError(f"No embedding shards found in {shard_dir}")

    arrays = [np.load(path, mmap_mode="r") for path in emb_files]
    dims = {array.shape[1] for array in arrays if array.ndim == 2}
    if len(dims) != 1:
        raise ValueError(f"Shard dimensions differ: {sorted(dims)}")
    dim = dims.pop()
    total = sum(array.shape[0] for array in arrays)
    output = np.lib.format.open_memmap(output_dir / "embeddings.npy", mode="w+", dtype=np.float16, shape=(total, dim))

    all_ids = []
    offset = 0
    for emb_file, array in zip(emb_files, arrays):
        rank = emb_file.stem.split("rank")[-1]
        ids_file = shard_dir / f"ids.rank{rank}.txt"
        ids = [line.strip() for line in ids_file.read_text(encoding="utf-8").splitlines() if line.strip()]
        if len(ids) != array.shape[0]:
            raise ValueError(f"ID count mismatch for rank {rank}: ids={len(ids)} rows={array.shape[0]}")
        output[offset : offset + array.shape[0]] = array
        offset += array.shape[0]
        all_ids.extend(ids)

    (output_dir / "ids.txt").write_text("\n".join(all_ids) + "\n", encoding="utf-8")
    (output_dir / "meta.json").write_text(
        json.dumps(
            {
                "rows": int(total),
                "dim": int(dim),
                "source_shard_dir": str(shard_dir),
                "format": "ids.txt + embeddings.npy",
                "dtype": "float16",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"merged rows={total} dim={dim} -> {output_dir}")


if __name__ == "__main__":
    main()

