import os
import multiprocessing
import subprocess

def download_shard(shard_id):
    url = f"https://d6108366.hf-mirror.com/datasets/pixparse/cc3m-wds/resolve/main/cc3m-train-{shard_id:04d}.tar"
    dest = f"/data2/mingyu/composed_image_retrieval/data/wds_cache/cc3m-train-{shard_id:04d}.tar"
    if os.path.exists(dest) and os.path.getsize(dest) > 100 * 1024 * 1024: 
        print(f"Skipping {dest}")
        return
    print(f"Downloading {url}...")
    subprocess.run(["wget", "-q", "-O", dest, url])

if __name__ == "__main__":
    pool = multiprocessing.Pool(processes=8) 
    pool.map(download_shard, range(576))