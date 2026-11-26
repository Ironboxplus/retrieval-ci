"""Pre-fill done.rank{N}.txt with existing vectors to skip re-embedding."""
import os, glob, sys

vectors_dir = sys.argv[1]  # e.g. /mnt/lsb/retrival/vectors
chunks_dir = sys.argv[2]   # e.g. /mnt/lsb/retrival/chunks
world = int(sys.argv[3])   # e.g. 7

shards = sorted(glob.glob(os.path.join(chunks_dir, "chunks_*.jsonl.gz")))
print(f"Total shards: {len(shards)}")

for rank in range(world):
    done_path = os.path.join(vectors_dir, f"done.rank{rank}.txt")
    lines = []
    new_count = 0
    for i, s in enumerate(shards):
        if i % world != rank:
            continue
        idx = os.path.basename(s).replace("chunks_", "").replace(".jsonl.gz", "")
        vec_name = f"vectors_{idx}.parquet"
        vec_path = os.path.join(vectors_dir, vec_name)
        if os.path.exists(vec_path):
            lines.append(vec_name)
        else:
            new_count += 1
    with open(done_path, "w") as f:
        f.write("\n".join(lines) + "\n" if lines else "")
    print(f"rank {rank}: {len(lines)} done, {new_count} new")
