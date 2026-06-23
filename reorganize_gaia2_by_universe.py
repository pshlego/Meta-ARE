"""Reorganize gaia2 dataset from config-first to universe-first layout.

Source:  datasets_local/gaia2/{config}/validation/{file}.json
Target:  datasets_local/gaia2_by_universe/universe_{id}/{config}/validation/{file}.json
"""

import os
import re
import shutil

SRC_BASE = "/sdb/v-parksungho/Meta-ARE/datasets_local/gaia2"
DST_BASE = "/sdb/v-parksungho/Meta-ARE/datasets_local/gaia2_by_universe"

configs = ["execution", "search", "adaptability", "time", "ambiguity"]
copied = 0

for cfg in configs:
    src_dir = os.path.join(SRC_BASE, cfg, "validation")
    if not os.path.isdir(src_dir):
        print(f"Skipping {src_dir} (not found)")
        continue

    for fname in sorted(os.listdir(src_dir)):
        m = re.search(r"universe_(\d+)", fname)
        if not m:
            print(f"  WARNING: no universe id in {fname}, skipping")
            continue
        universe_id = m.group(1)

        dst_dir = os.path.join(DST_BASE, f"universe_{universe_id}", cfg, "validation")
        os.makedirs(dst_dir, exist_ok=True)
        shutil.copy2(os.path.join(src_dir, fname), os.path.join(dst_dir, fname))
        copied += 1

print(f"\nDone. Copied {copied} files to {DST_BASE}")

# Print summary
for d in sorted(os.listdir(DST_BASE)):
    full = os.path.join(DST_BASE, d)
    if os.path.isdir(full):
        count = sum(
            len(files)
            for _, _, files in os.walk(full)
        )
        print(f"  {d}: {count} files")
