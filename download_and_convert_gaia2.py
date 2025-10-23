from datasets import load_dataset
import os, json

BASE = "/Data2/juezhang/meta_are/datasets_local/gaia2"
# configs = ["execution","search","adaptability","time","ambiguity"]
configs = ["search"]

for cfg in configs:
    ds = load_dataset("meta-agents-research-environments/gaia2",
                      name=cfg, split="validation")
    out_dir = os.path.join(BASE, cfg, "validation")
    os.makedirs(out_dir, exist_ok=True)
    for i, row in enumerate(ds):
        # Each row has: id, scenario_id, split, data (json)
        path = os.path.join(out_dir, f"{i:04d}_{row['scenario_id']}.json")
        with open(path, "w", encoding="utf-8") as f:
            f.write(row["data"])
