Env setup:
1. install `uv` env according to `https://facebookresearch.github.io/meta-agents-research-environments/user_guide/installation.html`
2. download demo filesystem from `https://huggingface.co/datasets/meta-agents-research-environments/gaia2_filesystem/resolve/main/demo_filesystem.zip` and unzip to a local folder, e.g., `/Data2/juezhang/meta_are/local_demo_filesystem/` via `download_demo_filesystem.sh` script.
3. download gaia2 datasets from `https://huggingface.co/datasets/meta-agents-research-environments/gaia2_datasets/resolve/main/gaia2_datasets.zip` and unzip to a local folder, e.g., `/Data2/juezhang/meta_are/datasets_local/` via `download_gaia2_datasets.py` script.
4. setup `.env` file in the root folder of `meta_are` repo with the following content:
```
DEMO_FS_PATH="/Data2/juezhang/meta_are/local_demo_filesystem"
```