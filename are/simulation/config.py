# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.


import os
import tempfile
from pathlib import Path

from dotenv import load_dotenv

# Root directory of the Meta Agents Research Environments project
ARE_SIMULATION_ROOT = Path(__file__).parent.parent.absolute()

load_dotenv()

DEMO_FS_DIR: str = "demo_filesystem"
DEFAULT_FS_PATH: Path = Path("/tmp/are_simulation_datasets/fs_states")

FS_PATH: str = os.environ.get("FS_PATH", str(DEFAULT_FS_PATH))
DEMO_FS_PATH: str = os.environ.get(
    "DEMO_FS_PATH",
    "hf://datasets/meta-agents-research-environments/gaia2_filesystem/demo_filesystem",
)

# Root folder for all are.simulation files.
ARE_SIMULATION_SANDBOX_PATH: str = tempfile.mkdtemp(prefix="are_simulation_sandbox_")


PROVIDERS = [
    "openai",
    "copilot",
    "meta",
    "local",
    "llama-api",
    "huggingface",
    "mock",
    "black-forest-labs",
    "cerebras",
    "cohere",
    "fal-ai",
    "featherless-ai",
    "fireworks-ai",
    "groq",
    "hf-inference",
    "hyperbolic",
    "nebius",
    "novita",
    "nscale",
    "openai",
    "replicate",
    "sambanova",
    "together",
]


# Note: Scenario discovery is now handled through entry points
# See the pyproject.toml file for the entry point configuration
