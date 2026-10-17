"""EnvHarness (arXiv 2608.19880) on GAIA2 with OpenClaw as the student.

A port of the original envharness pipeline (corpus -> skill bank -> eval,
SWE-bench variant): the designer reshapes the environment with Contract rules
(A/T/O, ``gaia2_cli.env_rules``) and Stage setup (``gaia2_cli.env_setup``)
inside the gaia2-cli container; skills are distilled from the corpus and
rendered into the OpenClaw harness's ``skills/``.
"""

from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
PROMPTS_DIR = PACKAGE_ROOT / "prompts"
GAIA2_CLI_ROOT = PACKAGE_ROOT.parent
