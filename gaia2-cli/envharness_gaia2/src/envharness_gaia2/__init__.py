"""EnvHarness-style skill-bank co-evolution for OpenClaw on GAIA2.

The OpenClaw seed harness (gaia2-cli/harness/openclaw) is the policy's
harness; only its ``skills/`` directory evolves. Environment variations are
EnvHarness "Rules" executed inside the gaia2-cli container by
``gaia2_cli.env_rules`` (block-before-execute and read-output filtering).
"""

from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
PROMPTS_DIR = PACKAGE_ROOT / "prompts"
GAIA2_CLI_ROOT = PACKAGE_ROOT.parent
