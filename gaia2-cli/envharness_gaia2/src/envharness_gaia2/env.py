"""Placeholder env class for envharness checkpoints.

Rollouts never instantiate it (they run in gaia2-cli containers); envharness
only needs ``env_type()`` to record accepted candidates as Checkpoints.
"""

from __future__ import annotations


class Gaia2ScenarioEnv:
    @classmethod
    def env_type(cls) -> str:
        return "gaia2_openclaw_scenario"

    @classmethod
    def tool_schemas(cls) -> list[dict]:
        from envharness_gaia2.orchestrator import EXEC_TOOL_SCHEMA

        return [EXEC_TOOL_SCHEMA]

    @classmethod
    def env_state_schema(cls) -> str:
        from envharness_gaia2.orchestrator import ENV_STATE_SCHEMA

        return ENV_STATE_SCHEMA
