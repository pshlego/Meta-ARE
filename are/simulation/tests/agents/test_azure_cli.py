import json
import subprocess
from pathlib import Path
from unittest.mock import patch

from are.simulation.agents.are_simulation_agent_config import LLMEngineConfig
from are.simulation.agents.llm.azure_cli import (
    AzureCliTokenError,
    AzureCliTokenProvider,
)
from are.simulation.agents.llm.llm_engine_builder import LLMEngineBuilder


def test_token_provider_isolates_profile_and_caches_token(tmp_path: Path) -> None:
    calls = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs["env"]))
        return subprocess.CompletedProcess(
            command,
            0,
            stdout=json.dumps(
                {"accessToken": "test-token", "expires_on": 4_102_444_800}
            ),
            stderr="",
        )

    provider = AzureCliTokenProvider(
        resource="api://test-resource",
        azure_config_dir=str(tmp_path),
    )
    with patch("subprocess.run", fake_run):
        assert provider() == "test-token"
        assert provider() == "test-token"

    assert len(calls) == 1
    assert calls[0][1]["AZURE_CONFIG_DIR"] == str(tmp_path)


def test_token_provider_rejects_missing_profile(tmp_path: Path) -> None:
    provider = AzureCliTokenProvider(
        resource="api://test-resource",
        azure_config_dir=str(tmp_path / "missing"),
    )

    try:
        provider()
    except AzureCliTokenError as exc:
        assert "profile directory not found" in str(exc)
    else:
        raise AssertionError("Expected AzureCliTokenError")


def test_azure_ces_builder_preserves_endpoint_and_profile() -> None:
    config = LLMEngineConfig(
        model_name="claude-sonnet-4.6",
        provider="azure-ces",
        endpoint="https://example.test/proxy",
        azure_config_dir="/tmp/claude-profile",
    )

    with patch(
        "are.simulation.agents.llm.litellm.litellm_engine.LiteLLMEngine"
    ) as engine:
        LLMEngineBuilder().create_engine(config)

    model_config = engine.call_args.kwargs["model_config"]
    assert model_config.provider == "azure-ces"
    assert model_config.endpoint == "https://example.test/proxy"
    assert model_config.azure_config_dir == "/tmp/claude-profile"