# Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import click
import pytest
from gaia2_runner import cli as runner_cli
from gaia2_runner.azure_proxy import (
    PROXY_API_KEY,
    AzureOpenAIProxy,
    AzureProxySettings,
    AzureRoute,
    adapt_request_body,
)
from gaia2_runner.config import load_runner_toml_config


# ---------------------------------------------------------------------------
# Request adaptation
# ---------------------------------------------------------------------------


def test_adapt_translates_reasoning_object_and_max_tokens_for_reasoning_models() -> None:
    body = {
        "model": "gpt-5.5-20260424",
        "messages": [],
        "stream": False,
        "stream_options": {"include_usage": True},
        "reasoning": {"effort": "high"},
        "max_tokens": 1000,
        "temperature": 0.2,
    }
    out = adapt_request_body(body, AzureRoute())
    assert out["reasoning_effort"] == "high"
    assert "reasoning" not in out
    assert out["max_completion_tokens"] == 1000
    assert "max_tokens" not in out
    assert "temperature" not in out
    assert "stream_options" not in out
    # The caller's dict is not mutated.
    assert body["reasoning"] == {"effort": "high"}


def test_adapt_route_effort_overrides_request_effort() -> None:
    out = adapt_request_body(
        {"model": "gpt-5.5", "reasoning": {"effort": "low"}, "reasoning_effort": "low"},
        AzureRoute(reasoning_effort="xhigh"),
    )
    assert out["reasoning_effort"] == "xhigh"


def test_adapt_drops_off_effort_and_keeps_non_reasoning_models_untouched() -> None:
    out = adapt_request_body(
        {
            "model": "Llama-3.3-70B-Instruct",
            "reasoning": {"effort": "off"},
            "max_tokens": 50,
            "temperature": 0,
            "stream": True,
            "stream_options": {"include_usage": True},
        },
        AzureRoute(),
    )
    assert "reasoning_effort" not in out
    assert out["max_tokens"] == 50
    assert out["temperature"] == 0
    assert out["stream_options"] == {"include_usage": True}


# ---------------------------------------------------------------------------
# Forwarding against a fake Azure upstream
# ---------------------------------------------------------------------------


class _FakeAzure:
    def __init__(self, status: int = 200) -> None:
        self.status = status
        self.requests: list[dict] = []
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args) -> None:
                pass

            def do_POST(self) -> None:  # noqa: N802
                length = int(self.headers.get("Content-Length") or 0)
                owner.requests.append(
                    {
                        "path": self.path,
                        "auth": self.headers.get("Authorization"),
                        "body": json.loads(self.rfile.read(length)),
                    }
                )
                payload = json.dumps(
                    {"choices": [{"message": {"role": "assistant", "content": "OK"}}]}
                    if owner.status == 200
                    else {"error": {"message": "rate limited"}}
                ).encode()
                self.send_response(owner.status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    @property
    def endpoint(self) -> str:
        return f"http://127.0.0.1:{self.server.server_address[1]}/openai/"

    def close(self) -> None:
        self.server.shutdown()
        self.server.server_close()


@pytest.fixture
def proxy(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        "gaia2_runner.azure_proxy.AzureCliTokenProvider.__call__",
        lambda self: "test-token",
    )
    instance = AzureOpenAIProxy(AzureProxySettings(api_version="2099-01-01"))
    instance.start()
    yield instance
    instance.stop()


def _post(url: str, body: dict) -> tuple[int, dict]:
    request = urllib.request.Request(
        url,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read())


def test_proxy_forwards_to_deployment_with_token_and_api_version(proxy) -> None:
    upstream = _FakeAzure()
    try:
        proxy.add_route(
            "gpt-5.5-20260424",
            AzureRoute(endpoint=upstream.endpoint, reasoning_effort="xhigh"),
        )
        status, payload = _post(
            proxy.base_url + "/chat/completions",
            {"model": "gpt-5.5-20260424", "messages": [], "max_tokens": 10},
        )
    finally:
        upstream.close()

    assert status == 200
    assert payload["choices"][0]["message"]["content"] == "OK"
    [seen] = upstream.requests
    assert seen["path"] == (
        "/openai/deployments/gpt-5.5-20260424/chat/completions?api-version=2099-01-01"
    )
    assert seen["auth"] == "Bearer test-token"
    assert seen["body"]["reasoning_effort"] == "xhigh"
    assert seen["body"]["max_completion_tokens"] == 10


def test_proxy_strips_litellm_openai_prefix_and_relays_errors(proxy) -> None:
    upstream = _FakeAzure(status=429)
    try:
        proxy.add_route("Llama-3.3-70B-Instruct", AzureRoute(endpoint=upstream.endpoint))
        status, payload = _post(
            proxy.base_url + "/chat/completions",
            {"model": "openai/Llama-3.3-70B-Instruct", "messages": []},
        )
    finally:
        upstream.close()

    assert status == 429
    assert payload["error"]["message"] == "rate limited"
    assert upstream.requests[0]["path"].startswith(
        "/openai/deployments/Llama-3.3-70B-Instruct/chat/completions"
    )
    assert upstream.requests[0]["body"]["model"] == "Llama-3.3-70B-Instruct"


def test_proxy_rejects_conflicting_routes() -> None:
    instance = AzureOpenAIProxy(AzureProxySettings())
    instance.add_route("m", AzureRoute(reasoning_effort="high"))
    instance.add_route("m", AzureRoute(reasoning_effort="high"))
    with pytest.raises(ValueError):
        instance.add_route("m", AzureRoute(reasoning_effort="low"))


def test_settings_from_env() -> None:
    settings = AzureProxySettings.from_env(
        {
            "AZURE_CONFIG_DIR": "/fallback",
            "GAIA2_AZURE_CONFIG_DIR": "/profile",
            "GAIA2_AZURE_API_VERSION": "2030-01-01",
            "GAIA2_AZURE_PROXY_PORT": "4150",
        }
    )
    assert settings.config_dir == "/profile"
    assert settings.api_version == "2030-01-01"
    assert settings.port == 4150


# ---------------------------------------------------------------------------
# Runner wiring
# ---------------------------------------------------------------------------


class _RecordingProxy:
    base_url = "http://127.0.0.1:4999/v1"

    def __init__(self) -> None:
        self.routes: dict[str, AzureRoute] = {}

    def add_route(self, model: str, route: AzureRoute) -> None:
        self.routes[model] = route


def _build(monkeypatch: pytest.MonkeyPatch, **overrides):
    recording = _RecordingProxy()
    monkeypatch.setattr(runner_cli, "_azure_proxy", lambda azure: recording)
    kwargs = dict(
        image="localhost/gaia2-oc:latest",
        runtime="docker",
        timeout=600,
        health_timeout=60,
        provider="azure",
        model="gpt-5.5-20260424",
        api_key=None,
        base_url="https://cloudgpt.example/openai/",
        thinking="high",
        judge_model="Llama-3.3-70B-Instruct",
        judge_provider="azure",
        judge_base_url="https://judge.example/openai/",
        judge_api_key=None,
        volumes=(),
        notification_mode="message",
        time_speed=None,
        reasoning_effort="xhigh",
    )
    kwargs.update(overrides)
    return recording, runner_cli._build_execution_config(**kwargs)


def test_azure_provider_routes_agent_and_judge_through_proxy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("GAIA2_AGENT_REASONING_EFFORT", raising=False)
    recording, (config, provider, model, judge_model, judge_provider, judge_url) = (
        _build(monkeypatch)
    )

    assert config.provider == "openai-compat"
    assert config.api_key == PROXY_API_KEY
    assert config.container_env["BASE_URL"] == recording.base_url
    assert config.container_env["GAIA2_JUDGE_PROVIDER"] == "openai-compat"
    assert config.container_env["GAIA2_JUDGE_BASE_URL"] == recording.base_url
    assert config.container_env["GAIA2_JUDGE_API_KEY"] == PROXY_API_KEY
    assert "REASONING_EFFORT" not in config.container_env
    assert recording.routes["gpt-5.5-20260424"] == AzureRoute(
        endpoint="https://cloudgpt.example/openai/", reasoning_effort="xhigh"
    )
    assert recording.routes["Llama-3.3-70B-Instruct"] == AzureRoute(
        endpoint="https://judge.example/openai/"
    )
    # User-facing metadata keeps the Azure provider and endpoint.
    assert (provider, judge_provider, judge_url) == (
        "azure",
        "azure",
        "https://judge.example/openai/",
    )


def test_non_azure_reasoning_effort_and_timeouts_go_to_container_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GAIA2_LLM_FETCH_TIMEOUT_MS", "900000")
    monkeypatch.setenv("AGENT_TIMEOUT_SECONDS", "7200")
    recording, (config, *_rest) = _build(
        monkeypatch,
        provider="openai-compat",
        api_key="k",
        base_url="http://127.0.0.1:1/v1",
        judge_provider="openai-compat",
        judge_base_url="http://127.0.0.1:1/v1",
    )
    assert recording.routes == {}
    assert config.container_env["REASONING_EFFORT"] == "xhigh"
    assert config.container_env["GAIA2_LLM_FETCH_TIMEOUT_MS"] == "900000"
    assert config.container_env["AGENT_TIMEOUT_SECONDS"] == "7200"


# ---------------------------------------------------------------------------
# TOML config
# ---------------------------------------------------------------------------


def _write_config(tmp_path: Path, extra: str) -> Path:
    (tmp_path / "scenario.json").write_text(
        json.dumps({"metadata": {"definition": {"scenario_id": "s"}}, "events": []})
    )
    path = tmp_path / "eval.toml"
    path.write_text(
        """
[target]
scenario = "scenario.json"

[agent]
image = "localhost/gaia2-oc:latest"
runtime = "docker"
provider = "azure"
model = "gpt-5.5-20260424"
reasoning_effort = "xhigh"

[judge]
provider = "azure"
model = "Llama-3.3-70B-Instruct"
"""
        + extra
    )
    return path


def test_load_config_reads_azure_table(tmp_path: Path) -> None:
    (tmp_path / "profile").mkdir()
    config = load_runner_toml_config(
        str(
            _write_config(
                tmp_path,
                """
[azure]
config_dir = "profile"
api_version = "2025-04-01-preview"
port = 4150
upstream_timeout = 900
""",
            )
        )
    )
    assert config.agent.reasoning_effort == "xhigh"
    assert config.azure is not None
    assert config.azure.config_dir == str((tmp_path / "profile").resolve())
    assert config.azure.port == 4150
    assert config.azure.upstream_timeout == 900.0


def test_load_config_without_azure_table(tmp_path: Path) -> None:
    config = load_runner_toml_config(str(_write_config(tmp_path, "")))
    assert config.azure is None


@pytest.mark.parametrize(
    "extra",
    ['\n[azure]\nbogus = 1\n', '\n[azure]\nconfig_dir = "missing"\n'],
)
def test_load_config_rejects_bad_azure_table(tmp_path: Path, extra: str) -> None:
    with pytest.raises(click.UsageError):
        load_runner_toml_config(str(_write_config(tmp_path, extra)))


def test_load_config_rejects_unknown_reasoning_effort(tmp_path: Path) -> None:
    path = _write_config(tmp_path, "")
    path.write_text(path.read_text().replace('"xhigh"', '"maximum"'))
    with pytest.raises(click.UsageError):
        load_runner_toml_config(str(path))
