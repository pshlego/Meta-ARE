# Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
"""Host-side OpenAI-compatible proxy for Azure OpenAI (CloudGPT) deployments.

Agent containers and the in-container judge only speak OpenAI-compatible
Chat Completions with a static API key. Azure OpenAI deployments behind
CloudGPT instead need an Azure AD bearer token (acquired through the Azure CLI,
optionally from an isolated ``AZURE_CONFIG_DIR`` profile), an ``api-version``
query parameter and a ``/deployments/<model>`` path.

This module runs a small threaded HTTP server on ``127.0.0.1`` that accepts
``POST .../chat/completions`` and forwards it to the configured Azure endpoint,
so tokens never enter the containers and are refreshed for long runs.
Containers reach it through ``--network=host``; ``127.0.0.1`` is already in
their ``no_proxy`` list.

Request adaptation (kept minimal and explicit):

* ``reasoning: {"effort": ...}`` (added by OpenClaw's ``launch.mjs``) becomes
  ``reasoning_effort``; a per-model override from the proxy config wins.
* For reasoning models (``gpt-5*``, ``o<N>*``), ``max_tokens`` becomes
  ``max_completion_tokens`` and sampling knobs Azure rejects are dropped.
* ``stream_options`` is dropped when the request is not streaming.
"""

from __future__ import annotations

import json
import logging
import os
import re
import subprocess
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

CLOUDGPT_ENDPOINT = "https://cloudgpt-openai.azure-api.net/openai/"
CLOUDGPT_RESOURCE = "api://feb7b661-cac7-44a8-8dc1-163b63c23df2"
DEFAULT_API_VERSION = "2025-04-01-preview"
DEFAULT_UPSTREAM_TIMEOUT_SECONDS = 1800.0
PROXY_API_KEY = "azure-ad-proxy"

_REASONING_MODEL_RE = re.compile(r"^(gpt-5|o\d)", re.IGNORECASE)
_REASONING_UNSUPPORTED_KEYS = ("temperature", "top_p", "presence_penalty", "frequency_penalty", "logprobs", "top_logprobs")
_HOP_BY_HOP_HEADERS = frozenset(
    {"connection", "keep-alive", "transfer-encoding", "content-length", "content-encoding"}
)


class AzureTokenError(RuntimeError):
    """Raised when an Azure AD access token cannot be acquired."""


class AzureCliTokenProvider:
    """Thread-safe Azure CLI token cache for one resource and CLI profile."""

    def __init__(
        self,
        resource: str,
        azure_config_dir: str | None = None,
        refresh_skew_seconds: int = 300,
    ) -> None:
        self._resource = resource
        self._azure_config_dir = (
            str(Path(azure_config_dir).expanduser().resolve()) if azure_config_dir else None
        )
        self._refresh_skew_seconds = refresh_skew_seconds
        self._token: str | None = None
        self._expires_at = 0.0
        self._lock = threading.Lock()

    def __call__(self) -> str:
        with self._lock:
            if self._token and self._expires_at > time.time() + self._refresh_skew_seconds:
                return self._token
            self._token, self._expires_at = self._acquire()
            return self._token

    def _acquire(self) -> tuple[str, float]:
        env = os.environ.copy()
        if self._azure_config_dir:
            if not Path(self._azure_config_dir).is_dir():
                raise AzureTokenError(
                    f"Azure CLI profile directory not found: {self._azure_config_dir}"
                )
            env["AZURE_CONFIG_DIR"] = self._azure_config_dir
        command = [
            "az", "account", "get-access-token",
            "--resource", self._resource,
            "--output", "json",
        ]
        try:
            result = subprocess.run(
                command, env=env, check=True, capture_output=True, text=True, timeout=60
            )
        except FileNotFoundError as exc:
            raise AzureTokenError("Azure CLI executable 'az' was not found") from exc
        except subprocess.CalledProcessError as exc:
            detail = (exc.stderr or "").strip()
            raise AzureTokenError(
                f"Azure CLI token acquisition failed: {detail or 'unknown error'}"
            ) from exc
        except subprocess.TimeoutExpired as exc:
            raise AzureTokenError("Azure CLI token acquisition timed out") from exc
        try:
            payload = json.loads(result.stdout)
            token = payload["accessToken"]
            expires_at = _parse_expiry(payload)
        except (KeyError, TypeError, ValueError) as exc:
            raise AzureTokenError("Azure CLI returned an invalid token response") from exc
        if not isinstance(token, str) or not token:
            raise AzureTokenError("Azure CLI returned an empty access token")
        return token, expires_at


def _parse_expiry(payload: dict[str, Any]) -> float:
    raw = payload.get("expires_on") or payload.get("expiresOn")
    if raw is None:
        raise ValueError("Azure CLI token response has no expiry")
    if isinstance(raw, (int, float)) or str(raw).isdigit():
        return float(raw)
    return datetime.fromisoformat(str(raw).replace("Z", "+00:00")).timestamp()


@dataclass(frozen=True, slots=True)
class AzureRoute:
    """Where requests for one model name are forwarded."""

    endpoint: str = CLOUDGPT_ENDPOINT
    reasoning_effort: str | None = None


@dataclass(slots=True)
class AzureProxySettings:
    """Proxy-wide Azure settings (shared by every route)."""

    config_dir: str | None = None
    resource: str = CLOUDGPT_RESOURCE
    api_version: str = DEFAULT_API_VERSION
    port: int = 0
    upstream_timeout_seconds: float = DEFAULT_UPSTREAM_TIMEOUT_SECONDS

    @classmethod
    def from_env(cls, env: dict[str, str] | None = None) -> "AzureProxySettings":
        env = dict(os.environ if env is None else env)
        return cls(
            config_dir=(env.get("GAIA2_AZURE_CONFIG_DIR") or env.get("AZURE_CONFIG_DIR") or None),
            resource=env.get("GAIA2_AZURE_RESOURCE") or CLOUDGPT_RESOURCE,
            api_version=env.get("GAIA2_AZURE_API_VERSION") or DEFAULT_API_VERSION,
            port=int(env.get("GAIA2_AZURE_PROXY_PORT") or 0),
            upstream_timeout_seconds=float(
                env.get("GAIA2_AZURE_UPSTREAM_TIMEOUT") or DEFAULT_UPSTREAM_TIMEOUT_SECONDS
            ),
        )


def is_reasoning_model(model: str) -> bool:
    return bool(_REASONING_MODEL_RE.match(model))


def adapt_request_body(body: dict[str, Any], route: AzureRoute) -> dict[str, Any]:
    """Translate an OpenAI-compatible request body for Azure OpenAI."""
    body = dict(body)
    model = str(body.get("model") or "")

    reasoning = body.pop("reasoning", None)
    effort = route.reasoning_effort
    if effort is None and isinstance(reasoning, dict):
        effort = reasoning.get("effort")
    if effort is None:
        effort = body.get("reasoning_effort")
    if effort and effort not in ("off", "none"):
        body["reasoning_effort"] = effort
    else:
        body.pop("reasoning_effort", None)

    if is_reasoning_model(model):
        if "max_tokens" in body:
            max_tokens = body.pop("max_tokens")
            body.setdefault("max_completion_tokens", max_tokens)
        for key in _REASONING_UNSUPPORTED_KEYS:
            body.pop(key, None)

    if not body.get("stream"):
        body.pop("stream_options", None)
    return body


class AzureOpenAIProxy:
    """Threaded HTTP server forwarding chat completions to Azure deployments."""

    def __init__(self, settings: AzureProxySettings) -> None:
        self.settings = settings
        self._token_provider = AzureCliTokenProvider(
            resource=settings.resource, azure_config_dir=settings.config_dir
        )
        self._routes: dict[str, AzureRoute] = {}
        self._routes_lock = threading.Lock()
        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None
        self.request_count = 0

    # -- configuration -----------------------------------------------------
    def add_route(self, model: str, route: AzureRoute) -> None:
        with self._routes_lock:
            existing = self._routes.get(model)
            if existing is not None and existing != route:
                raise ValueError(
                    f"Conflicting Azure routes for model {model!r}: {existing} vs {route}"
                )
            self._routes[model] = route

    def route_for(self, model: str) -> AzureRoute:
        with self._routes_lock:
            if model in self._routes:
                return self._routes[model]
            if model.startswith("openai/") and model[len("openai/"):] in self._routes:
                return self._routes[model[len("openai/"):]]
        return AzureRoute()

    def routes(self) -> dict[str, AzureRoute]:
        with self._routes_lock:
            return dict(self._routes)

    # -- lifecycle ---------------------------------------------------------
    @property
    def base_url(self) -> str:
        if self._server is None:
            raise RuntimeError("Azure proxy is not running")
        return f"http://127.0.0.1:{self._server.server_address[1]}/v1"

    def check_token(self) -> None:
        """Fail fast when the Azure CLI profile cannot issue a token."""
        self._token_provider()

    def start(self) -> None:
        if self._server is not None:
            return
        self.check_token()
        proxy = self

        class Handler(_ProxyHandler):
            owner = proxy

        self._server = ThreadingHTTPServer(("127.0.0.1", self.settings.port), Handler)
        self._server.daemon_threads = True
        self._thread = threading.Thread(
            target=self._server.serve_forever, name="azure-openai-proxy", daemon=True
        )
        self._thread.start()
        logger.info("Azure OpenAI proxy listening on %s", self.base_url)

    def stop(self) -> None:
        if self._server is None:
            return
        self._server.shutdown()
        self._server.server_close()
        self._server = None
        self._thread = None

    # -- forwarding --------------------------------------------------------
    def forward(self, body: dict[str, Any]) -> tuple[int, dict[str, str], Any]:
        model = str(body.get("model") or "")
        if model.startswith("openai/"):
            model = model[len("openai/"):]
            body = {**body, "model": model}
        route = self.route_for(model)
        payload = adapt_request_body(body, route)
        url = (
            route.endpoint.rstrip("/")
            + "/deployments/"
            + urllib.parse.quote(model, safe="")
            + "/chat/completions?api-version="
            + urllib.parse.quote(self.settings.api_version, safe="")
        )
        request = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            method="POST",
            headers={
                "Authorization": f"Bearer {self._token_provider()}",
                "Content-Type": "application/json",
                "Accept": "text/event-stream" if payload.get("stream") else "application/json",
            },
        )
        self.request_count += 1
        try:
            response = urllib.request.urlopen(
                request, timeout=self.settings.upstream_timeout_seconds
            )
        except urllib.error.HTTPError as exc:
            return exc.code, dict(exc.headers.items()), exc
        return response.status, dict(response.headers.items()), response


class _ProxyHandler(BaseHTTPRequestHandler):
    owner: AzureOpenAIProxy
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt: str, *args: Any) -> None:  # noqa: D401
        logger.debug("azure-proxy: " + fmt, *args)

    def _send_json(self, status: int, payload: Any) -> None:
        data = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:  # noqa: N802
        path = urllib.parse.urlparse(self.path).path.rstrip("/")
        if path in ("/health", "/v1/health"):
            self._send_json(200, {"status": "ok"})
        elif path in ("/models", "/v1/models"):
            self._send_json(
                200,
                {
                    "object": "list",
                    "data": [
                        {"id": name, "object": "model", "owned_by": "azure"}
                        for name in sorted(self.owner.routes())
                    ],
                },
            )
        else:
            self._send_json(404, {"error": {"message": f"Unknown path {path}"}})

    def do_POST(self) -> None:  # noqa: N802
        path = urllib.parse.urlparse(self.path).path.rstrip("/")
        if not path.endswith("/chat/completions"):
            self._send_json(404, {"error": {"message": f"Unsupported path {path}"}})
            return
        try:
            length = int(self.headers.get("Content-Length") or 0)
            body = json.loads(self.rfile.read(length) or b"{}")
            if not isinstance(body, dict) or not body.get("model"):
                raise ValueError("request body must be a JSON object with 'model'")
        except (ValueError, json.JSONDecodeError) as exc:
            self._send_json(400, {"error": {"message": f"Invalid request: {exc}"}})
            return

        started = time.monotonic()
        try:
            status, headers, upstream = self.owner.forward(body)
        except AzureTokenError as exc:
            logger.error("azure-proxy token error: %s", exc)
            self._send_json(401, {"error": {"message": str(exc), "type": "azure_token_error"}})
            return
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            logger.warning("azure-proxy upstream error for %s: %s", body.get("model"), exc)
            self._send_json(502, {"error": {"message": f"Upstream error: {exc}"}})
            return

        try:
            self.send_response(status)
            for key, value in headers.items():
                if key.lower() not in _HOP_BY_HOP_HEADERS:
                    self.send_header(key, value)
            if body.get("stream") and status == 200:
                # Relay server-sent events as they arrive.
                self.send_header("Transfer-Encoding", "chunked")
                self.end_headers()
                while True:
                    chunk = upstream.read1(65536) if hasattr(upstream, "read1") else upstream.read(65536)
                    if not chunk:
                        break
                    self.wfile.write(f"{len(chunk):X}\r\n".encode() + chunk + b"\r\n")
                    self.wfile.flush()
                self.wfile.write(b"0\r\n\r\n")
            else:
                data = upstream.read()
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
        except (BrokenPipeError, ConnectionResetError):
            logger.warning("azure-proxy client disconnected (model=%s)", body.get("model"))
        finally:
            try:
                upstream.close()
            except Exception:
                pass
        logger.info(
            "azure-proxy %s -> HTTP %s in %.1fs",
            body.get("model"), status, time.monotonic() - started,
        )


_SHARED_PROXY: AzureOpenAIProxy | None = None
_SHARED_LOCK = threading.Lock()


def shared_proxy(settings: AzureProxySettings | None = None) -> AzureOpenAIProxy:
    """Return the process-wide proxy, starting it on first use."""
    global _SHARED_PROXY
    with _SHARED_LOCK:
        if _SHARED_PROXY is None:
            _SHARED_PROXY = AzureOpenAIProxy(settings or AzureProxySettings.from_env())
            _SHARED_PROXY.start()
        return _SHARED_PROXY
