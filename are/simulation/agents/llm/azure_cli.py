"""Profile-isolated Azure CLI access token provider."""

from __future__ import annotations

import json
import os
import subprocess
import threading
import time
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Any


class AzureCliTokenError(RuntimeError):
    """Raised when an access token cannot be acquired from Azure CLI."""


class AzureCliTokenProvider:
    """Callable access token provider backed by an isolated Azure CLI profile."""

    def __init__(
        self,
        resource: str,
        azure_config_dir: str | None = None,
        refresh_skew_seconds: int = 300,
    ) -> None:
        self._resource = resource
        self._azure_config_dir = (
            str(Path(azure_config_dir).expanduser().resolve())
            if azure_config_dir
            else None
        )
        self._refresh_skew_seconds = refresh_skew_seconds
        self._token: str | None = None
        self._expires_at = 0.0
        self._lock = threading.Lock()

    def __call__(self) -> str:
        with self._lock:
            now = time.time()
            if self._token and self._expires_at > now + self._refresh_skew_seconds:
                return self._token

            token, expires_at = self._acquire_token()
            self._token = token
            self._expires_at = expires_at
            return token

    def _acquire_token(self) -> tuple[str, float]:
        env = os.environ.copy()
        if self._azure_config_dir:
            if not Path(self._azure_config_dir).is_dir():
                raise AzureCliTokenError(
                    f"Azure CLI profile directory not found: {self._azure_config_dir}"
                )
            env["AZURE_CONFIG_DIR"] = self._azure_config_dir

        command = [
            "az",
            "account",
            "get-access-token",
            "--resource",
            self._resource,
            "--output",
            "json",
        ]
        try:
            result = subprocess.run(
                command,
                env=env,
                check=True,
                capture_output=True,
                text=True,
                timeout=60,
            )
        except FileNotFoundError as exc:
            raise AzureCliTokenError("Azure CLI executable 'az' was not found") from exc
        except subprocess.CalledProcessError as exc:
            detail = (exc.stderr or "").strip()
            raise AzureCliTokenError(
                f"Azure CLI token acquisition failed: {detail or 'unknown error'}"
            ) from exc
        except subprocess.TimeoutExpired as exc:
            raise AzureCliTokenError("Azure CLI token acquisition timed out") from exc

        try:
            payload: dict[str, Any] = json.loads(result.stdout)
            token = payload["accessToken"]
            expires_at = _parse_expiry(payload)
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise AzureCliTokenError("Azure CLI returned an invalid token response") from exc

        if not isinstance(token, str) or not token:
            raise AzureCliTokenError("Azure CLI returned an empty access token")
        return token, expires_at


def _parse_expiry(payload: dict[str, Any]) -> float:
    raw_expiry = payload.get("expires_on") or payload.get("expiresOn")
    if raw_expiry is None:
        raise ValueError("Azure CLI token response has no expiry")

    if isinstance(raw_expiry, (int, float)) or str(raw_expiry).isdigit():
        return float(raw_expiry)

    parsed = datetime.fromisoformat(str(raw_expiry).replace("Z", "+00:00"))
    return parsed.timestamp()


@lru_cache(maxsize=16)
def get_azure_cli_token_provider(
    resource: str,
    azure_config_dir: str | None = None,
    refresh_skew_seconds: int = 300,
) -> AzureCliTokenProvider:
    """Return a shared token provider keyed by resource and CLI profile."""
    return AzureCliTokenProvider(
        resource=resource,
        azure_config_dir=azure_config_dir,
        refresh_skew_seconds=refresh_skew_seconds,
    )