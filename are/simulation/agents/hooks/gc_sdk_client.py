# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""GC-SDK (GitHub Copilot SDK) client wrapper for agent-type hook handlers."""

import asyncio
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


async def _run_gc_sdk_session(
    cwd: str | Path,
    prompt: str,
    *,
    model: str = "Claude Opus 4.6",
    timeout: float = 600.0,
) -> str:
    """
    Run a GC-SDK agentic session and return the final assistant response.

    The agent has access to built-in tools (file read, search, terminal)
    scoped to the given `cwd` directory.
    """
    from copilot import CopilotClient, PermissionHandler

    client = CopilotClient({"cwd": str(cwd)})
    tool_calls: list[dict[str, Any]] = []

    try:
        await client.start()

        session = await client.create_session(
            {
                "model": model,
                "on_permission_request": PermissionHandler.approve_all,
            }
        )

        # Track tool calls for logging
        from copilot.generated.session_events import SessionEventType

        def _event_handler(event):
            if event.type == SessionEventType.TOOL_EXECUTION_START:
                tool_calls.append(
                    {
                        "tool": getattr(event.data, "tool_name", "unknown"),
                        "arguments": getattr(event.data, "arguments", {}),
                    }
                )
            elif event.type == SessionEventType.TOOL_EXECUTION_COMPLETE:
                result = getattr(event.data, "result", None)
                if tool_calls:
                    preview = str(result)[:500] if result else ""
                    tool_calls[-1]["result_preview"] = preview

        session.on(_event_handler)

        resp = await session.send_and_wait(
            {"prompt": prompt}, timeout=timeout
        )
        raw = resp.data.content if resp else ""

        if tool_calls:
            logger.info(
                f"[Hook-Agent] GC-SDK session completed with {len(tool_calls)} tool call(s)"
            )

        return raw
    finally:
        await client.stop()


def run_gc_sdk_agent(
    cwd: str | Path,
    prompt: str,
    *,
    model: str = "Claude Opus 4.6",
    timeout: float = 600.0,
) -> str:
    """
    Synchronous wrapper for the async GC-SDK session.

    Handles event loop creation/reuse for callers that are not async.
    """
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        # We're inside an existing event loop — run in a new thread
        import concurrent.futures

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(
                asyncio.run,
                _run_gc_sdk_session(
                    cwd, prompt, model=model, timeout=timeout
                ),
            )
            return future.result(timeout=timeout + 30)
    else:
        return asyncio.run(
            _run_gc_sdk_session(cwd, prompt, model=model, timeout=timeout)
        )
