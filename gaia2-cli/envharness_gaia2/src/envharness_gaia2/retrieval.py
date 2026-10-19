"""Per-scenario skill retrieval: the original evals' ``Bank.retrieve`` top-k.

Every original eval retrieves top-k bank items by the task text and injects
only those (SWE-bench top-5 cosine on the problem statement, ALFWorld and
SpreadsheetBench top-5 MMR, OfficeQA top-3, WebArena top-5 via reproduce.py's
TOPK; its dispatcher.sh defaults to 3). Here the query is
the scenario's first user request (the message the student receives first; no
oracle content), truncated like the OfficeQA/SpreadsheetBench ``task_query``,
and the chosen items are rendered into that scenario's ``skills/``.

Query vectors are cached in the run directory so a resumed run retrieves the
same items.
"""

from __future__ import annotations

import hashlib
import json
import threading
from collections.abc import Callable, Sequence
from pathlib import Path

from envharness.reasoning_bank.bank import Bank, MemoryItem

from envharness_gaia2.config import RetrievalConfig
from envharness_gaia2.gaia2_run import CaseRef

QUERY_CHARS = 2000


def task_query(scenario: Path) -> str:
    """The scenario's first user message (USER ``send_message_to_agent`` with no dependencies)."""
    events = json.loads(Path(scenario).read_text(encoding="utf-8")).get("events") or []
    for event in events:
        action = event.get("action") or {}
        if (event.get("event_type") == "USER" and not event.get("dependencies")
                and action.get("function") == "send_message_to_agent"):
            for arg in action.get("args") or []:
                if arg.get("name") == "content":
                    return str(arg.get("value") or "").strip()[:QUERY_CHARS]
    raise ValueError(f"{scenario}: no initial user message")


def selection_key(items: Sequence[MemoryItem]) -> str:
    """Directory name for a rendered selection; equal selections share one harness."""
    blob = json.dumps([[item.title, item.description, item.content] for item in items])
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


class Retriever:
    def __init__(self, cfg: RetrievalConfig, embed: Callable[[list[str]], list[list[float]]], cache_path: Path) -> None:
        self.cfg = cfg
        self.embed = embed
        self.cache_path = cache_path
        self._lock = threading.Lock()
        self._cache: dict[str, dict] = (json.loads(cache_path.read_text(encoding="utf-8"))
                                        if cache_path.is_file() else {})

    def query_embedding(self, case: CaseRef) -> list[float]:
        query = task_query(case.source)
        digest = "sha256:" + hashlib.sha256(query.encode()).hexdigest()
        with self._lock:
            hit = self._cache.get(case.scenario_id)
            if hit and hit["query_sha256"] == digest:
                return hit["embedding"]
            vector = self.embed([query])[0]
            self._cache[case.scenario_id] = {"query_sha256": digest, "embedding": vector}
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            self.cache_path.write_text(json.dumps(self._cache) + "\n", encoding="utf-8")
            return vector

    def select(self, bank: Sequence[MemoryItem], case: CaseRef) -> list[MemoryItem]:
        if not bank:
            return []
        return Bank(list(bank)).retrieve("", k=self.cfg.top_k, query_embedding=self.query_embedding(case),
                                         mode=self.cfg.mode, mmr_lambda=self.cfg.mmr_lambda)
