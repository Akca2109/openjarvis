"""Persistent memory service: async fact extraction integrated into core.

``MemoryService`` runs fact extraction on a dedicated background thread so it
never blocks ``jarvis serve`` request handling or the ``jarvis chat`` REPL.
Callers hand off an exchange via :meth:`submit`, which enqueues the work and
returns immediately — the slow model call and disk write happen out of band.
The worker swallows every per-job error (including ``BrokenPipeError`` when a
client disconnects mid-extraction), so a flaky extraction model can never take
down the host process.

The service is started and stopped as part of the OpenJarvis lifecycle (see
``cli/serve.py`` and ``cli/chat_cmd.py``) and is configured through the
``[memory]`` section of ``config.toml``.
"""

from __future__ import annotations

import logging
import queue
import threading
from typing import Any, Dict, List, Mapping, Optional

from openjarvis.conversations.store import OWNER_USER_ID
from openjarvis.core.events import Event, EventBus, EventType
from openjarvis.memory.credentials import contains_credential
from openjarvis.memory.extractor import FactExtractor
from openjarvis.memory.store import (
    TRUST_AUTO,
    TRUST_UNTRUSTED,
    Fact,
    FactStore,
    create_fact_store,
)

logger = logging.getLogger(__name__)

# Only these severities suppress a whole exchange; see _blocks_exchange().
_BLOCKING_THREAT_LEVELS = frozenset({"high", "critical"})

# Sentinel pushed onto the queue to wake the worker for shutdown.
_STOP = object()

# Surfaces whose published exchanges may become automatic facts, mapped to
# the canonical user those facts belong to. Only the local CLI chat is tied
# to the conversation store's owner; server exchanges come from API clients
# with no authenticated owner mapping, so they must not write the owner's
# memory.
_AUTOMATIC_FACT_ORIGINS: Dict[str, str] = {"cli.chat": OWNER_USER_ID}


class MemoryService:
    """Background long-term-memory extraction and persistence service."""

    def __init__(
        self,
        store: FactStore,
        extractor: FactExtractor,
        *,
        event_bus: EventBus | None = None,
        scanner: Any = None,
        max_queue: int = 256,
    ) -> None:
        self._store = store
        self._extractor = extractor
        self._event_bus = event_bus
        self._scanner = scanner
        self._subscribed = False
        self._queue: "queue.Queue[Any]" = queue.Queue(maxsize=max(1, max_queue))
        self._thread: Optional[threading.Thread] = None
        self._running = threading.Event()

    # -- lifecycle ----------------------------------------------------------

    def start(self) -> None:
        """Start the background worker thread (idempotent)."""
        if self._running.is_set():
            return
        self._running.set()
        self._subscribe_events()
        self._thread = threading.Thread(
            target=self._loop,
            name="memory-service",
            daemon=True,
        )
        self._thread.start()
        logger.debug("Memory service started")

    def stop(self, timeout: float = 2.0) -> None:
        """Signal the worker to drain and stop, then join it (idempotent)."""
        if not self._running.is_set():
            return
        self._running.clear()
        try:
            self._queue.put_nowait(_STOP)
        except queue.Full:
            pass  # worker will notice the cleared flag on its next loop
        thread = self._thread
        if thread is not None:
            thread.join(timeout=timeout)
        self._thread = None
        self._unsubscribe_events()
        logger.debug("Memory service stopped")

    @property
    def is_running(self) -> bool:
        return self._running.is_set()

    # -- submission ---------------------------------------------------------

    def submit(
        self,
        user_text: str,
        assistant_text: str = "",
        *,
        provenance: Optional[Mapping[str, str]] = None,
    ) -> bool:
        """Queue an exchange for extraction. Non-blocking; never raises.

        Facts are extracted from *user_text* only; *assistant_text* is used
        solely for the injection scan. *provenance* (see
        ``memory.store.PROVENANCE_FIELDS``) is recorded on every stored fact.

        Returns True if the job was enqueued, False if the service is not
        running or the queue is full (in which case the exchange is dropped
        rather than blocking the caller — extraction is best-effort).
        """
        if not self._running.is_set():
            return False
        if not user_text or not user_text.strip():
            return False
        try:
            self._queue.put_nowait((user_text, assistant_text, dict(provenance or {})))
            return True
        except queue.Full:
            logger.debug("Memory service queue full; dropping exchange")
            return False

    def _subscribe_events(self) -> None:
        """Subscribe to lifecycle events that feed automatic memory."""
        if self._event_bus is None or self._subscribed:
            return
        self._event_bus.subscribe(
            EventType.CHAT_EXCHANGE_COMPLETED,
            self._on_completed_exchange,
        )
        self._subscribed = True

    def _unsubscribe_events(self) -> None:
        """Unsubscribe from lifecycle events (idempotent)."""
        if self._event_bus is None or not self._subscribed:
            return
        self._event_bus.unsubscribe(
            EventType.CHAT_EXCHANGE_COMPLETED,
            self._on_completed_exchange,
        )
        self._subscribed = False

    def _on_completed_exchange(self, event: Event) -> None:
        """Queue a completed chat exchange published on the event bus.

        Only exchanges from an owner-bound surface are accepted; anything
        else (e.g. ``server.chat``) is ignored.
        """
        data = event.data or {}
        origin = str(data.get("source", "") or "")
        owner = _AUTOMATIC_FACT_ORIGINS.get(origin)
        if owner is None:
            return
        self.submit(
            str(data.get("user_text", "") or ""),
            str(data.get("assistant_text", "") or ""),
            provenance={
                "origin": origin,
                "conversation_id": str(data.get("conversation_id", "") or ""),
                "user_message_id": str(data.get("user_message_id", "") or ""),
                "owner": owner,
            },
        )

    # -- worker -------------------------------------------------------------

    def _loop(self) -> None:
        while True:
            try:
                job = self._queue.get(timeout=0.5)
            except queue.Empty:
                if not self._running.is_set():
                    break
                continue
            if job is _STOP:
                self._queue.task_done()
                break
            try:
                self._process(job)
            except Exception:  # noqa: BLE001 — a bad job must not kill the worker
                logger.debug("Memory extraction job failed", exc_info=True)
            finally:
                self._queue.task_done()
            if not self._running.is_set() and self._queue.empty():
                break

    def _scan(self, text: str) -> Optional[Any]:
        """Run the injection scanner, or return ``None`` when it is absent or
        errors. Scanning fails *open*: an outage must not silently switch off
        memory capture."""
        if self._scanner is None:
            return None
        try:
            return self._scanner.scan(text)
        except BaseException as exc:  # noqa: BLE001 — scanning is best-effort
            # PyO3 deliberately exposes a Rust panic as ``PanicException``, a
            # direct BaseException subclass. Keep the memory worker alive if a
            # native scanner ever panics, while preserving Python's control-
            # flow exceptions instead of swallowing shutdown/interrupts.
            rust_panic = (
                exc.__class__.__module__ == "pyo3_runtime"
                and exc.__class__.__name__ == "PanicException"
            )
            if not isinstance(exc, Exception) and not rust_panic:
                raise
            logger.debug("Injection scan failed; proceeding (fail-open)", exc_info=True)
            return None

    @staticmethod
    def _flagged(result: Any) -> bool:
        """True if *result* carries any finding at all."""
        return result is not None and not getattr(result, "is_clean", True)

    @classmethod
    def _blocks_exchange(cls, result: Any) -> bool:
        """True only for a *severe* finding.

        Dropping an exchange is destructive and unrecoverable, so it is
        reserved for HIGH/CRITICAL hits. Lower-severity patterns (a pasted
        shell one-liner, a role-delimiter fence quoted in a code discussion)
        are ordinary developer traffic; those exchanges are still extracted,
        and anything suspicious is caught per-fact by the quarantine tier.
        """
        if not cls._flagged(result):
            return False
        level = getattr(result, "threat_level", "")
        name = str(getattr(level, "value", level) or "").strip().lower()
        return name in _BLOCKING_THREAT_LEVELS

    def _process(self, job: Any) -> None:
        user_text, assistant_text, *rest = job
        provenance: Dict[str, str] = dict(rest[0]) if rest else {}
        # Scan BEFORE extraction so an overt injection attempt never reaches the
        # extraction model or the store at all. The whole exchange is scanned
        # even though only the user's side is extracted from.
        if self._blocks_exchange(self._scan(f"{user_text}\n{assistant_text}")):
            # info, not debug: a silently-dropped exchange must be distinguishable
            # from "nothing to extract" in the logs.
            logger.info("Memory extraction skipped: injection detected in exchange")
            return
        # Assistant output is never a fact source (see FactExtractor).
        facts = self._extractor.extract(user_text)
        if not facts:
            return
        # Plaintext secrets are dropped outright, never quarantined: an
        # untrusted row would still keep the credential on disk. Log the
        # count only — never the fact text.
        kept = [fact for fact in facts if not contains_credential(fact)]
        if len(kept) < len(facts):
            logger.info(
                "Memory: dropped %d extracted fact(s) containing credentials",
                len(facts) - len(kept),
            )
        facts = kept
        if not facts:
            return
        provenance = {**provenance, "derived_from": "user"}
        # Provenance is per fact, not per exchange: a fact whose own text trips
        # the scanner is quarantined (stored for audit, filtered out of every
        # model-facing path), while clean facts stay recallable — otherwise
        # tagging everything "untrusted" would make automatic memory write-only.
        clean: List[str] = []
        quarantined: List[str] = []
        for fact in facts:
            target = quarantined if self._flagged(self._scan(fact)) else clean
            target.append(fact)
        stored = self._store.add_many_with_trust(
            clean, source="auto", trust=TRUST_AUTO, provenance=provenance
        )
        if quarantined:
            stored += self._store.add_many_with_trust(
                quarantined,
                source="auto",
                trust=TRUST_UNTRUSTED,
                provenance=provenance,
            )
            logger.info(
                "Memory: quarantined %d extracted fact(s) as untrusted",
                len(quarantined),
            )
        if stored:
            logger.debug("Memory service stored %d new fact(s)", stored)

    # -- store passthroughs -------------------------------------------------

    def list_facts(self) -> List[Fact]:
        return self._store.list()

    def clear_facts(self) -> int:
        return self._store.clear()

    def fact_count(self) -> int:
        return self._store.count()


def build_memory_service(
    config: Any,
    engine: Any,
    default_model: str = "",
    *,
    event_bus: EventBus | None = None,
) -> Optional[MemoryService]:
    """Build a :class:`MemoryService` from config, or ``None`` if disabled.

    Reads the ``[memory]`` section (``config.memory`` / ``config.tools.storage``)
    for ``enabled``, ``backend``, ``extraction_model``, ``max_facts`` and
    ``facts_path``.  Returns ``None`` when memory is disabled or no engine /
    extraction model is available, so callers can simply do::

        svc = build_memory_service(config, engine, model)
        if svc is not None:
            svc.start()
    """
    mem = getattr(config, "memory", None)
    if mem is None or not getattr(mem, "enabled", False):
        return None
    if engine is None:
        return None

    model = getattr(mem, "extraction_model", "") or default_model
    if not model:
        logger.debug("Memory service disabled: no extraction model available")
        return None

    store = create_fact_store(
        getattr(mem, "backend", "local"),
        path=getattr(mem, "facts_path", None),
        max_facts=getattr(mem, "max_facts", 1000),
    )
    extractor = FactExtractor(engine, model)
    return MemoryService(
        store, extractor, event_bus=event_bus, scanner=_build_scanner()
    )


def _build_scanner() -> Any:
    """Construct an injection scanner, or ``None`` if unavailable. Never raises —
    memory must work even if the security module can't load."""
    try:
        from openjarvis.security.injection_scanner import InjectionScanner

        return InjectionScanner()
    except Exception:  # noqa: BLE001 — scanner is an optional defence layer
        logger.debug(
            "Injection scanner unavailable; memory capture unguarded", exc_info=True
        )
        return None


def publish_completed_exchange(
    bus: EventBus | None,
    user_text: str,
    assistant_text: str = "",
    *,
    source: str = "",
    conversation_id: str | None = None,
    user_message_id: str | None = None,
) -> bool:
    """Publish a completed chat exchange for lifecycle subscribers.

    *conversation_id* / *user_message_id* identify the durable transcript
    rows the exchange came from, when the surface records one.
    """
    if bus is None or not user_text or not user_text.strip():
        return False
    bus.publish(
        EventType.CHAT_EXCHANGE_COMPLETED,
        {
            "user_text": user_text,
            "assistant_text": assistant_text or "",
            "source": source,
            "conversation_id": conversation_id or "",
            "user_message_id": user_message_id or "",
        },
    )
    return True


__all__ = ["MemoryService", "build_memory_service", "publish_completed_exchange"]
