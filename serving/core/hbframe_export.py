"""HBFrame-facing serving frontend without ASTRA-Sim execution.

This module keeps LLMServingSim's request, routing, scheduling, KV-block, and
prefix-cache behavior available as a callable frontend. It intentionally does
not generate Chakra graphs or launch ASTRA-Sim; HBFrame supplies the execution
time for each scheduled batch and feeds completion back into the scheduler.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import json

from .memory_model import Device
from .router import Router
from .scheduler import Scheduler


@dataclass(frozen=True)
class HBFrameFrontendConfig:
    """Minimal single-instance configuration for HBFrame serving export."""

    model: str
    max_num_seqs: int = 128
    max_num_batched_tokens: int = 2048
    long_prefill_token_threshold: int = 0
    block_size: int = 16
    dtype_bits: int = 16
    kv_cache_dtype: str = "auto"
    npu_mem_gb: float = 80.0
    cpu_mem_gb: float = 512.0
    cxl_mem_gb: float = 0.0
    num_npus: int = 1
    tp_size: int = 1
    pp_size: int = 1
    ep_size: int = 1
    instance_id: int = 0
    node_id: int = 0
    start_npu: int = 0
    pd_type: str | None = None
    prioritize_prefill: bool = False
    enable_prefix_caching: bool = False
    enable_prefix_sharing: bool = False
    enable_chunked_prefill: bool = True
    prefix_storage: str | None = None
    request_routing_policy: str = "LOAD"
    is_init: bool = True
    req_num: int = 0


@dataclass(frozen=True)
class HBFrameBatchSnapshot:
    """Serializable view of a scheduled LLMServingSim Batch."""

    batch_id: int
    model: str
    instance_id: int
    node_id: int
    start_npu: int
    batch_time_ns: int
    total_len: int
    kv_len: int
    num_prefill: int
    num_decode: int
    prefill_q_list: tuple[int, ...]
    prefill_k_list: tuple[int, ...]
    decode_k_list: tuple[int, ...]
    q_list: tuple[int, ...]
    k_list: tuple[int, ...]
    kv_size_bytes: int
    evict_bytes: int
    load_bytes: int
    scheduled_tokens: dict[int, int]
    requests: tuple[dict[str, object], ...]
    attrs: dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {
            "batch_id": self.batch_id,
            "model": self.model,
            "instance_id": self.instance_id,
            "node_id": self.node_id,
            "start_npu": self.start_npu,
            "batch_time_ns": self.batch_time_ns,
            "total_len": self.total_len,
            "kv_len": self.kv_len,
            "num_prefill": self.num_prefill,
            "num_decode": self.num_decode,
            "prefill_q_list": list(self.prefill_q_list),
            "prefill_k_list": list(self.prefill_k_list),
            "decode_k_list": list(self.decode_k_list),
            "q_list": list(self.q_list),
            "k_list": list(self.k_list),
            "kv_size_bytes": self.kv_size_bytes,
            "evict_bytes": self.evict_bytes,
            "load_bytes": self.load_bytes,
            "scheduled_tokens": {str(key): value for key, value in self.scheduled_tokens.items()},
            "requests": list(self.requests),
            "attrs": dict(self.attrs),
        }


def _prefix_storage_device(name: str | None):
    if name is None or str(name).lower() in {"none", ""}:
        return None
    normalized = str(name).upper()
    if normalized == "CPU":
        return Device.CPU
    if normalized == "CXL":
        return Device.CXL
    raise ValueError(f"Unsupported prefix_storage: {name}")


def _request_snapshot(req, scheduled_tokens: dict[int, int]) -> dict[str, object]:
    return {
        "id": int(req.id),
        "model": req.model,
        "input_tokens": int(req.original_input),
        "output_tokens": int(req.output),
        "arrival_time_ns": int(req.arrival),
        "instance_id": int(req.instance_id),
        "is_init": bool(req.is_init),
        "is_prefill": bool(req.is_prefill()),
        "num_computed_tokens": int(req.num_computed_tokens),
        "scheduled_tokens": int(scheduled_tokens.get(req.id, 0)),
        "chunk_len": int(getattr(req, "chunk_len", 0)),
        "prefix_cache_hit": int(getattr(req, "prefix_cache_hit", 0)),
        "npu_cache_hit": int(getattr(req, "npu_cache_hit", 0)),
        "storage_cache_hit": int(getattr(req, "storage_cache_hit", 0)),
        "session_id": getattr(req, "session_id", None),
        "sub_request_index": getattr(req, "sub_request_index", None),
    }


def snapshot_batch(batch, scheduler: Scheduler) -> HBFrameBatchSnapshot:
    scheduled_tokens = {
        int(key): int(value)
        for key, value in (getattr(batch, "scheduled_tokens", None) or {}).items()
    }
    return HBFrameBatchSnapshot(
        batch_id=int(batch.batch_id),
        model=batch.model,
        instance_id=int(scheduler.instance_id),
        node_id=int(scheduler.node_id),
        start_npu=int(scheduler.start_npu),
        batch_time_ns=int(batch.batch_time),
        total_len=int(batch.total_len),
        kv_len=int(batch.kv_len),
        num_prefill=int(batch.num_prefill),
        num_decode=int(batch.num_decode),
        prefill_q_list=tuple(int(value) for value in batch.prefill_q_list),
        prefill_k_list=tuple(int(value) for value in batch.prefill_k_list),
        decode_k_list=tuple(int(value) for value in batch.decode_k_list),
        q_list=tuple(int(value) for value in batch.q_list),
        k_list=tuple(int(value) for value in batch.k_list),
        kv_size_bytes=int(batch.kv_size),
        evict_bytes=int(batch.evict),
        load_bytes=int(batch.load),
        scheduled_tokens=scheduled_tokens,
        requests=tuple(_request_snapshot(req, scheduled_tokens) for req in batch.requests),
        attrs={
            "fired": tuple(int(value) for value in batch.fired),
            "end": tuple(int(value) for value in batch.end),
        },
    )


class HBFrameServingFrontend:
    """Callable LLMServingSim scheduler wrapper for HBFrame."""

    def __init__(self, config: HBFrameFrontendConfig):
        self.config = config
        prefix_storage = _prefix_storage_device(config.prefix_storage)
        self.scheduler = Scheduler(
            config.model,
            config.node_id,
            config.instance_id,
            config.max_num_seqs,
            config.max_num_batched_tokens,
            config.num_npus,
            config.tp_size,
            config.pp_size,
            config.npu_mem_gb,
            config.cpu_mem_gb,
            config.start_npu,
            config.pd_type,
            config.dtype_bits,
            config.block_size,
            config.req_num,
            config.prioritize_prefill,
            config.enable_prefix_caching,
            config.enable_prefix_sharing,
            None,
            prefix_storage,
            config.enable_chunked_prefill,
            config.long_prefill_token_threshold,
            config.cxl_mem_gb,
            ep_size=config.ep_size,
            kv_cache_dtype=config.kv_cache_dtype,
        )
        self.router = Router(
            1,
            [self.scheduler],
            config.req_num,
            routing_policy=config.request_routing_policy,
        )
        self.current_time_ns = 0
        self._active_batches: dict[int, object] = {}

    def load_requests(self, path: str | Path) -> None:
        """Load a JSONL workload path without depending on ``astra-sim`` cwd."""
        path = Path(path)
        if not path.is_absolute():
            repo_root = Path(__file__).resolve().parents[2]
            cwd_candidate = Path.cwd() / path
            repo_candidate = repo_root / path
            path = cwd_candidate if cwd_candidate.exists() else repo_candidate

        self.router._enable_prefix_caching = self.config.enable_prefix_caching
        self.router._is_init = self.config.is_init
        loaded_lines = 0
        with open(path, "r", encoding="utf-8") as handle:
            for line in handle:
                if self.config.req_num > 0 and loaded_lines >= self.config.req_num:
                    break
                row = json.loads(line)
                if "sub_requests" in row:
                    self.router._load_agentic_session(row, self.config.enable_prefix_caching)
                else:
                    self.router._load_flat_request(row, self.config.enable_prefix_caching)
                loaded_lines += 1
        self.router._pending_requests.sort(key=lambda request: request["arrival_time_ns"])

    def route_arrivals(self, current_time_ns: int) -> int:
        self.current_time_ns = int(current_time_ns)
        return int(self.router.route_arrived_requests(self.current_time_ns))

    def schedule_next(self, current_time_ns: int | None = None) -> HBFrameBatchSnapshot | None:
        if current_time_ns is not None:
            self.route_arrivals(int(current_time_ns))
        batch = self.scheduler.schedule(self.current_time_ns, self.scheduler.start_npu)
        if batch is None:
            return None
        self._active_batches[int(batch.batch_id)] = batch
        return snapshot_batch(batch, self.scheduler)

    def complete_batch(self, batch_id: int, completion_time_ns: int) -> tuple[dict[str, object], ...]:
        """Mark a scheduled batch complete and return newly finished requests."""
        batch_id = int(batch_id)
        completion_time_ns = int(completion_time_ns)
        if batch_id not in self._active_batches:
            raise KeyError(f"Unknown active batch_id={batch_id}")
        prompt_t, gen_t, finished = self.scheduler.add_done(
            batch_id + 1,
            self.scheduler.start_npu,
            completion_time_ns,
        )
        self._active_batches.pop(batch_id, None)
        self.current_time_ns = max(self.current_time_ns, completion_time_ns)
        for req in finished:
            self.router.notify_request_completed(req.id, completion_time_ns)
        return tuple(
            {
                "id": int(req.id),
                "arrival_time_ns": int(req.arrival),
                "end_time_ns": int(req.end_time),
                "latency_ns": int(req.latency),
                "ttft_ns": int(req.ttft),
                "tpot_ns": int(req.tpot),
                "itl_ns": [int(value) for value in req.itl],
                "prompt_tokens_accounted": int(prompt_t),
                "generation_tokens_accounted": int(gen_t),
            }
            for req in finished
        )

    def next_arrival_time_ns(self) -> int | None:
        value = self.router.get_next_pending_arrival()
        return None if value is None else int(value)

    def has_pending_work(self) -> bool:
        return (
            self.router.has_pending_requests()
            or bool(self.scheduler.request)
            or bool(self.scheduler.inflight)
            or bool(self._active_batches)
        )

    def request_metrics(self) -> tuple[dict[str, object], ...]:
        return tuple(
            {
                "id": int(req.id),
                "arrival_time_ns": int(req.arrival),
                "end_time_ns": int(req.end_time),
                "latency_ns": int(req.latency),
                "ttft_ns": int(req.ttft),
                "tpot_ns": int(req.tpot),
                "itl_ns": [int(value) for value in req.itl],
                "input_tokens": int(req.original_input),
                "output_tokens": int(req.output),
            }
            for req in sorted(self.scheduler.done, key=lambda item: item.id)
        )

    def dump_step(self, snapshot: HBFrameBatchSnapshot, handle) -> None:
        handle.write(json.dumps(snapshot.to_dict(), sort_keys=True) + "\n")
