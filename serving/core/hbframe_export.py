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
from .gate_function import GateRouter
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
    npu_kv_capacity_mb: float | None = None
    weight_residency: str = "npu"
    num_npus: int = 1
    num_instances: int = 1
    pd_mode: str = "colocated"
    prefill_instances: int = 1
    decode_instances: int = 1
    pd_handoff_policy: str = "metadata_only"
    pd_handoff_latency_ns: int = 0
    tp_size: int = 1
    pp_size: int = 1
    ep_size: int = 1
    expert_routing_policy: str = "BALANCED"
    moe_hot_expert_fraction: float = 0.125
    moe_hot_token_fraction: float = 0.8
    instance_id: int = 0
    node_id: int = 0
    start_npu: int = 0
    pd_type: str | None = None
    prioritize_prefill: bool = False
    enable_prefix_caching: bool = False
    enable_prefix_sharing: bool = False
    enable_chunked_prefill: bool = True
    prefix_storage: str | None = None
    prefix_cache_mode: str = "manual"
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
    """Export generated-token count, not Request.output's total sequence limit."""
    payload = {
        "id": int(req.id),
        "model": req.model,
        "input_tokens": int(req.original_input),
        "output_tokens": max(0, int(req.output) - int(req.original_input)),
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
        "prefix_cache_source_request_id": getattr(req, "prefix_cache_source_request_id", None),
        "resume_from_initialized_context": bool(
            getattr(req, "resume_from_initialized_context", False)
        ),
        "session_id": getattr(req, "session_id", None),
        "sub_request_index": getattr(req, "sub_request_index", None),
    }
    for name in (
        "hbframe_prefill_source_instance_id",
        "hbframe_prefill_batch_id",
        "hbframe_prefill_done_ns",
        "hbframe_handoff_ready_ns",
        "hbframe_handoff_policy",
    ):
        if hasattr(req, name):
            payload[name] = getattr(req, name)
    return payload


def snapshot_batch(batch, scheduler: Scheduler, *, extra_attrs=None) -> HBFrameBatchSnapshot:
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
            "pd_role": scheduler.pd_type or "colocated",
            "hbframe_scheduler_decision": dict(getattr(scheduler, "_hbframe_last_decision", {}) or {}),
            "logical_memory": {
                "schema": "hbframe-logical-memory-v1",
                "pools": {
                    "L1": {
                        "capacity_bytes": int(scheduler.memory.npu_mem),
                        "used_bytes": int(scheduler.memory.npu_used),
                        "reserved_bytes": 0,
                        "scope": "per_rank",
                    },
                    "L2": {
                        "capacity_bytes": int(scheduler.memory.cpu_mem),
                        "used_bytes": int(scheduler.memory.cpu_used),
                        "reserved_bytes": 0,
                        "scope": "node",
                    },
                },
                "transfers": tuple(
                    dict(item)
                    for item in getattr(batch, "logical_pool_transfers", ())
                ),
            },
            **dict(extra_attrs or {}),
        },
    )


class HBFrameServingFrontend:
    """Callable LLMServingSim scheduler wrapper for HBFrame."""

    def __init__(self, config: HBFrameFrontendConfig, scheduler_oracle=None):
        self.config = config
        prefix_storage = _prefix_storage_device(config.prefix_storage)
        self.schedulers = []
        self._prefill_schedulers = []
        self._decode_schedulers = []
        self._pd_handoff_records: list[dict[str, object]] = []
        self._pd_decode_rr_counter = 0
        self._moe_gate = None
        self._moe_config = None

        def _make_scheduler(index: int, pd_type: str | None, *, start_npu: int | None = None):
            scheduler = Scheduler(
                config.model,
                config.node_id,
                config.instance_id + index,
                config.max_num_seqs,
                config.max_num_batched_tokens,
                config.num_npus,
                config.tp_size,
                config.pp_size,
                config.npu_mem_gb,
                config.cpu_mem_gb,
                config.start_npu if start_npu is None else start_npu,
                pd_type,
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
                weight_residency=config.weight_residency,
            )
            if config.npu_kv_capacity_mb is not None:
                kv_bytes = int(float(config.npu_kv_capacity_mb) * 1024 * 1024)
                resident_weight = int(getattr(scheduler.memory, "resident_weight", scheduler.memory.weight))
                scheduler.memory.npu_mem = resident_weight + max(0, kv_bytes)
                if getattr(scheduler.memory, "enable_prefix_caching", False):
                    scheduler.memory.mem_for_kv = max(0, scheduler.memory.npu_mem - resident_weight)
                    if hasattr(scheduler.memory, "npu_prefix_cache"):
                        scheduler.memory.npu_prefix_cache.capacity = scheduler.memory.mem_for_kv
            scheduler.hbframe_oracle = scheduler_oracle
            return scheduler

        if str(config.pd_mode).lower() == "disaggregated":
            prefill_count = max(1, int(config.prefill_instances))
            decode_count = max(1, int(config.decode_instances))
            for index in range(prefill_count):
                scheduler = _make_scheduler(index, "prefill", start_npu=config.start_npu + index * config.num_npus)
                self.schedulers.append(scheduler)
                self._prefill_schedulers.append(scheduler)
            for index in range(decode_count):
                absolute_index = prefill_count + index
                scheduler = _make_scheduler(absolute_index, "decode", start_npu=config.start_npu + absolute_index * config.num_npus)
                self.schedulers.append(scheduler)
                self._decode_schedulers.append(scheduler)
        else:
            for index in range(max(1, int(config.num_instances))):
                scheduler = _make_scheduler(index, config.pd_type)
                self.schedulers.append(scheduler)
            self._prefill_schedulers = list(self.schedulers)
            self._decode_schedulers = []
        self.scheduler = self.schedulers[0]
        model_config = self.scheduler.memory.config
        num_experts = model_config.get(
            "num_local_experts",
            model_config.get("num_experts", model_config.get("n_routed_experts")),
        )
        if num_experts:
            self._moe_config = model_config
            self._moe_gate = GateRouter(
                config.node_id,
                config.instance_id,
                num_experts,
                num_experts_per_tok=model_config.get("num_experts_per_tok", 1),
                routing_policy=config.expert_routing_policy,
                seed=42,
                block_copy=False,
                hot_expert_fraction=config.moe_hot_expert_fraction,
                hot_token_fraction=config.moe_hot_token_fraction,
            )
        self.router = Router(
            len(self.schedulers),
            self.schedulers,
            config.req_num,
            routing_policy=config.request_routing_policy,
            prefix_cache_mode=config.prefix_cache_mode,
        )
        self.current_time_ns = 0
        self._active_batches: dict[tuple[int, int], object] = {}
        self._initial_requests: list[dict[str, object]] = []
        self._initial_request_ids: set[int] = set()

    @property
    def pd_disaggregated(self) -> bool:
        return str(self.config.pd_mode).lower() == "disaggregated"

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
        loaded_requests = 0
        with open(path, "r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                if bool(row.get("initialize_only", False)):
                    self._load_initial_request(row)
                    continue
                if self.config.req_num > 0 and loaded_requests >= self.config.req_num:
                    continue
                if "sub_requests" in row:
                    self.router._load_agentic_session(row, self.config.enable_prefix_caching)
                else:
                    row = self._annotate_initialized_decode_resume(row)
                    self.router._load_flat_request(row, self.config.enable_prefix_caching)
                loaded_requests += 1
        self.router._pending_requests.sort(key=lambda request: request["arrival_time_ns"])

    def _annotate_initialized_decode_resume(
        self,
        row: dict[str, object],
    ) -> dict[str, object]:
        """Mark a full initialized-history continuation as decode-ready."""
        normalized = dict(row)
        input_tokens = int(normalized.get("input_toks", 0) or 0)
        prefix_hit = max(0, int(normalized.get("prefix_cache_hit", 0) or 0))
        source_id = normalized.get("prefix_cache_source_request_id")
        source_is_initialized = (
            source_id is not None and int(source_id) in self._initial_request_ids
        )
        explicit = normalized.get("resume_from_initialized_context")
        if explicit is None:
            resume = (
                self.config.enable_prefix_caching
                and str(self.config.prefix_cache_mode).lower() != "none"
                and source_is_initialized
                and input_tokens > 0
                and prefix_hit == input_tokens
            )
        else:
            resume = bool(explicit)
        if resume:
            if not source_is_initialized:
                raise ValueError(
                    "resume_from_initialized_context requires "
                    "prefix_cache_source_request_id to name an initialize_only request"
                )
            if input_tokens <= 0 or prefix_hit != input_tokens:
                raise ValueError(
                    "resume_from_initialized_context requires "
                    "prefix_cache_hit == input_toks > 0"
                )
        normalized["resume_from_initialized_context"] = resume
        return normalized

    def _load_initial_request(self, row: dict[str, object]) -> None:
        """Reserve and normalize one historical request without scheduling it."""
        if "sub_requests" in row:
            raise ValueError("initialize_only requests must use the flat workload format")
        if row.get("request_id") is None:
            raise ValueError("initialize_only requests require an explicit request_id")
        request_id = self.router._allocate_request_id(row["request_id"])
        input_tokens = int(row["input_toks"])
        output_tokens = int(row.get("output_toks", 0) or 0)
        prefix_hit = max(0, int(row.get("prefix_cache_hit", 0) or 0))
        if input_tokens < 0 or output_tokens < 0:
            raise ValueError(
                f"initialize_only request {request_id} requires non-negative input_toks and output_toks"
            )
        if prefix_hit > input_tokens:
            raise ValueError(
                f"initialize_only request {request_id} prefix_cache_hit={prefix_hit} "
                f"exceeds input_toks={input_tokens}"
            )
        source_request_id = row.get("prefix_cache_source_request_id")
        if prefix_hit > 0 and source_request_id is None:
            raise ValueError(
                f"initialize_only request {request_id} declares prefix_cache_hit={prefix_hit} "
                "but has no prefix_cache_source_request_id"
            )
        normalized: dict[str, object] = {
            "request_id": request_id,
            "session_id": row.get("session_id"),
            "input_toks": input_tokens,
            "output_toks": output_tokens,
            "prefix_cache_hit": prefix_hit,
            "initialize_only": True,
        }
        if source_request_id is not None:
            normalized["prefix_cache_source_request_id"] = int(source_request_id)
        self._initial_requests.append(normalized)
        self._initial_request_ids.add(request_id)

    @property
    def initial_requests(self) -> tuple[dict[str, object], ...]:
        """Historical requests to materialize before the serving timeline starts."""
        return tuple(dict(request) for request in self._initial_requests)

    def route_arrivals(self, current_time_ns: int) -> int:
        self.current_time_ns = int(current_time_ns)
        return int(self.router.route_arrived_requests(self.current_time_ns))

    def schedule_next(self, current_time_ns: int | None = None) -> HBFrameBatchSnapshot | None:
        if current_time_ns is not None:
            self.route_arrivals(int(current_time_ns))
        for scheduler in self.schedulers:
            batch = scheduler.schedule(self.current_time_ns, scheduler.start_npu)
            if batch is None:
                continue
            self._active_batches[(int(scheduler.instance_id), int(batch.batch_id))] = batch
            return snapshot_batch(
                batch,
                scheduler,
                extra_attrs=self._moe_execution_for_batch(batch),
            )
        return None

    def _moe_layer_ids(self):
        if self._moe_config is None:
            return ()
        layer_count = int(self._moe_config.get("num_hidden_layers", 0) or 0)
        first_dense = int(self._moe_config.get("first_k_dense_replace", 0) or 0)
        frequency = max(
            1,
            int(
                self._moe_config.get(
                    "moe_layer_freq",
                    self._moe_config.get("decoder_sparse_step", 1),
                )
                or 1
            ),
        )
        mlp_only_layers = {
            int(layer_id)
            for layer_id in (self._moe_config.get("mlp_only_layers", ()) or ())
        }
        return tuple(
            layer_id
            for layer_id in range(layer_count)
            if layer_id >= first_dense
            and ((layer_id - first_dense) % frequency == 0 or layer_id in mlp_only_layers)
        )

    @staticmethod
    def _batch_token_identities(batch):
        identities = []
        scheduled_tokens = getattr(batch, "scheduled_tokens", None) or {}
        for request in batch.requests:
            count = max(0, int(scheduled_tokens.get(request.id, 0) or 0))
            start = max(0, int(request.num_computed_tokens))
            identities.extend((int(request.id), start + offset) for offset in range(count))
        target = max(0, int(batch.total_len))
        while len(identities) < target:
            identities.append((int(batch.batch_id), len(identities)))
        return tuple(identities[:target])

    def _moe_execution_for_batch(self, batch):
        if self._moe_gate is None:
            return {}
        ep_size = max(1, int(self.config.ep_size))
        token_identities = self._batch_token_identities(batch)
        layers = {}
        expert_owners = None
        for layer_id in self._moe_layer_ids():
            route = self._moe_gate.route_ep(
                layer_id,
                batch.batch_id,
                batch.total_len,
                ep_size,
                token_identities=token_identities,
            )
            if expert_owners is None:
                expert_owners = list(route.expert_owners)
            layers[str(layer_id)] = {
                "expert_token_counts": {
                    str(expert_id): int(count)
                    for expert_id, count in sorted(route.expert_token_counts.items())
                    if int(count) > 0
                },
                "source_rank_expert_tokens": [
                    {
                        str(expert_id): int(count)
                        for expert_id, count in sorted(rank_counts.items())
                        if int(count) > 0
                    }
                    for rank_counts in route.source_rank_expert_tokens
                ],
                "source_tokens": [int(value) for value in route.source_tokens],
                "local_tokens": [int(value) for value in route.local_tokens],
                "activated_experts": [int(value) for value in route.activated_experts],
            }
        return {
            "moe_execution": {
                "schema": "llmservingsim-moe-execution-v1",
                "routing_source": "llmservingsim_frontend",
                "routing_policy": str(self.config.expert_routing_policy).lower(),
                "ep_size": ep_size,
                "expert_count": int(self._moe_gate.E),
                "top_k": int(self._moe_gate.k),
                "expert_owners": expert_owners or [],
                "layers": layers,
            }
        }

    def complete_batch(self, batch_id: int, completion_time_ns: int, instance_id: int | None = None) -> tuple[dict[str, object], ...]:
        """Mark a scheduled batch complete and return newly finished requests."""
        batch_id = int(batch_id)
        completion_time_ns = int(completion_time_ns)
        if instance_id is None:
            matches = [key for key in self._active_batches if key[1] == batch_id]
            if len(matches) != 1:
                raise KeyError(f"Unknown or ambiguous active batch_id={batch_id}")
            instance_id = matches[0][0]
        key = (int(instance_id), batch_id)
        if key not in self._active_batches:
            raise KeyError(f"Unknown active batch key={key}")
        scheduler = next(item for item in self.schedulers if int(item.instance_id) == int(instance_id))
        prompt_t, gen_t, finished = self._complete_scheduler_batch(scheduler, batch_id, completion_time_ns)
        self._active_batches.pop(key, None)
        self.current_time_ns = max(self.current_time_ns, completion_time_ns)
        if self.pd_disaggregated and scheduler.pd_type == "prefill":
            self._handoff_prefill_requests(finished, completion_time_ns, from_instance_id=int(instance_id), batch_id=batch_id)
            return ()
        for req in finished:
            self.router.notify_request_completed(req.id, completion_time_ns)
        return tuple(self._metric_dict(req, prompt_t=prompt_t, gen_t=gen_t) for req in finished)

    def _complete_scheduler_batch(self, scheduler: Scheduler, batch_id: int, completion_time_ns: int):
        systems = [int(scheduler.start_npu)]
        if scheduler.pd_type == "prefill":
            systems.append(int(scheduler.start_npu + scheduler.num_npus * 2 - 1))
        elif int(scheduler.num_npus) > 1:
            systems.append(int(scheduler.start_npu + scheduler.num_npus - 1))
        prompt_t = 0
        gen_t = 0
        finished = []
        seen = set()
        for sys_id in dict.fromkeys(systems):
            prompt_t, gen_t, finished = scheduler.add_done(batch_id + 1, sys_id, completion_time_ns)
            if finished:
                seen.update(int(req.id) for req in finished)
        return prompt_t, gen_t, finished

    def _handoff_prefill_requests(self, requests, completion_time_ns: int, *, from_instance_id: int, batch_id: int) -> None:
        if not requests:
            return
        decode_schedulers = self._decode_schedulers or [scheduler for scheduler in self.schedulers if scheduler.pd_type == "decode"]
        if not decode_schedulers:
            raise RuntimeError("PD disaggregated mode requires at least one decode scheduler")
        ready_time_ns = int(completion_time_ns) + max(0, int(self.config.pd_handoff_latency_ns))
        for req in requests:
            decode_scheduler = decode_schedulers[self._pd_decode_rr_counter % len(decode_schedulers)]
            self._pd_decode_rr_counter += 1
            original_arrival = int(getattr(req, "hbframe_original_arrival", req.arrival))
            req.hbframe_original_arrival = original_arrival
            req.hbframe_prefill_ttft = int(req.ttft)
            req.hbframe_prefill_source_instance_id = int(from_instance_id)
            req.hbframe_prefill_batch_id = int(batch_id)
            req.hbframe_prefill_done_ns = int(completion_time_ns)
            req.hbframe_handoff_ready_ns = int(ready_time_ns)
            req.hbframe_handoff_policy = self.config.pd_handoff_policy
            req.arrival = int(ready_time_ns)
            decode_scheduler.add_decode(req)
            decode_scheduler.request.sort(key=lambda item: (item.arrival, item.id))
            self._pd_handoff_records.append(
                {
                    "request_id": int(req.id),
                    "from_instance_id": int(from_instance_id),
                    "to_instance_id": int(decode_scheduler.instance_id),
                    "prefill_batch_id": int(batch_id),
                    "prefill_done_ns": int(completion_time_ns),
                    "decode_ready_ns": int(ready_time_ns),
                    "policy": self.config.pd_handoff_policy,
                    "input_tokens": int(req.original_input),
                    "computed_tokens": int(req.num_computed_tokens),
                }
            )

    def _metric_dict(self, req, *, prompt_t: int = 0, gen_t: int = 0) -> dict[str, object]:
        arrival_ns = int(getattr(req, "hbframe_original_arrival", req.arrival))
        end_ns = int(req.end_time)
        latency_ns = max(0, end_ns - arrival_ns)
        ttft_ns = int(getattr(req, "hbframe_prefill_ttft", req.ttft))
        output_tokens = max(0, int(req.output) - int(req.original_input))
        if output_tokens <= 1:
            tpot_ns = 0
        else:
            tpot_ns = max(0, latency_ns - ttft_ns) // max(1, output_tokens - 1)
        return {
            "id": int(req.id),
            "arrival_time_ns": arrival_ns,
            "end_time_ns": end_ns,
            "latency_ns": latency_ns,
            "ttft_ns": ttft_ns,
            "tpot_ns": int(tpot_ns),
            "output_tokens": output_tokens,
            "itl_ns": [int(value) for value in req.itl],
            "prompt_tokens_accounted": int(prompt_t),
            "generation_tokens_accounted": int(gen_t),
        }

    def next_arrival_time_ns(self) -> int | None:
        candidates = []
        value = self.router.get_next_pending_arrival()
        if value is not None:
            candidates.append(int(value))
        for scheduler in self.schedulers:
            if scheduler.request:
                candidates.append(int(scheduler.request[0].arrival))
        return min(candidates) if candidates else None

    def has_pending_work(self) -> bool:
        return (
            self.router.has_pending_requests()
            or bool(self.scheduler.request)
            or any(bool(scheduler.request) or bool(scheduler.inflight) for scheduler in self.schedulers)
            or bool(self._active_batches)
        )

    def request_metrics(self) -> tuple[dict[str, object], ...]:
        return tuple(
            {
                **self._metric_dict(req),
                "input_tokens": int(req.original_input),
            }
            for req in sorted((req for scheduler in self.schedulers for req in scheduler.done), key=lambda item: item.id)
        )

    def pd_handoffs(self) -> tuple[dict[str, object], ...]:
        return tuple(dict(record) for record in self._pd_handoff_records)

    def dump_step(self, snapshot: HBFrameBatchSnapshot, handle) -> None:
        handle.write(json.dumps(snapshot.to_dict(), sort_keys=True) + "\n")
