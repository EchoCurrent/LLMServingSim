import os, threading
from .utils import get_config
from .radix_tree import *
import logging
from enum import Enum

GB_TO_BYTE = 1024 * 1024 * 1024
MB_TO_BYTE = 1024 * 1024
KB_TO_BYTE = 1024

def _uses_mla(config):
    return all(int(config.get(key, 0) or 0) > 0 for key in (
        "qk_nope_head_dim",
        "qk_rope_head_dim",
        "v_head_dim",
        "kv_lora_rank",
    ))

def _attention_dims(config):
    n_embd = config['hidden_size']
    n_head = config['num_attention_heads']
    if _uses_mla(config):
        qk_nope_head_dim = int(config["qk_nope_head_dim"])
        qk_rope_head_dim = int(config["qk_rope_head_dim"])
        qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
        v_head_dim = int(config["v_head_dim"])
        kv_lora_rank = int(config["kv_lora_rank"])
        return {
            "is_mla": True,
            "head_dim": qk_head_dim,
            "kv_head": 1,
            "q_dim": n_head * qk_head_dim,
            "v_dim": n_head * v_head_dim,
            # vLLM MLA stores one latent KV vector per token, not separate
            # per-head K and V tensors.
            "kv_cache_dim": kv_lora_rank + qk_rope_head_dim,
            "qk_nope_head_dim": qk_nope_head_dim,
            "qk_rope_head_dim": qk_rope_head_dim,
            "v_head_dim": v_head_dim,
            "kv_lora_rank": kv_lora_rank,
            "q_lora_rank": config.get("q_lora_rank"),
        }
    head_dim = config.get('head_dim', n_embd // n_head)
    kv_head = config.get("num_key_value_heads", n_head)
    kv_dim = kv_head * head_dim
    return {
        "is_mla": False,
        "head_dim": head_dim,
        "kv_head": kv_head,
        "q_dim": n_head * head_dim,
        "v_dim": n_head * head_dim,
        "kv_cache_dim": kv_dim,
    }

class Device(Enum):
    NPU = 1
    CPU = 2
    CXL = 3


def _union_bytes(ranges):
    end = total = 0
    for left, right in sorted(ranges):
        total += max(0, right - max(left, end))
        end = max(end, right)
    return total


class MemoryModel():
    def __init__(self, model, instance_id, node_id, num_npus, tp_size, npu_mem, cpu_mem, block_size, fp, enable_prefix_caching, enable_prefix_sharing, prefix_pool, prefix_storage, cxl_mem=0, ep_size=1, pp_size=1, kv_cache_dtype='auto', weight_residency='npu', l2_retains_l1=False, l2_scope='node', kv_eviction_has_fallback=True, model_config=None, rank_memory=None):
        self.model = model
        self.node_id = node_id
        self.instance_id = instance_id
        self.num_npus = num_npus
        self.tp_size = tp_size
        self.pp_size = pp_size
        self.ep_size = ep_size
        self.npu_mem = npu_mem * GB_TO_BYTE # GB -> Byte
        self.cpu_mem = cpu_mem * GB_TO_BYTE # GB -> Byte
        self.cxl_mem = cxl_mem * GB_TO_BYTE
        self.block_size = block_size
        self.fp = fp // 8 # bit -> byte of floating point
        self.kv_fp = 1 if kv_cache_dtype == 'fp8' else self.fp  # KV cache bytes per element
        self.enable_prefix_caching = enable_prefix_caching
        self.enable_prefix_sharing = enable_prefix_sharing
        self.prefix_storage = prefix_storage
        self.l2_retains_l1 = bool(l2_retains_l1)
        self.l2_scope = str(l2_scope)
        # No fallback tier means the deployment forbids KV offload: an evicted
        # range is dropped, not written out, so it must not be charged to L2.
        self.kv_eviction_has_fallback = bool(kv_eviction_has_fallback)
        self._l1_request_bytes = {}
        self._l2_request_bytes = {}
        # Initialized history is capacity ballast: it occupies L1 so the
        # workload starts under real pressure, and it may be evicted to make
        # room. It has no backing copy, so an evicted history request is
        # dropped rather than reloaded.
        self._initial_request_ids = set()

        self.config = get_config(model) if model_config is None else model_config
        self.rank_memory = rank_memory
        self.n_embd = self.config['hidden_size']
        self.n_layer = self.config['num_hidden_layers']
        self.n_head = self.config['num_attention_heads']
        attention_dims = _attention_dims(self.config)
        self.head_dim = attention_dims["head_dim"]
        self.kv_head = attention_dims["kv_head"]
        self.q_dim = attention_dims["q_dim"]
        self.kv_dim = attention_dims["kv_cache_dim"]
        self.uses_mla = attention_dims["is_mla"]
        self.vocab_size = self.config['vocab_size']
        # Accept either the Mistral-style ``num_local_experts`` or the
        # HF/Qwen-style ``num_experts`` key — profiler configs track
        # upstream HF naming which varies per family.
        self.is_moe = 'num_local_experts' in self.config or 'num_experts' in self.config

        self.logger = get_logger(self.__class__, node_id=node_id, instance_id=instance_id)

        # Memory model. The default preserves LLMServingSim's original
        # assumption that full weights are resident on NPU memory. HBFrame can
        # explicitly request offloaded weights and model weight movement itself.
        self.weight = (rank_memory[min(rank_memory)]['weight_bytes']
                       if rank_memory is not None else self.get_weight())
        self.weight_residency = str(weight_residency or 'npu').lower()
        npu_resident_modes = {'npu', 'hbm', 'resident', 'resident_npu'}
        offloaded_modes = {'offload', 'offloaded', 'streaming', 'external', 'hbf', 'ssd', 'cpu'}
        if self.weight_residency not in npu_resident_modes and self.weight_residency not in offloaded_modes:
            raise ValueError(f"Unsupported weight_residency: {weight_residency}")
        self.weight_resident_on_npu = self.weight_residency in npu_resident_modes
        self.resident_weight = self.weight if self.weight_resident_on_npu else 0
        self.npu_used = self.resident_weight
        self.cpu_used = 0
        if self.weight_resident_on_npu and self.weight > self.npu_mem:
            raise RuntimeError(f"[MemoryModel] [node={self.node_id},inst={self.instance_id}]: Model size {self.weight*self.num_npus//GB_TO_BYTE}GB exceeds total NPU memory {self.npu_mem*self.num_npus//GB_TO_BYTE}GB")

        if enable_prefix_caching:
            one_token_kv_size = self.get_kv(1)
            self.mem_for_kv = self.npu_mem - self.resident_weight
            self.npu_prefix_cache = RadixCache(device='NPU', 
                                               node_id=self.node_id,
                                               instance_id=self.instance_id,
                                               page_size=self.block_size,
                                               capacity=self.mem_for_kv,
                                               kv_size=one_token_kv_size,
                                               enable_kv_cache_events=True,
                                                )
            if prefix_storage is not None:
                if enable_prefix_sharing and prefix_pool is not None:
                    self.second_tier_prefix_cache = prefix_pool
                else:
                    prefix_cache_capacity = 0
                    if prefix_storage == Device.CPU:
                        device = "CPU"
                        prefix_cache_capacity = self.cpu_mem
                    elif prefix_storage == Device.CXL:
                        device = "CXL"
                        prefix_cache_capacity = self.cxl_mem
                    else:
                        raise RuntimeError(f"[MemoryModel] [node_id={self.node_id},inst={self.instance_id}]: Device {prefix_storage} is currently not supported as a second tier prefix cache storage")
                    # print("[instance {}] prefix_cache_capacity : {}".format(instance_id, prefix_cache_capacity // GB_TO_BYTE))
                    self.second_tier_prefix_cache = RadixCache(device=device, 
                                                    node_id=self.node_id,
                                                    instance_id=self.instance_id,
                                                    page_size=1,
                                                    capacity=prefix_cache_capacity,
                                                    kv_size=(one_token_kv_size * self.num_npus),
                                                    enable_kv_cache_events=True,
                                                    )
                
        # Hash id -> token length for corresponding prefix cache block
        self._npu_cache_hashtolen = {}
        self._cpu_cache_hashtolen = {}
        self.system_state = None
        self.require_local_kv = False
        self._bytes_per_token = self.get_kv(1)  # bytes per token for kv cache

    def update_system_state(self, state, active_requests):
        """Rebase the native ledger from a quiescent, data-only system snapshot."""
        self._system_active_requests = tuple(active_requests)
        if self.system_state is None:
            self._system_l1_limit = self.npu_mem - self.resident_weight
            self._system_l2_limit = self.cpu_mem
        self.system_state = state
        self._system_evicted = set()
        self._system_objects = {row["object_uid"]: row for row in state.kv_objects}
        self._system_mappings = {}
        for row in state.kv_mappings:
            if row["instance_id"] == self.instance_id:
                self._system_mappings.setdefault((row["rank_id"], row["request_id"]), []).append(row)
        self._system_pools = {row["pool_id"]: {**row,
            "capacity_bytes": min(row["capacity_bytes"], self._system_l1_limit
                if row["logical_tier"] == "L1" else self._system_l2_limit),
            "used_bytes": row["used_bytes"] + row["reserved_bytes"]}
            for row in state.pools if row["instance_id"] == self.instance_id}
        self._system_l1 = {row["rank_id"]: row for row in self._system_pools.values()
                           if row["logical_tier"] == "L1"}
        self._system_l2 = {row["rank_id"]: row for row in self._system_pools.values()
                           if row["logical_tier"] == "L2"}
        self._system_pool_objects = {pid: {uid for uid, obj in self._system_objects.items()
            if pid in obj["ready_pool_ids"]} for pid in self._system_pools}
        # Identity projections charge object bytes. Replicated views publish
        # overrides so readable backing is not mistaken for reclaimable space.
        self._system_pool_charges = {pid: {uid: obj.get("pool_charge_bytes", {}).get(pid, obj["size_bytes"])
            for uid, obj in self._system_objects.items()
            if pid in obj["ready_pool_ids"] or pid in obj.get("pool_charge_bytes", {})}
            for pid in self._system_pools}
        self._system_active_ids = {req.id for req in self._system_active_requests if not req.evict}
        self._system_live_ids = {req.id for req in self._system_active_requests}
        self._refresh_system_reclaimable()
        self._sync_system_ledger()

    def _refresh_system_reclaimable(self):
        self._system_cold_objects = {}
        for pid, charges in self._system_pool_charges.items():
            protected_ids = (self._system_live_ids if self._system_pools[pid]["logical_tier"] == "L2"
                             else self._system_active_ids - self._system_evicted)
            protected = {row["object_uid"] for (rank, rid), rows in self._system_mappings.items()
                         if rid in protected_ids for row in rows}
            self._system_cold_objects[pid] = charges.keys() - protected
        self._system_cold_bytes = {pid: sum(self._system_pool_charges[pid][uid] for uid in objects)
                                   for pid, objects in self._system_cold_objects.items()}

    def _sync_system_ledger(self):
        """Expose one rank's ledger for observations; admission checks every pool."""
        self._system_reference_rank = min(self._system_l1)
        l1 = self._system_l1[self._system_reference_rank]
        l2 = self._system_l2.get(self._system_reference_rank, self._system_l2.get(None))
        self.npu_mem = l1["capacity_bytes"] + self.resident_weight
        self.npu_used = l1["used_bytes"] + self.resident_weight
        self.cpu_mem, self.cpu_used = l2["capacity_bytes"], l2["used_bytes"]

    def _system_batch_space(self, requests, scheduled_tokens):
        """Count new bytes and the union of restored ranges in each logical pool."""
        growth = {pid: 0 for pid in self._system_pools}
        data_growth = {pid: 0 for pid, pool in self._system_pools.items() if "data_capacity_bytes" in pool}
        protected = set()
        for rank, pool in self._system_l1.items():
            pid = pool["pool_id"]
            restored = {}
            restored_slack = {}
            scale = pool.get("allocation_charge_scale", 1)
            unit = self.get_kv(1) if self.rank_memory is None else self.rank_memory[rank]['kv_bytes_per_token']
            for req in requests:
                rows = self._system_mappings.get((rank, req.id), ())
                if not rows and req.prefix_cache_source_request_id is not None:
                    rows = self._system_mappings.get((rank, req.prefix_cache_source_request_id), ())
                mapped = 0
                slack = 0
                continuation = {}
                boundary = req.num_computed_tokens // self.block_size * self.block_size
                for row in rows:
                    count = max(0, min(req.num_computed_tokens,
                        row["token_start"] + row["token_count"]) - row["token_start"])
                    uid = row["object_uid"]
                    if not count or not self._system_objects[uid]["ready_pool_ids"]:
                        continue
                    size = row["size_bytes"] * count // row["token_count"]
                    mapped += size
                    if (row["token_start"] <= boundary < row["token_start"] + count
                            and "append_retains_allocation" in row):
                        continuation[row["layer_id"]] = row["append_retains_allocation"]
                    owned_slack = (row.get("reserved_bytes", 0) if row["request_id"] == req.id
                                   and count == row["token_count"] else 0)
                    slack += owned_slack
                    protected.add(uid)
                    if uid not in self._system_pool_objects[pid]:
                        offset = row["object_offset_bytes"]
                        restored.setdefault(uid, []).append((offset, offset + size))
                        restored_slack[uid] = max(restored_slack.get(uid, 0), owned_slack)
                after = req.num_computed_tokens + scheduled_tokens[req.id]
                full = (after + self.block_size - 1) // self.block_size * self.block_size * unit
                added = max(0, full - mapped - slack)
                if pid in data_growth:
                    data_growth[pid] += added
                fraction = pool.get("streaming_allocation_fraction")
                count = scheduled_tokens[req.id]
                if fraction is not None and count >= 4 * self.block_size:
                    # Streaming allocations release completed units while the
                    # retained prefix and unfinished boundary units stay live.
                    start = req.num_computed_tokens
                    end = start + count
                    block = self.block_size
                    retained = round(count * fraction / block) * block
                    retained_end = min(end, max(start, (start + retained + block // 2) // block * block))
                    head_end = min(end, (start + block - 1) // block * block)
                    live = max(head_end, retained_end) - start
                    live += max(0, (end + block - 1) // block * block
                                - max(head_end, retained_end, end // block * block))
                    charge = max(0, live * unit - slack)
                else:
                    charge = added
                    distribution = pool.get("allocation_distribution")
                    if distribution is not None and added:
                        # A published allocation unit may be released after
                        # filling. Charge its peak footprint, not an average
                        # fraction of the requested data bytes.
                        first = full // unit - added // unit
                        charge = 0
                        for allocation in distribution["units"]:
                            live = peak = 0
                            for left in range(first // self.block_size * self.block_size,
                                              full // unit, self.block_size):
                                size = (min(left+self.block_size, full//unit) - max(left, first))
                                peak = max(peak, live + size)
                                ordinal = max(0, req.id) + allocation["phase"] + left // self.block_size
                                released = distribution["released_weight"]
                                total = distribution["total_weight"]
                                closes = left + self.block_size <= after
                                retain = (ordinal+1)*released//total == ordinal*released//total
                                if left == boundary and allocation["layer_id"] in continuation:
                                    retain = continuation[allocation["layer_id"]]
                                if not closes or retain:
                                    live += size
                            charge += peak * allocation["bytes_per_token"]
                growth[pid] += int(charge * scale)
                if self.l2_retains_l1:
                    backing = self._system_l2.get(rank, self._system_l2.get(None))
                    growth[backing["pool_id"]] += added
            for uid, ranges in restored.items():
                obj = self._system_objects[uid]
                cost = obj.get("restore_charge_bytes", {}).get(pid, obj["size_bytes"] * scale)
                growth[pid] += int(_union_bytes(ranges) * cost / obj["size_bytes"]
                                         + restored_slack[uid] * scale)
                if pid in data_growth:
                    data_growth[pid] += _union_bytes(ranges) + restored_slack[uid]
        reclaimable = {pid: min(pool["used_bytes"], self._system_cold_bytes[pid]
            - sum(self._system_pool_charges[pid][uid] for uid in protected & self._system_cold_objects[pid]))
            for pid, pool in self._system_pools.items()}
        data_demand = {pid: (added, sum(self._system_objects[uid].get("allocation_bytes",
            self._system_objects[uid]["size_bytes"]) for uid in self._system_cold_objects[pid] - protected
            if uid in self._system_pool_objects[pid])) for pid, added in data_growth.items()}
        return growth, reclaimable, data_demand

    def _system_growth(self, requests, scheduled_tokens):
        """Additional L1 bytes for a native candidate batch; aliases count once."""
        growth, _, _ = self._system_batch_space(requests, scheduled_tokens)
        return growth[self._system_l1[self._system_reference_rank]["pool_id"]]

    def batch_memory(self, requests, scheduled_tokens, *, reclaim_prefix=False):
        """Return admission and representative byte demand for a candidate batch."""
        if self.system_state is None:
            size = self.get_block_kv(requests, len(requests), scheduled_tokens)
            available = self.avail_bytes(Device.NPU)
            if reclaim_prefix:
                available = self.avail_size(Device.NPU) + self.evictable_size(Device.NPU)
            return size <= available, size
        growth, reclaimable, data_demand = self._system_batch_space(requests, scheduled_tokens)
        fits = all(pool["used_bytes"] + growth[pid] - reclaimable[pid] <= pool["capacity_bytes"]
                   for pid, pool in self._system_pools.items())
        fits = fits and all(self._system_pools[pid]["data_used_bytes"] + added - cold
            <= self._system_pools[pid]["data_capacity_bytes"] for pid, (added, cold) in data_demand.items())
        return fits, growth[self._system_l1[self._system_reference_rank]["pool_id"]]

    def prefix_availability(self, req):
        """Resolve an explicit prefix source from the last system snapshot."""
        source = req.prefix_cache_source_request_id
        if self.system_state is None or source is None:
            return {}
        objects = {row["object_uid"]: row for row in self.system_state.kv_objects}
        ranges = {}
        limit = int(req.prefix_cache_hit)
        for row in self.system_state.kv_mappings:
            if row["instance_id"] != self.instance_id or row["request_id"] != source:
                continue
            key = (row["rank_id"], row["layer_id"])
            ranges.setdefault(key, []).append(row)
        ranks = {row["rank_id"] for row in self.system_state.pools
                 if row["instance_id"] == self.instance_id and row["logical_tier"] == "L1"}
        required = {(rank, layer) for rank in ranks for layer in self.system_state.kv_layer_ids}
        hits, local_hits = [], []
        for key in required:
            rows = ranges.get(key, ())
            end = local_end = 0
            for row in sorted(rows, key=lambda item: item["token_start"]):
                pools = objects[row["object_uid"]]["ready_pool_ids"]
                if not pools or row["token_start"] > end:
                    break
                end = max(end, row["token_start"] + row["token_count"])
                if row["token_start"] <= local_end and any(pool.endswith(":L1") for pool in pools):
                    local_end = max(local_end, row["token_start"] + row["token_count"])
            hits.append(min(limit, end))
            local_hits.append(min(limit, local_end))
        hit = min(hits, default=0)
        local = min(local_hits, default=0)
        return dict(prefix_cache_hit=hit, npu_cache_hit=local,
                    storage_cache_hit=hit if hit > local else 0,
                    restore_prefix_to_l1=hit > local)

    @property
    def _l2_size_scale(self):
        return 1 if self.l2_scope == 'per_rank' else self.num_npus

    def reserve_logical_batch(self, requests, scheduled_tokens):
        """Reserve inclusive L2 backing and record active L1 membership."""
        if self.system_state is not None:
            growth, reclaimable, data_demand = self._system_batch_space(requests, scheduled_tokens)
            for pid, pool in self._system_pools.items():
                deficit = max(0, pool["used_bytes"] + growth[pid] - pool["capacity_bytes"])
                if deficit > reclaimable[pid]:
                    raise RuntimeError(f"Candidate exceeds logical pool {pid}")
                pool["used_bytes"] += growth[pid] - deficit
            for pid, (added, cold) in data_demand.items():
                pool = self._system_pools[pid]
                deficit = max(0, pool["data_used_bytes"] + added - pool["data_capacity_bytes"])
                if deficit > cold:
                    raise RuntimeError(f"Candidate exceeds logical data capacity {pid}")
                pool["data_used_bytes"] += added - deficit
            self._sync_system_ledger()
            return
        if not self.l2_retains_l1:
            return
        updates = []
        added_l2 = 0
        for req in requests:
            scheduled = max(0, int(scheduled_tokens.get(req.id, 0) or 0))
            tokens_after = max(0, int(req.num_computed_tokens) + scheduled)
            shared_prefix = (
                max(0, int(getattr(req, 'prefix_cache_hit', 0) or 0))
                if getattr(req, 'prefix_cache_source_request_id', None) is not None
                else 0
            )
            owned_tokens = max(0, tokens_after - shared_prefix)
            blocks = (owned_tokens + self.block_size - 1) // self.block_size
            backed_bytes = self.get_kv(blocks * self.block_size)
            previous = int(self._l2_request_bytes.get(int(req.id), 0))
            added_l2 += max(0, backed_bytes - previous) * self._l2_size_scale
            active_blocks = (tokens_after + self.block_size - 1) // self.block_size
            active_bytes = self.get_kv(active_blocks * self.block_size)
            updates.append((int(req.id), backed_bytes, active_bytes))
        if self.cpu_used + added_l2 > self.cpu_mem:
            raise RuntimeError(
                f"[MemoryModel] [node_id={self.node_id},inst={self.instance_id}] "
                f"L2 backing needs {added_l2 / MB_TO_BYTE:.2f}MB but only "
                f"{(self.cpu_mem - self.cpu_used) / MB_TO_BYTE:.2f}MB is available."
            )
        self.cpu_used += added_l2
        for request_id, backed_bytes, active_bytes in updates:
            self._l2_request_bytes[request_id] = max(
                int(self._l2_request_bytes.get(request_id, 0)), backed_bytes
            )
            self._l1_request_bytes[request_id] = active_bytes

    def reserve_initial_request(self, request_id, token_count, shared_tokens=0):
        """Charge an initialized history request to the L1 and L2 ledgers.

        ``initialize_only`` requests never enter the scheduler, so the ordinary
        ``allocate(kv_size, Device.NPU)`` call cannot charge their KV. Their
        state is already committed on the host, so the frontend must account
        for it here or it will see free capacity that does not exist and skip
        eviction and admission limits. Initialized KV is always placed in a
        compute-visible tier, which is exactly L1.
        """
        request_id = int(request_id)
        owned_tokens = max(0, int(token_count) - max(0, int(shared_tokens)))
        blocks = (owned_tokens + self.block_size - 1) // self.block_size
        active_bytes = self.get_kv(blocks * self.block_size)
        if self.npu_used + active_bytes > self.npu_mem:
            raise RuntimeError(
                f"[MemoryModel] [node_id={self.node_id},inst={self.instance_id}] "
                f"initialized L1 KV needs {active_bytes / MB_TO_BYTE:.2f}MB but only "
                f"{(self.npu_mem - self.npu_used) / MB_TO_BYTE:.2f}MB is available."
            )
        self.npu_used += active_bytes
        self._l1_request_bytes[request_id] = active_bytes
        self._initial_request_ids.add(request_id)

        if not self.l2_retains_l1:
            return
        previous = int(self._l2_request_bytes.get(request_id, 0))
        added = max(0, active_bytes - previous) * self._l2_size_scale
        if self.cpu_used + added > self.cpu_mem:
            raise RuntimeError(
                f"[MemoryModel] [node_id={self.node_id},inst={self.instance_id}] "
                f"initialized L2 backing needs {added / MB_TO_BYTE:.2f}MB but only "
                f"{(self.cpu_mem - self.cpu_used) / MB_TO_BYTE:.2f}MB is available."
            )
        self.cpu_used += added
        self._l2_request_bytes[request_id] = max(previous, active_bytes)

    def deactivate_l1(self, request_id):
        if self.system_state is not None:
            self._system_evicted.add(request_id)
            for rank, pool in self._system_l1.items():
                owned = self._system_exclusive_objects(rank, request_id)
                removed = owned & self._system_pool_charges[pool["pool_id"]].keys()
                if "data_used_bytes" in pool:
                    pool["data_used_bytes"] -= sum(self._system_objects[uid].get("allocation_bytes",
                        self._system_objects[uid]["size_bytes"]) for uid in removed
                        if uid in self._system_pool_objects[pool["pool_id"]])
                freed = min(pool["used_bytes"], sum(self._system_pool_charges[pool["pool_id"]][uid] for uid in removed))
                pool["used_bytes"] -= freed
                self._system_pool_objects[pool["pool_id"]].difference_update(removed)
                for uid in removed:
                    del self._system_pool_charges[pool["pool_id"]][uid]
                if self.kv_eviction_has_fallback and not self.l2_retains_l1:
                    l2 = self._system_l2.get(rank, self._system_l2.get(None))
                    l2["used_bytes"] += freed
            self._refresh_system_reclaimable()
            self._sync_system_ledger()
        if self.l2_retains_l1:
            self._l1_request_bytes.pop(int(request_id), None)

    def discard_l1(self, request_id):
        """Release one request's L1 charge without producing a backing copy.

        Used when a deployment has no fallback tier: the KV is dropped, not
        written out, so the L2 ledger must not grow. Initialized history is
        dropped the same way, which is why it is tracked separately.
        """
        request_id = int(request_id)
        if self.system_state is not None:
            self.deactivate_l1(request_id)
            return
        released = int(self._l1_request_bytes.pop(request_id, 0))
        self._initial_request_ids.discard(request_id)
        if released > 0:
            self.npu_used = max(self.resident_weight, self.npu_used - released)
        if self.l2_retains_l1:
            self._l2_request_bytes.pop(request_id, None)
        return released

    def has_backing_for(self, request_id) -> bool:
        """Whether an evicted request can be reloaded instead of recomputed."""
        return int(request_id) in self._l2_request_bytes and self.l2_retains_l1

    def evict_initial_history(self, required_bytes: int) -> list[int]:
        """Drop initialized history until ``required_bytes`` is free.

        Initialized history is capacity ballast with no backing copy, so it is
        the first thing to give up when an active sequence needs room. Returns
        the request ids that were dropped, oldest charge first.
        """
        freed = 0
        dropped: list[int] = []
        for request_id in sorted(self._initial_request_ids):
            if freed >= int(required_bytes):
                break
            released = self.discard_l1(request_id)
            if released > 0:
                freed += released
                dropped.append(int(request_id))
        return dropped

    def initial_history_bytes(self) -> int:
        return sum(int(self._l1_request_bytes.get(rid, 0)) for rid in self._initial_request_ids)

    def avail_bytes(self, device) -> int:
        """Free bytes on a device, independent of the prefix-cache accounting."""
        if device == Device.NPU:
            return max(0, int(self.npu_mem) - int(self.npu_used))
        if device in (Device.CPU, Device.CXL):
            limit = self.cpu_mem if device == Device.CPU else self.cxl_mem
            return max(0, int(limit) - int(self.cpu_used))
        raise RuntimeError(
            f"[MemoryModel] [node_id={self.node_id},inst={self.instance_id}] "
            f"Trying to get free size of unsupported device {device}"
        )

    def release_logical_request(self, request_id):
        if not self.l2_retains_l1:
            return
        request_id = int(request_id)
        self._l1_request_bytes.pop(request_id, None)
        released = self._l2_request_bytes.pop(request_id, 0) * self._l2_size_scale
        self.cpu_used -= released

    def logical_pool_snapshot(self):
        l1_kv_used = max(0, int(self.npu_used) - int(self.resident_weight))
        overlap = 0
        if self.system_state is not None:
            l1 = self._system_l1[self._system_reference_rank]
            l2 = self._system_l2.get(self._system_reference_rank, self._system_l2.get(None))
            paired = sum(self._system_objects[uid]["size_bytes"] for uid in
                self._system_pool_objects[l1["pool_id"]] & self._system_pool_objects[l2["pool_id"]])
            # Balanced L1 charges the active HBM working footprint in logical
            # bytes; readable cold HBF backing can exceed that active footprint.
            overlap = min(l1_kv_used, l2["used_bytes"], paired)
        elif self.l2_retains_l1:
            overlap = min(
                l1_kv_used,
                sum(
                    min(size, int(self._l2_request_bytes.get(request_id, 0)))
                    for request_id, size in self._l1_request_bytes.items()
                ),
            )
        # ``used_bytes`` counts everything the ledger holds, which on L1 includes
        # resident model weights; ``kv_used_bytes`` isolates the KV occupancy a
        # caller comparing against a KV budget wants. ``weight_bytes`` closes the
        # identity used_bytes == kv_used_bytes + weight_bytes on both tiers.
        return {
            'L1': {
                'capacity_bytes': int(self.npu_mem),
                'used_bytes': int(self.npu_used),
                'kv_used_bytes': int(l1_kv_used),
                'weight_bytes': int(self.resident_weight),
                'reserved_bytes': 0,
                'scope': 'per_rank',
                'overlap_bytes': int(overlap),
            },
            'L2': {
                'capacity_bytes': int(self.cpu_mem),
                'used_bytes': int(self.cpu_used),
                'kv_used_bytes': int(self.cpu_used),
                'weight_bytes': 0,
                'reserved_bytes': 0,
                'scope': self.l2_scope,
                'overlap_bytes': int(overlap),
            },
        }

    def get_weight(self):
        """Per-GPU model weight in bytes.

        Conservative upper bound across PP ranks: assumes a single rank
        holds embedding + final_layernorm + lm_head along with its share
        of transformer blocks (n_layer // pp_size). In real PP these
        non-block weights live on the first/last rank only, so middle
        ranks are lighter — but using the heaviest-rank value here keeps
        the `weight > npu_mem` check safe.
        """
        tp = self.tp_size
        pp = max(self.pp_size, 1)
        ep = self.ep_size
        fp = self.fp
        weight = 0

        _, embedding, _ = calculate_sizes(self.model, 'embedding', 1, parallel=tp, fp=fp)
        weight += embedding
        weight += self._get_weight_per_block(tp, ep, fp) * (self.n_layer // pp)
        _, ln_f, _ = calculate_sizes(self.model, 'final_layernorm', 1, parallel=tp, fp=fp)
        weight += ln_f
        _, lm_head, _ = calculate_sizes(self.model, 'lm_head', 1, parallel=tp, fp=fp)
        weight += lm_head

        self.logger.info(
            "NPU: model weight %dMB loaded",
            weight * tp // MB_TO_BYTE,
        )
        return weight

    def _get_weight_per_block(self, tp, ep, fp):
        """Per-block weight: dense layers use TP, MoE experts use EP."""
        block_weight = 0
        _, ln_w, _ = calculate_sizes(self.model, 'layernorm', 1, parallel=tp, fp=fp)
        block_weight += ln_w  # input layernorm
        _, qkv_w, _ = calculate_sizes(self.model, 'qkv_proj', 1, parallel=tp, fp=fp)
        block_weight += qkv_w
        _, o_w, _ = calculate_sizes(self.model, 'o_proj', 1, parallel=tp, fp=fp)
        block_weight += o_w
        block_weight += ln_w  # post layernorm (same weight size)
        if self.is_moe:
            _, moe_w, _ = calculate_sizes(self.model, 'moe', 1, parallel=ep, fp=fp)
            block_weight += moe_w
        else:
            _, ffn1_w, _ = calculate_sizes(self.model, 'gate_up_proj', 1, parallel=tp, fp=fp)
            block_weight += ffn1_w
            _, ffn2_w, _ = calculate_sizes(self.model, 'down_proj', 1, parallel=tp, fp=fp)
            block_weight += ffn2_w
        return block_weight

    def get_kv(self, seq, rank_id=None):
        if self.rank_memory is not None:
            rank = min(self.rank_memory) if rank_id is None else rank_id
            return seq * self.rank_memory[rank]['kv_bytes_per_token']
        # MLA latent KV is replicated across TP ranks; only PP shards layers.
        kv_factor = 1 if self.uses_mla else 2
        divisor = self.pp_size if self.uses_mla else self.pp_size * min(self.tp_size, self.kv_head)
        return kv_factor * self.kv_dim * seq * self.n_layer * self.kv_fp // divisor
    
    # get the total size of current kv cache for the request
    # used when adding prefilled request to decode instance.
    def get_total_kv(self, req):
        # ceil division: (n + block_size - 1) // block_size
        num_blocks = (req.num_computed_tokens + self.block_size - 1) // self.block_size
        return self.get_kv(num_blocks * self.block_size)

    # get size of kv block that should be 'added'. including new init requests
    # also checks evicted request and include its kv cache
    # scheduled_tokens: dict mapping request id to number of tokens scheduled this step
    # 
    # vLLM-style cumulative allocation:
    #   blocks_after = ceil((computed + scheduled) / block_size)
    #   blocks_before = ceil(computed / block_size) if computed > 0 else 0
    #   new_blocks = blocks_after - blocks_before
    def get_block_kv(self, batch_req, batch_len, scheduled_tokens=None):
        if self.system_state is not None:
            return self._system_growth(batch_req[:batch_len], scheduled_tokens)
        # print("[get_block_kv] current batch_req length : {}".format(batch_len))
        block_kv_size = 0
        for i in range(batch_len):
            req = batch_req[i]
            if req.evict or req.is_prefill() or req.hbframe_prefix_restore_pending:
                # (decode + evict) or (prefill) should load all of KV caches
                hit = req.npu_cache_hit if self.enable_prefix_caching else 0
                
                if scheduled_tokens and req.id in scheduled_tokens:
                    tokens_this_step = scheduled_tokens[req.id]
                else:
                    raise RuntimeError("[MemoryModel] [node_id={self.node_id},inst={self.instance_id}]: scheduled_tokens cannot be None")
                
                # vLLM-style cumulative block allocation
                computed_before = req.num_computed_tokens
                
                total_after = computed_before + tokens_this_step
                
                # Calculate blocks needed (cumulative)
                blocks_after = (total_after + self.block_size - 1) // self.block_size
                blocks_before = (computed_before + self.block_size - 1) // self.block_size if computed_before > 0 else 0
                
                
                # Evicted KV has no local blocks: admission must charge the
                # restored context as well as the new tokens.
                if req.evict or req.hbframe_prefix_restore_pending:
                    blocks_before = 0
                new_blocks = max(0, blocks_after - blocks_before)
                block_kv_size += self.get_kv(new_blocks * self.block_size)
                # print("[DEBUG] hit : {} | tokens_this_step : {} | computed_before : {} | total_after : {} | new_blocks : {} | block_kv_size : {}".format(
                #     hit, tokens_this_step, computed_before, total_after, new_blocks, block_kv_size
                # ))
            else:
                # Decode: use num_computed_tokens (or input for backwards compat)
                computed = req.num_computed_tokens
                num_before = (computed + self.block_size - 1) // self.block_size if computed > 0 else 0
                num_after = (computed + 1 + self.block_size - 1) // self.block_size
                if num_after > num_before: # difference of the block is maximum one block
                    block_kv_size += self.get_kv(self.block_size)
        return block_kv_size
    
    def _system_exclusive_objects(self, rank, request_id):
        active = self._system_active_ids - {request_id} - self._system_evicted
        shared = {row["object_uid"] for rid in active
                  for row in self._system_mappings.get((rank, rid), ())}
        return {row["object_uid"] for row in self._system_mappings.get((rank, request_id), ())} - shared

    def can_evict(self, req):
        """A victim must release working capacity in at least one maintained pool."""
        if self.system_state is None:
            return self.get_evict_kv(req) > 0
        for rank, pool in self._system_l1.items():
            pid = pool["pool_id"]
            for uid in self._system_exclusive_objects(rank, req.id):
                if self._system_pool_charges[pid].get(uid, 0) > 0 or (
                        "data_capacity_bytes" in pool and uid in self._system_pool_objects[pid]):
                    return True
        return False

    # Logical payload bytes on the representative rank, separate from victim eligibility.
    def get_evict_kv(self, req):
        if self.system_state is not None:
            owned = self._system_exclusive_objects(self._system_reference_rank, req.id)
            tier = ":L2" if req.evict else ":L1"
            return sum(self._system_objects[uid]["size_bytes"] for uid in owned
                       if any(pool.endswith(tier) for pool in self._system_objects[uid]["ready_pool_ids"]))
        evict_size = 0
        # Use num_computed_tokens if available, fallback to input for backwards compat
        computed = req.num_computed_tokens
        hit = req.npu_cache_hit if self.enable_prefix_caching else 0
        needed = max(0, computed - hit)
        # ceil division: (needed + block_size - 1) // block_size
        num_blocks = (needed + self.block_size - 1) // self.block_size
        evict_size += self.get_kv(num_blocks * self.block_size)
        return evict_size

    def free_weight(self):
        if not self.weight_resident_on_npu:
            self.logger.info(
                "NPU: model weight %.2fMB is not resident; no NPU weight memory to free",
                self.weight / MB_TO_BYTE,
            )
            return
        if self.npu_used - self.resident_weight < 0:
            raise RuntimeError(
                f"[MemoryModel] [node={self.node_id}, inst={self.instance_id}] NPU: tried to free model weight {self.weight / MB_TO_BYTE:.2f}MB "
                f"but only {self.npu_used / MB_TO_BYTE:.2f}MB is used."
            )
        self.logger.info(
            "NPU: used: %.2fMB remove: %.2fMB after: %.2fMB",
            self.npu_used / MB_TO_BYTE,
            self.weight / MB_TO_BYTE,
            (self.npu_used - self.resident_weight) / MB_TO_BYTE,
        )
        self.npu_used -= self.resident_weight

    def is_free(self):
        is_free = self.npu_used == 0 and self.cpu_used == 0
        if not is_free:
            self.logger.error(
                "Memory leak detected: NPU used: %.2fMB, CPU used: %.2fMB",
                self.node_id,
                self.instance_id,
                self.npu_used / MB_TO_BYTE,
                self.cpu_used / MB_TO_BYTE,
            )
        return

    # -------------------- Memory Management --------------------
    
    def allocate(self, size, device):
        if device == Device.NPU:
            if self.npu_used + size > self.npu_mem:
                raise RuntimeError(
                    f"[MemoryModel] [node_id={self.node_id},inst={self.instance_id}] NPU: tried to load {size / MB_TO_BYTE:.2f}MB but only {(self.npu_mem - self.npu_used) / MB_TO_BYTE:.2f}MB is available."
                )
            self.logger.info(
                "NPU: used: %.2fMB load: %.2fMB after: %.2fMB",
                self.npu_used / MB_TO_BYTE,
                size / MB_TO_BYTE,
                (self.npu_used + size) / MB_TO_BYTE,
            )
            self.npu_used += size
        elif device == Device.CPU:
            if self.prefix_storage == Device.CPU and self.enable_prefix_sharing:
                self.second_tier_prefix_cache.allocate(size)
            else:
                if self.cpu_used + size > self.cpu_mem:
                    raise RuntimeError(
                        f"[MemoryModel] [node_id={self.node_id},inst={self.instance_id}] CPU: tried to load {size / MB_TO_BYTE:.2f}MB "
                        f"but only {(self.cpu_mem - self.cpu_used) / MB_TO_BYTE:.2f}MB is available."
                    )
                self.logger.info(
                    "CPU: used: %.2fMB load: %.2fMB after: %.2fMB",
                    self.cpu_used / MB_TO_BYTE,
                    size / MB_TO_BYTE,
                    (self.cpu_used + size) / MB_TO_BYTE,
                )
                self.cpu_used += size
        elif device == Device.CXL:
            self.second_tier_prefix_cache.allocate(size)
        else:
            raise RuntimeError(f"[MemoryModel] [node_id={self.node_id},inst={self.instance_id}] Trying to allocate KV cache in unsupported device {device}")
    
    def free(self, size, device):
        if device == Device.NPU:
            if self.npu_used - size < self.resident_weight:
                raise RuntimeError(
                    f"[MemoryModel] [node_id={self.node_id},inst={self.instance_id}] NPU: tried to free {size / MB_TO_BYTE:.2f}MB but only {(self.npu_used - self.resident_weight) / MB_TO_BYTE:.2f}MB is used."
                )
            self.logger.info(
                "NPU: used: %.2fMB remove: %.2fMB after: %.2fMB",
                self.npu_used / MB_TO_BYTE,
                size / MB_TO_BYTE,
                (self.npu_used - size) / MB_TO_BYTE,
            )
            self.npu_used -= size

        elif device == Device.CPU:
            if self.prefix_storage == Device.CPU and self.enable_prefix_sharing:
                self.second_tier_prefix_cache.free(size)
            else:
                if self.cpu_used - size < 0:
                    raise RuntimeError(
                        f"[MemoryModel] [node_id={self.node_id},inst={self.instance_id}] CPU: tried to free {size / MB_TO_BYTE:.2f}MB "
                        f"but only {self.cpu_used / MB_TO_BYTE:.2f}MB is used."
                    )
                self.logger.info(
                    "CPU: used: %.2fMB remove: %.2fMB after: %.2fMB",
                    self.cpu_used / MB_TO_BYTE,
                    size / MB_TO_BYTE,
                    (self.cpu_used - size) / MB_TO_BYTE,
                )
                self.cpu_used -= size
        elif device == Device.CXL:
            self.second_tier_prefix_cache.free(size)
        else:
            raise RuntimeError(f"[MemoryModel] [node_id={self.node_id},inst={self.instance_id}] Trying to free KV cache in unsupported device {device}")
    
    def is_avail(self, size, device):
        if device == Device.NPU:
            if self.npu_mem - self.npu_used >= size:
                return True
            else:
                return False 
        elif device == Device.CPU:
            if self.enable_prefix_sharing:
                return self.second_tier_prefix_cache.is_avail(size)
            else:
                if self.cpu_mem - self.cpu_used >= size:
                    return True
                else:
                    return False 
        elif device == Device.CXL:
            return self.second_tier_prefix_cache.is_avail(size)
        else:
            raise RuntimeError(f"[MemoryModel] [node_id={self.node_id},inst={self.instance_id}] Trying to check available size of unsupported device {device}")
    
    def need_size(self, size, device):
        if device == Device.NPU:
            needed = (size - (self.npu_mem - self.npu_used))
            if needed > 0:
                return needed
            else:
                return 0
        elif device == Device.CPU:
            if self.enable_prefix_sharing:
                return self.second_tier_prefix_cache.need_size(size)
            else:
                needed = (size - (self.cpu_mem - self.cpu_used))
                if needed > 0:
                    return needed
                else:
                    return 0
        elif device == Device.CXL:
            return self.second_tier_prefix_cache.need_size(size)
        else:
            raise RuntimeError(f"[MemoryModel] [node_id={self.node_id},inst={self.instance_id}] Trying to check available size of unsupported device {device}")

    def avail_size(self, device):
        if self.system_state is not None:
            return self.avail_bytes(device)
        if not self.enable_prefix_caching:
            return 0
        
        if device == Device.NPU:
            return self.npu_prefix_cache.avail_size()
        elif device == Device.CPU or device == Device.CXL:
            return self.second_tier_prefix_cache.avail_size()
        else:
            raise RuntimeError(f"[MemoryModel] [node_id={self.node_id},inst={self.instance_id}] Trying to get available size of prefix cache in unsupported device {device}")
    
    # -------------------- Prefix Cache Management --------------------

    def storage_cache_evicted_req(self, req):
        if self.enable_prefix_caching:
            new_last_node = self.second_tier_prefix_cache.cache_unfinished_req(req, update=False) # do not update hit counts
            # should lock evicted kv cache in cpu
            self.second_tier_prefix_cache.inc_lock_ref(new_last_node)
            req.cpu_last_node = new_last_node
            self.apply_kv_cache_events()

    def evictable_size(self, device):
        if self.system_state is not None and device == Device.NPU:
            _, reclaimable, _ = self._system_batch_space((), {})
            return reclaimable[self._system_l1[self._system_reference_rank]["pool_id"]]
        if not self.enable_prefix_caching:
            return 0
        
        if device == Device.NPU:
            return self.npu_prefix_cache.evictable_size() * self._bytes_per_token
        elif device == Device.CPU or device == Device.CXL:
            return self.second_tier_prefix_cache.evictable_size() * self._bytes_per_token
        else:
            raise RuntimeError(f"[MemoryModel] [node_id={self.node_id},inst={self.instance_id}] Trying to get evictable size of prefix cache in unsupported device {device}")


    def lock_prefix(self, req, device): 
        # Increment lock ref count on req.npu_last_node (set by prefix_match)
        if not self.enable_prefix_caching:
            return
        
        if device == Device.NPU and req.npu_last_node is not None:
            node = req.npu_last_node
            # print(f"[LOCK] req={req.id} lock_prefix node_id={node.id} lock_ref_BEFORE={node.lock_ref}")
            self.npu_prefix_cache.inc_lock_ref(req.npu_last_node)
            # print(f"[LOCK] req={req.id} lock_prefix node_id={node.id} lock_ref_AFTER={node.lock_ref}")
        elif (device == Device.CPU or device == Device.CXL) and req.cpu_last_node is not None:
            self.second_tier_prefix_cache.inc_lock_ref(req.cpu_last_node)
        else:
            raise RuntimeError(f"[MemoryModel] [node_id={self.node_id},inst={self.instance_id}] Trying to lock prefix cache in unsupported device {device}")
    
    def unlock_prefix(self, req, device):
        # Decrement lock ref count on req.npu_last_node (set by prefix_match)
        if not self.enable_prefix_caching:
            return
        
        if device == Device.NPU and req.npu_last_node is not None:
            node = req.npu_last_node
            # print(f"[UNLOCK] req={req.id} unlock_prefix node_id={node.id} lock_ref_BEFORE={node.lock_ref}")
            self.npu_prefix_cache.dec_lock_ref(req.npu_last_node)
            # print(f"[UNLOCK] req={req.id} unlock_prefix node_id={node.id} lock_ref_AFTER={node.lock_ref}")
            req.npu_last_node = None
            req._prefix_locked = False
        elif device == Device.CPU and req.cpu_last_node is not None:
            self.second_tier_prefix_cache.dec_lock_ref(req.cpu_last_node)
            req.cpu_last_node = None
        else:
            raise RuntimeError(f"[MemoryModel] [node_id={self.node_id},inst={self.instance_id}] Trying to unlock prefix cache in unsupported device {device}")
    
    def cache_unfinished_req(self, req, device):
        # Get new_last_node via cache_unfinished_req (replaces last node)
        # Decrement old node's lock ref count, increment new node's lock ref count
        if not self.enable_prefix_caching:
            return
        
        if device == Device.NPU:
            new_last_node = self.npu_prefix_cache.cache_unfinished_req(req)
            
            old_node = req.npu_last_node
            # print(f"[CACHE_UNFINISHED] req={req.id} old_node_id={old_node.id if old_node else None}(lock_ref={old_node.lock_ref if old_node else 'N/A'}) -> new_node_id={new_last_node.id}(lock_ref={new_last_node.lock_ref})")
            if old_node is not None and req._prefix_locked:
                self.npu_prefix_cache.dec_lock_ref(old_node)
            self.npu_prefix_cache.inc_lock_ref(new_last_node)
            # print(f"[CACHE_UNFINISHED] req={req.id} AFTER: old_node_id={old_node.id}(lock_ref={old_node.lock_ref}) new_node_id={new_last_node.id}(lock_ref={new_last_node.lock_ref})")
            req.npu_last_node = new_last_node
            req._prefix_locked = True
            if self.logger.isEnabledFor(logging.DEBUG):
                # print(f"cache_unfinished_req of req {req.id}")
                # print(f"===============NPU PREFIX CAHCE of Instance[{self.instance_id}]=================")
                self.npu_prefix_cache.pretty_print()
        elif device == Device.CPU or device == Device.CXL:
            self.second_tier_prefix_cache.cache_unfinished_req(req)
            if self.logger.isEnabledFor(logging.DEBUG):
                # print(f"cache_unfinished_req of req {req.id}")
                # print(f"===============AFTER INSERT: {self.second_tier_prefix_cache.device} PREFIX CAHCE at pid={os.getpid()} tid={threading.get_ident()} pool_id={id(self.second_tier_prefix_cache)}, size={self.second_tier_prefix_cache.total_size()}=================")
                self.second_tier_prefix_cache.pretty_print()
        else:
            raise RuntimeError(f"[MemoryModel] [node_id={self.node_id},inst={self.instance_id}] Trying to cache prefix cache of unfinished request to unsupported device {device}")
        
        self.apply_kv_cache_events()

    def cache_finished_req(self, req, device):
        if not self.enable_prefix_caching:
            return
        
        if device == Device.NPU:
            self.npu_prefix_cache.cache_finished_req(req)
            # Only dec_lock_ref if the request was locked
            node = req.npu_last_node
            if not req._prefix_locked:
                # Never locked → skip dec
                pass
                # print(f"[CACHE_FINISHED] req={req.id} node_id={node.id if node else None} lock_ref={node.lock_ref if node else 'N/A'} (SKIPPED dec - not locked)")
            else:
                # print(f"[CACHE_FINISHED] req={req.id} node_id={node.id if node else None} lock_ref_BEFORE={node.lock_ref if node else 'N/A'}")
                if node is not None:
                    self.npu_prefix_cache.dec_lock_ref(node)
                    req.npu_last_node = None
                req._prefix_locked = False
            # node = req.npu_last_node
            # print(f"[CACHE_FINISHED] req={req.id} node_id={node.id if node else None} lock_ref_BEFORE={node.lock_ref if node else 'N/A'}")
            # self.npu_prefix_cache.dec_lock_ref(req.npu_last_node)
                # print(f"[CACHE_FINISHED] req={req.id} node_id={node.id if node else None} lock_ref_AFTER={node.lock_ref if node else 'N/A'}")
            # print(f"[CACHE_FINISHED] req={req.id} evictable_size={self.npu_prefix_cache.evictable_size()} protected_size={self.npu_prefix_cache.protected_size()} total_size={self.npu_prefix_cache.total_size()}")
            if self.logger.isEnabledFor(logging.DEBUG):
                print(f"cache_finished_req of req {req.id}")
                print(f"===============NPU PREFIX CACHE of Instance[{self.instance_id}]=================")
                self.npu_prefix_cache.pretty_print()
        elif device == Device.CPU or device == Device.CXL:
            self.second_tier_prefix_cache.cache_finished_req(req)
            if self.logger.isEnabledFor(logging.DEBUG):
                # print(f"cache_finished_req of req {req.id}")
                # print(f"===============AFTER INSERT: {self.second_tier_prefix_cache.device} PREFIX CAHCE at pid={os.getpid()} tid={threading.get_ident()} pool_id={id(self.second_tier_prefix_cache)}, size={self.second_tier_prefix_cache.total_size()}=================")
                self.second_tier_prefix_cache.pretty_print()
        else:
            raise RuntimeError(f"[MemoryModel] [node_id={self.node_id},inst={self.instance_id}] Trying to cache prefix cache of finished request to unsupported device {device}")
        
        self.apply_kv_cache_events()

    def evict_prefix_cache(self, bytes, device):
        if self.system_state is not None and device == Device.NPU:
            # Native batch admission reserves against all pools and protects
            # the selected prefix ranges in reserve_logical_batch().
            return
        if not self.enable_prefix_caching or bytes <= 0:
            return

        if device == Device.NPU:
            cache = self.npu_prefix_cache
        elif device == Device.CPU:
            cache = self.second_tier_prefix_cache
        else:
            raise RuntimeError(f"[MemoryModel] [node_id={self.node_id},inst={self.instance_id}] Trying to evict prefix cache to unsupported device {device}")

        # Each cache instance carries its own bytes-per-token in kv_size:
        # per-rank for NPU, full-cluster for the second-tier pool.
        space_needed = (bytes + cache.kv_size - 1) // cache.kv_size
        cache.evict(space_needed)

        self.apply_kv_cache_events()

    # -------------------- Prefix Cache Helpers --------------------

    def prefix_match(self, req): # req.prefix_cache_hit initialization 
        if not self.enable_prefix_caching:
            return
        declared_hit = max(0, int(getattr(req, "declared_prefix_cache_hit", 0) or 0))
        
        tokens = req.input_hash_ids
        if tokens is None:
            req.storage_cache_hit = max(int(getattr(req, "storage_cache_hit", 0) or 0), declared_hit)
            req.prefix_cache_hit = max(int(getattr(req, "npu_cache_hit", 0) or 0), req.storage_cache_hit)
            return
        old_node = req.npu_last_node
        res = self.npu_prefix_cache.match_prefix(tokens[:req.input])
        req.npu_cache_hit = res.hit_length
        req.npu_last_node = res.last_device_node
        # print(f"[PREFIX_MATCH] req={req.id} old_node_id={old_node.id if old_node else None}(lock_ref={old_node.lock_ref if old_node else 'N/A'}) -> new_node_id={res.last_device_node.id}(lock_ref={res.last_device_node.lock_ref}) hit={res.hit_length} num_computed={req.num_computed_tokens}")

        if self.prefix_storage is not None:
            res_storage = self.second_tier_prefix_cache.match_prefix(tokens[:req.input])
            req.storage_cache_hit = max(res_storage.hit_length, declared_hit)
            req.storage_last_node = res_storage.last_device_node
        else:
            req.storage_cache_hit = declared_hit
            req.storage_last_node = None

        if (self.system_state is not None and not req.has_executed
                and req.declared_prefix_source_request_id is None):
            donor = res
            if self.prefix_storage is not None and res_storage.hit_length > donor.hit_length:
                donor = res_storage
            req.prefix_cache_source_request_id = (
                donor.last_device_node.source_request_id if donor.hit_length else None)
        
        req.prefix_cache_hit = max(req.npu_cache_hit, req.storage_cache_hit)
        # if req.num_computed_tokens < req.prefix_cache_hit:
        #     req.num_computed_tokens = req.prefix_cache_hit
        if req.num_computed_tokens == 0:
            req.num_computed_tokens = req.prefix_cache_hit
            # print(f"Request[{req.id}] prefix cache hit: {req.prefix_cache_hit} tokens (NPU: {req.npu_cache_hit}, {self.prefix_storage}: {req.storage_cache_hit})")
        # for debugging
        
        # print(f"===============NPU PREFIX CAHCE of Instance[{self.instance_id}]=================")
        # self.npu_prefix_cache.pretty_print()
        # print("===============CPU PREFIX CAHCE=================")
        # self.second_tier_prefix_cache.pretty_print()
    
    def erase_prefix_info(self, req):
        if not self.enable_prefix_caching:
            return
        
        req.prefix_cache_hit = 0
        req.npu_cache_hit = 0
        req.storage_cache_hit = 0
        req.npu_last_node = None
        req.storage_last_node = None

    def free_prefix_cache(self):
        if not self.enable_prefix_caching:
            return
        # free evictable prefix cache, if evictable_size != total_size there is locked prefix cache
        self.free(self.npu_prefix_cache.evictable_size() * self._bytes_per_token, Device.NPU)
        if not self.enable_prefix_sharing and self.prefix_storage is not None:
            self.free(self.second_tier_prefix_cache.evictable_size() * self._bytes_per_token * self.num_npus, self.prefix_storage)
    
    # Count load/unload events from prefix cache and update memory usage
    def apply_kv_cache_events(self):
        if self.system_state is not None:
            self.npu_prefix_cache.take_events()
            if self.prefix_storage is not None:
                self.second_tier_prefix_cache.take_events()
            return
        # if not self.enable_prefix_caching:
        #     return
        npu_byte_alloc = 0
        npu_byte_free = 0
        cpu_byte_alloc = 0
        cpu_byte_free = 0
        # self.npu_prefix_cache.take_events() -> [BlockStored, BlockStored, BlockRemoved, ...]
        for ev in self.npu_prefix_cache.take_events():
            # print(f" current event block: {ev}")
            if isinstance(ev, BlockStored):
                tlen = len(ev.token_ids)
                for h in ev.block_hashes:
                    # self._npu_cache_hashtolen[h] = tlen
                    if h in self._npu_cache_hashtolen:
                        self._npu_cache_hashtolen[h][1] += 1
                        # if self._npu_cache_hashtolen[h][1] >= 2:
                        #     print("duplicated hash occurs!! h : {}".format(h))
                    else:
                        self._npu_cache_hashtolen[h] = [tlen, 1]
                npu_byte_alloc += self.get_kv(tlen)
            elif isinstance(ev, BlockRemoved):
                for h in ev.block_hashes:
                    # tlen = self._npu_cache_hashtolen.pop(h, 0)
                    # if tlen == 0:
                    if h in self._npu_cache_hashtolen:
                        tlen = self._npu_cache_hashtolen[h][0]
                        self._npu_cache_hashtolen[h][1] -= 1
                        if self._npu_cache_hashtolen[h][1] <= 0:
                            del self._npu_cache_hashtolen[h]
                        npu_byte_free += self.get_kv(tlen)
                    else:
                        print(f"[HASH_MISS] BlockRemoved hash={h} NOT FOUND in map (map_size={len(self._npu_cache_hashtolen)})")
                        self.logger.warning(f"NPU prefix cache remove unknown block hash {h}")
                    # else:
                    #     print(f"[HASH_HIT] BlockRemoved hash={h} tlen={tlen}")
                    # npu_byte_free += self.get_kv(tlen)
        # free first, then allocate
        if npu_byte_free > 0:
            self.free(npu_byte_free, Device.NPU)
        if npu_byte_alloc > 0:
            self.allocate(npu_byte_alloc, Device.NPU)
        # if npu_byte_free > 0:
        #     self.free(npu_byte_free, Device.NPU)

        # Second-tier (CPU/CXL) prefix cache events.
        if self.prefix_storage is None:
            return

        if self.prefix_storage is Device.CPU and not self.enable_prefix_sharing:
            # Non-shared CPU second_tier: bridge events into the instance's
            # cpu_used counter so allocations are bounded by cpu_mem.
            for ev in self.second_tier_prefix_cache.take_events():
                if isinstance(ev, BlockStored):
                    tlen = len(ev.token_ids)
                    for h in ev.block_hashes:
                        if h in self._cpu_cache_hashtolen:
                            self._cpu_cache_hashtolen[h][1] += 1
                        else:
                            self._cpu_cache_hashtolen[h] = [tlen, 1]
                    cpu_byte_alloc += self.get_kv(tlen) * self.num_npus
                elif isinstance(ev, BlockRemoved):
                    for h in ev.block_hashes:
                        if h in self._cpu_cache_hashtolen:
                            tlen = self._cpu_cache_hashtolen[h][0]
                            self._cpu_cache_hashtolen[h][1] -= 1
                            if self._cpu_cache_hashtolen[h][1] <= 0:
                                del self._cpu_cache_hashtolen[h]
                            cpu_byte_free += self.get_kv(tlen) * self.num_npus
                        else:
                            self.logger.warning(f"CPU prefix cache remove unknown block hash {h}")

            if cpu_byte_free > 0:
                self.free(cpu_byte_free, Device.CPU)
            if cpu_byte_alloc > 0:
                self.allocate(cpu_byte_alloc, Device.CPU)
        else:
            # Shared pool or CXL: the cache itself accounts via
            # total_memory_usage (= kv_stored + total_size * kv_size),
            # so no instance-side counter update is needed. Drain the
            # queue to prevent it from growing unboundedly.
            self.second_tier_prefix_cache.take_events()

    def return_prefix_info(self):
        if not self.enable_prefix_caching:
            return (0, 0, 0, 0)
        if self.prefix_storage is None:
            return (self.npu_prefix_cache.return_prefix_info(), (0, 0))
        return (self.npu_prefix_cache.return_prefix_info(), self.second_tier_prefix_cache.return_prefix_info())

        
def full_cluster_kv_bytes_per_token(model, fp, kv_cache_dtype='auto'):
    """Bytes of KV cache per token aggregated over the full TP cluster.

    Mirrors MemoryModel.get_kv(1) * num_npus but computes directly, avoiding
    the per-rank floor-division roundoff. ``fp`` is the model weight dtype
    in bits (16, 32, ...). ``kv_cache_dtype='fp8'`` forces 1 byte per element
    for the KV cache regardless of weight dtype.
    """
    config = get_config(model)
    attention_dims = _attention_dims(config)
    kv_dim = attention_dims["kv_cache_dim"]
    n_layer = config['num_hidden_layers']
    kv_fp = 1 if kv_cache_dtype == 'fp8' else fp // 8
    kv_factor = 1 if attention_dims["is_mla"] else 2
    return kv_factor * kv_dim * n_layer * kv_fp


# calculate the per-rank input, weight, output size of each layer
def calculate_sizes(model, layer_name, length, kv_len=None, pim=False, parallel=1, fp=2):
    """Calculate input, weight, and output tensor sizes for a given layer.

    Args:
        parallel: parallelism degree for weight/activation sharding.
            For dense layers this is TP; for MoE experts this is EP.
    """
    config = get_config(model)
    n_embd = config['hidden_size']
    n_head = config['num_attention_heads']
    attention_dims = _attention_dims(config)
    head_dim = attention_dims["head_dim"]
    vocab_size = config['vocab_size']
    kv_head = attention_dims["kv_head"]
    q_dim = attention_dims["q_dim"]
    v_dim = attention_dims["v_dim"]
    kv_dim = attention_dims["kv_cache_dim"]
    is_mla = attention_dims["is_mla"]
    ffn_dim = config.get("intermediate_size", config.get("ffn_dim"))  # dense FFN dim
    moe_ffn_dim = config.get("moe_intermediate_size", ffn_dim)  # per-expert FFN dim (may differ from dense)
    # Same both-name fallback as MemoryModel.__init__ — HF / Qwen use
    # ``num_experts`` while Mistral uses ``num_local_experts``.
    num_local_experts = config.get(
        "num_local_experts", config.get("num_experts", 1)
    )
    num_shared_experts = int(config.get("n_shared_experts", 0) or 0)

    p = max(int(parallel), 1)

    # NOTE (vLLM-style assumptions):
    # NOTE (vLLM-style assumptions):
    # - Embedding / LM head: vocab-parallel → split vocab_size across ranks.
    # - Q/K/V: ColumnParallelLinear         → split output dim across ranks.
    # - o_proj: RowParallelLinear           → split input dim across ranks.
    # - LayerNorm weights: replicated (NOT sharded).
    # - MoE experts: parallel = EP degree, each rank holds num_local_experts // p experts.

    # ----------------- Embedding & Norms -----------------
    if layer_name == "embedding":
        input_size = length * fp * 2  # token_ids are int32 or int64
        weight_size = (vocab_size // p) * n_embd * fp
        output_size = length * n_embd * fp

    elif layer_name in ["input_layernorm", "post_layernorm", "final_layernorm", "layernorm"]:
        input_size = length * n_embd * fp
        weight_size = 1 * n_embd * fp  # scale only
        output_size = length * n_embd * fp

    elif layer_name == "qk_norm":
        input_size = length * (q_dim + kv_dim) // p * fp
        weight_size = 2 * head_dim * fp
        output_size = length * (q_dim + kv_dim) // p * fp

    # ----------------- RoPE & Attention Core -----------------
    elif layer_name == "rotary_emb":
        if is_mla:
            rope_dim = attention_dims["qk_rope_head_dim"]
            input_size = ((n_head // p) + 1) * length * rope_dim * fp
        else:
            input_size = ((n_head // p) + (kv_head // p)) * length * head_dim * fp
        weight_size = 0
        output_size = input_size

    elif layer_name == "attention":
        if is_mla:
            effective_len = 1 if pim else length
            effective_kv_len = 1 if pim else kv_len
            input_size = (
                (n_head // p) * effective_len * head_dim * fp +
                effective_kv_len * kv_dim * fp
            )
            weight_size = 0
            output_size = (v_dim // p) * effective_len * fp
        else:
            if not pim:
                input_size = (
                    (n_head // p) * length * head_dim * fp +
                    (kv_head // p) * kv_len * head_dim * fp * 2
                )
                output_size = (n_head // p) * length * head_dim * fp
            else:
                input_size = (
                    (n_head // p) * 1 * head_dim * fp +
                    (kv_head // p) * 1 * head_dim * fp * 2
                )
                output_size = (n_head // p) * 1 * head_dim * fp
            weight_size = 0

    # ----------------- QKV Projection (fused) -----------------
    elif layer_name == "qkv_proj":
        input_size = length * n_embd * fp
        if is_mla:
            q_lora_rank = attention_dims["q_lora_rank"]
            kv_lora_rank = attention_dims["kv_lora_rank"]
            qk_rope_head_dim = attention_dims["qk_rope_head_dim"]
            qk_nope_head_dim = attention_dims["qk_nope_head_dim"]
            v_head_dim = attention_dims["v_head_dim"]
            if q_lora_rank is None:
                q_weight = n_embd * (q_dim // p)
                q_output = q_dim // p
                norm_weight = kv_lora_rank * fp
            else:
                q_lora_rank = int(q_lora_rank)
                q_weight = n_embd * q_lora_rank + q_lora_rank * (q_dim // p)
                q_output = q_dim // p
                norm_weight = (q_lora_rank + kv_lora_rank) * fp
            kv_a_weight = n_embd * (kv_lora_rank + qk_rope_head_dim)
            kv_b_output = (n_head * (qk_nope_head_dim + v_head_dim)) // p
            kv_b_weight = kv_lora_rank * kv_b_output
            weight_size = (q_weight + kv_a_weight + kv_b_weight) * fp + norm_weight
            output_size = length * (q_output + kv_dim + kv_b_output) * fp
        else:
            weight_size = n_embd * ((q_dim + 2 * kv_dim) // p) * fp
            output_size = length * ((q_dim + 2 * kv_dim) // p) * fp

    elif layer_name == "o_proj":
        input_dim = v_dim if is_mla else q_dim
        input_size = length * (input_dim // p) * fp
        weight_size = (input_dim // p) * n_embd * fp
        output_size = length * n_embd * fp

    elif layer_name == "gate_up_proj":
        input_size = length * n_embd * fp
        weight_size = n_embd * 2 * (ffn_dim // p) * fp
        output_size = length * 2 * (ffn_dim // p) * fp

    elif layer_name == "act_fn":
        input_size = length * 2 * (ffn_dim // p) * fp
        weight_size = 0
        output_size = length * (ffn_dim // p) * fp

    elif layer_name == "down_proj":
        input_size = length * (ffn_dim // p) * fp
        weight_size = (ffn_dim // p) * n_embd * fp
        output_size = length * n_embd * fp

    elif layer_name == "sampler":
        input_size = length * (vocab_size // p) * fp
        weight_size = 0
        output_size = length * 4  # int32 token IDs

    elif layer_name == "moe":
        experts_per_rank = num_local_experts // p
        input_size = length * n_embd * fp
        weight_size = (n_embd * num_local_experts * fp  # gate (replicated)
                     + experts_per_rank * 3 * n_embd * moe_ffn_dim * fp)  # local experts
        if num_shared_experts > 0:
            weight_size += num_shared_experts * 3 * n_embd * moe_ffn_dim * fp
        output_size = length * n_embd * fp

    # ----------------- LM Head -----------------
    elif layer_name == "lm_head":
        input_size = length * n_embd * fp
        weight_size = n_embd * (vocab_size // p) * fp
        output_size = length * (vocab_size // p) * fp

    else:
        raise ValueError(f"No matching layer name {layer_name} found for model {model}.")

    return input_size, weight_size, output_size
