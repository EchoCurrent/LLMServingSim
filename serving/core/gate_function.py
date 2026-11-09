import random
from dataclasses import dataclass
from .logger import get_logger


@dataclass
class RoutingResult:
    """Per-EP-rank routing information for a single MoE layer."""
    local_tokens: list     # [rank] -> token count routed to this rank
    activated_experts: list # [rank] -> number of distinct experts activated on this rank
    source_tokens: list    # [rank] -> token count originating from this rank before dispatch
    expert_token_counts: dict  # expert_id -> routed token count
    source_rank_expert_tokens: list  # [source_rank] -> {expert_id: token count}
    expert_owners: list  # [expert_id] -> EP owner rank


class GateRouter:
    """Simulator-side model of the MoE gate + expert dispatch.

    Policies:
        BALANCED  (default) — closed-form pigeonhole approximation of
                              a trained learned gate with load-balancing
                              auxiliary loss. Deterministic.
        UNIFORM             — alias of BALANCED.
        RR                  — deterministic round-robin per token.
        RAND                — uniform random per token (seedable).
        HASH                — deterministic request/token hash routing.
        HOTSPOT             — deterministic routing with a hot expert pool.
        REQUEST_AFFINE      — deterministic request-local expert affinity.
        CUSTOM              — user-supplied ``_custom_gate_function``.

    ``block_copy``: simulator-side optimization that emits one
    transformer block's trace and replays it ``num_hidden_layers``
    times instead of re-computing the routing every layer. Cuts
    trace-generation time by roughly ``num_hidden_layers`` × on MoE
    models. Safe whenever every layer's routing produces the same
    (local_tokens, activated_experts) pair — which is true for
    BALANCED (deterministic), and a harmless approximation for
    RR / RAND (per-layer variance in activated-count is small once
    the batch is at saturation). Default True for speed; CUSTOM
    policies that legitimately need per-layer variance can set
    ``block_copy=False`` in the constructor.
    """

    _SUPPORTED_POLICIES = (
        "BALANCED", "UNIFORM", "RR", "RAND", "HASH", "HOTSPOT",
        "REQUEST_AFFINE", "CUSTOM",
    )

    def __init__(
        self,
        node_id,
        instance_id,
        num_local_experts,
        num_experts_per_tok=1,
        routing_policy="BALANCED",
        seed=42,
        block_copy=True,
        hot_expert_fraction=0.125,
        hot_token_fraction=0.8,
        hash_layers=0,
    ):
        self.instance_id = instance_id
        self.E = int(num_local_experts)
        self.k = max(1, min(int(num_experts_per_tok), self.E))
        self.routing_policy = routing_policy.upper()
        self.seed = seed
        self.rnd = random.Random(seed) if seed is not None else random
        self.block_copy = bool(block_copy)
        self.hash_layers = int(hash_layers)
        self.hot_expert_fraction = max(0.0, min(1.0, float(hot_expert_fraction)))
        self.hot_token_fraction = max(0.0, min(1.0, float(hot_token_fraction)))

        if self.routing_policy == "RR":
            self.routing_fn = self._rr_routing
        elif self.routing_policy == "RAND":
            self.routing_fn = self._rand_routing
        elif self.routing_policy in {"BALANCED", "UNIFORM"}:
            # ``route_ep`` bypasses ``routing_fn`` for this policy —
            # the per-rank (local_tokens, activated_experts) pair is
            # computed analytically. ``routing_fn`` still points at a
            # valid function for the few non-route_ep call sites.
            self.routing_fn = self._rand_routing
        elif self.routing_policy in {"HASH", "HOTSPOT", "REQUEST_AFFINE"}:
            self.routing_fn = self._hash_routing
        elif self.routing_policy == "CUSTOM":
            self.routing_fn = self._custom_gate_function
        else:
            raise ValueError(
                f"Unknown routing_policy {routing_policy!r}. "
                f"Supported: {', '.join(self._SUPPORTED_POLICIES)}"
            )
        self.logger = get_logger(self.__class__, node_id=node_id, instance_id=instance_id)

    @staticmethod
    def expert_owner(expert_id, ep_size, num_experts):
        """Determine which EP rank owns a given expert. Even distribution across ranks."""
        return min(int(expert_id * ep_size // num_experts), ep_size - 1)

    def _rr_routing(self, token_idx, E, k):
        base = token_idx % E
        return [(base + o) % E for o in range(k)]

    def _rand_routing(self, token_idx, E, k):
        return self.rnd.sample(range(E), k)

    @staticmethod
    def _stable_mix(*values):
        result = 0x9E3779B185EBCA87
        for value in values:
            result ^= int(value) + 0x9E3779B97F4A7C15 + ((result << 6) & 0xFFFFFFFFFFFFFFFF) + (result >> 2)
            result &= 0xFFFFFFFFFFFFFFFF
        return result

    def _hash_routing(self, token_idx, E, k, *, request_id=0, layer_num=0, policy=None):
        seed = self._stable_mix(self.seed or 0, request_id, token_idx, layer_num)
        hot_count = min(E, max(k, int(round(E * self.hot_expert_fraction))))
        policy = self.routing_policy if policy is None else policy
        if policy == "HOTSPOT":
            use_hot = int(seed % 10_000) < int(self.hot_token_fraction * 10_000)
            pool_start = 0 if use_hot else hot_count
            pool_size = hot_count if use_hot else max(k, E - hot_count)
        elif policy == "REQUEST_AFFINE":
            pool_size = hot_count
            pool_start = int(self._stable_mix(request_id, layer_num) % E)
        else:
            pool_start = 0
            pool_size = E
        cursor = int(seed % max(1, pool_size))
        return [
            (pool_start + cursor + offset) % E
            for offset in range(k)
        ]

    def _custom_gate_function(self, token_idx, E, k):
        raise NotImplementedError("Implement custom gate function.")

    def route(self, layer_num, batch_id, total_len):
        """Returns flat token counts per expert (used when EP=1)."""
        counts = [0] * self.E
        for t in range(int(total_len)):
            exps = self._selected_experts(layer_num, t, None)
            for e in exps:
                counts[e] += 1

        self.logger.info(
            "layer=%d policy=%s E=%d k=%d batch=%s tokens=%d assigned=%s",
            layer_num, self.routing_policy, self.E, self.k,
            batch_id, total_len, counts,
        )
        return counts

    def route_ep(self, layer_num, batch_id, total_len, ep_size, token_identities=None):
        """EP-aware routing: returns per-rank token counts and activated experts.

        Tokens are distributed evenly across EP ranks before dispatch
        (matching vLLM's EP execution model). Each token selects k
        experts; the owning rank receives the token for local
        execution. Expert-to-rank assignment uses even partitioning:
        ``expert_id * ep // num_experts``.

        BALANCED uses a deterministic, evenly spread synthetic route.
        Real gate captures can be supplied by a frontend in place of this
        synthetic policy.
        """
        total_len = int(total_len)
        ep_size = max(1, int(ep_size))

        # Distribute source tokens evenly across EP ranks
        base = total_len // ep_size
        remainder = total_len % ep_size
        source_tokens = [base + (1 if r < remainder else 0) for r in range(ep_size)]

        expert_counts, source_expert_counts, routed_local_tokens, routed_activated = (
            self._route_expert_counts(
                layer_num, total_len, ep_size, source_tokens, token_identities,
            )
        )
        local_tokens = routed_local_tokens
        activated_counts = routed_activated

        self.logger.info(
            "layer=%d policy=%s E=%d k=%d ep=%d batch=%s tokens=%d local=%s activated=%s",
            layer_num, self.routing_policy, self.E, self.k, ep_size,
            batch_id, total_len, local_tokens, activated_counts,
        )

        return RoutingResult(
            local_tokens=local_tokens,
            activated_experts=activated_counts,
            source_tokens=source_tokens,
            expert_token_counts=expert_counts,
            source_rank_expert_tokens=source_expert_counts,
            expert_owners=[self.expert_owner(expert_id, ep_size, self.E) for expert_id in range(self.E)],
        )

    @staticmethod
    def _token_identity(token_identities, token_index):
        if token_identities and token_index < len(token_identities):
            request_id, token_position = token_identities[token_index]
            return int(request_id), int(token_position)
        return 0, int(token_index)

    def _selected_experts(self, layer_num, token_index, token_identities):
        request_id, token_position = self._token_identity(token_identities, token_index)
        if layer_num < self.hash_layers:
            return self._hash_routing(token_position,self.E,self.k,request_id=request_id,layer_num=layer_num,policy="HASH")
        if self.routing_policy in {"BALANCED", "UNIFORM"} and layer_num >= self.hash_layers:
            offset = int(self._stable_mix(self.seed or 0, layer_num) % self.E)
            stride = max(1, self.E // self.k)
            return [
                (offset + token_index + item * stride) % self.E
                for item in range(self.k)
            ]
        if self.routing_policy == "RR":
            return self._rr_routing(token_index, self.E, self.k)
        if self.routing_policy in {"HASH", "HOTSPOT", "REQUEST_AFFINE"}:
            return self._hash_routing(
                token_position, self.E, self.k,
                request_id=request_id, layer_num=layer_num,
            )
        return self.routing_fn(token_index, self.E, self.k)

    def _route_expert_counts(self, layer_num, total_len, ep_size, source_tokens, token_identities):
        if self.routing_policy in {"BALANCED", "UNIFORM"} and layer_num >= self.hash_layers:
            return self._route_balanced_expert_counts(
                layer_num,
                total_len,
                ep_size,
                source_tokens,
            )

        owners = [self.expert_owner(expert_id, ep_size, self.E) for expert_id in range(self.E)]
        expert_counts_dense = [0] * self.E
        source_counts_dense = [[0] * self.E for _ in range(ep_size)]
        local_tokens = [0] * ep_size
        token_cursor = 0
        for source_rank, count in enumerate(source_tokens):
            rank_counts = source_counts_dense[source_rank]
            for _ in range(count):
                selected = self._selected_experts(layer_num, token_cursor, token_identities)
                destination_mask = 0
                for expert_id in selected:
                    expert_counts_dense[expert_id] += 1
                    rank_counts[expert_id] += 1
                    destination_mask |= 1 << owners[expert_id]
                while destination_mask:
                    lowest_bit = destination_mask & -destination_mask
                    local_tokens[lowest_bit.bit_length() - 1] += 1
                    destination_mask ^= lowest_bit
                token_cursor += 1
        expert_counts = {
            expert_id: count
            for expert_id, count in enumerate(expert_counts_dense)
            if count
        }
        source_counts = [
            {expert_id: count for expert_id, count in enumerate(rank_counts) if count}
            for rank_counts in source_counts_dense
        ]
        activated_counts = [0] * ep_size
        for expert_id, count in enumerate(expert_counts_dense):
            if count:
                activated_counts[owners[expert_id]] += 1
        return (
            expert_counts,
            source_counts,
            local_tokens,
            activated_counts,
        )

    def _route_balanced_expert_counts(self, layer_num, total_len, ep_size, source_tokens):
        """Exactly count deterministic balanced routing without a token loop.

        BALANCED selects experts from a periodic sequence whose period is the
        expert count.  Count complete periods in bulk and inspect at most one
        partial period per source rank.  This preserves concrete expert IDs
        and source-rank traffic while making long-prefill routing independent
        of the scheduled token count.
        """
        period = max(1, self.E)
        offset = int(self._stable_mix(self.seed or 0, layer_num) % period)
        stride = max(1, period // self.k)
        owners = [self.expert_owner(expert_id, ep_size, period) for expert_id in range(period)]

        expert_pattern = []
        owner_pattern = []
        cycle_expert_counts = [0] * period
        cycle_owner_hits = [0] * ep_size
        for token_index in range(period):
            selected = tuple(
                (offset + token_index + item * stride) % period
                for item in range(self.k)
            )
            selected_owners = frozenset(owners[expert_id] for expert_id in selected)
            expert_pattern.append(selected)
            owner_pattern.append(selected_owners)
            for expert_id in selected:
                cycle_expert_counts[expert_id] += 1
            for owner in selected_owners:
                cycle_owner_hits[owner] += 1

        source_counts = [dict() for _ in range(ep_size)]
        local_tokens = [0] * ep_size
        token_cursor = 0
        for source_rank, count in enumerate(source_tokens):
            count = max(0, int(count))
            full_cycles, remainder = divmod(count, period)
            rank_counts = source_counts[source_rank]
            if full_cycles:
                for expert_id, per_cycle in enumerate(cycle_expert_counts):
                    if per_cycle:
                        rank_counts[expert_id] = full_cycles * per_cycle
                for owner, per_cycle in enumerate(cycle_owner_hits):
                    local_tokens[owner] += full_cycles * per_cycle
            start = token_cursor % period
            for relative_index in range(remainder):
                pattern_index = (start + relative_index) % period
                for expert_id in expert_pattern[pattern_index]:
                    rank_counts[expert_id] = rank_counts.get(expert_id, 0) + 1
                for owner in owner_pattern[pattern_index]:
                    local_tokens[owner] += 1
            token_cursor += count

        expert_counts = {}
        activated_experts = [set() for _ in range(ep_size)]
        for rank_counts in source_counts:
            for expert_id, count in rank_counts.items():
                expert_counts[expert_id] = expert_counts.get(expert_id, 0) + count
                activated_experts[owners[expert_id]].add(expert_id)
        if token_cursor != int(total_len):
            raise ValueError(
                f"source token total {token_cursor} does not match total_len {total_len}"
            )
        return (
            expert_counts,
            source_counts,
            local_tokens,
            [len(experts) for experts in activated_experts],
        )
