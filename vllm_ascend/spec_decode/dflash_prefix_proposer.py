"""Ascend v1 correctness implementation for the DFlash prefix selector."""

import torch
from vllm.logger import init_logger
from vllm_ascend.spec_decode.dflash_proposer import AscendDflashProposer

logger = init_logger(__name__)


def is_dflash_prefix_draft(speculative_config):
    return speculative_config.method == "dflash" and "DFlashPrefixDraftModel" in (
        speculative_config.draft_model_config.architectures or []
    )


class AscendDflashPrefixProposer(AscendDflashProposer):
    def __init__(self, vllm_config, device, runner=None):
        super().__init__(vllm_config, device, runner=runner)
        spec = vllm_config.speculative_config
        if spec.draft_sample_method == "probabilistic":
            raise ValueError(
                "Initial DFlashPrefix Ascend adapter supports greedy proposals only. "
                "Set draft_sample_method=greedy; probabilities are point masses."
            )
        cfg = self.draft_model_config.hf_config.dflash_config
        if self.num_speculative_tokens != cfg["prefix_block_size"] - 1:
            raise ValueError("num_speculative_tokens must equal trained block_size - 1")
        if self.dynamic_spec is not None:
            raise ValueError(
                "Disable dynamic_spec for the initial prefix-selector experiment"
            )
        self.prefix_top_k = int(cfg["prefix_top_k"])
        self._prefix_disable_selector = bool(cfg.get("prefix_disable_selector", False))
        self._prefix_walk_backend = cfg.get("prefix_walk_backend", "torch")
        if self._prefix_walk_backend not in {"torch", "triton", "auto"}:
            raise ValueError("prefix_walk_backend must be torch, triton, or auto")
        if (cfg.get("prefix_selector_kind") == "local_prefix_v2"
                and self._prefix_walk_backend != "torch"):
            raise ValueError("local_prefix_v2 requires prefix_walk_backend=torch")
        self._prefix_fused_walk = None
        self._prefix_walk_checked = self._prefix_walk_backend == "torch"
        # This existing v1 dispatch flag calls compute_draft_token_ids before
        # ordinary argmax/reduce-sample paths. Its historical name is DFlash2,
        # but we supply our own selector through the overridden method below.
        self.use_dflash2_selector = True
        self._prefix_anchor_indices = torch.arange(
            self.max_batch_size, device=device, dtype=torch.long
        ) * (1 + self.num_speculative_tokens)

    def _maybe_share_lm_head(self, model):
        if getattr(self.model, "draft_id_to_target_id", None) is not None:
            raise ValueError("DFlashPrefix requires a full-vocabulary draft")
        self.model.has_own_lm_head = False
        super()._maybe_share_lm_head(model)

    def compute_draft_token_ids(self, hidden_states, sampling_metadata=None):
        del sampling_metadata  # greedy draft policy is independent of target sampling
        if self._prefix_disable_selector:
            return self.model.compute_unary_token_ids(hidden_states), None
        steps = self.num_speculative_tokens
        if hidden_states.shape[0] % steps:
            raise ValueError(
                "DFlashPrefix received an incompatible hidden-state layout"
            )
        n = hidden_states.shape[0] // steps
        if n > self.max_batch_size:
            raise ValueError("DFlashPrefix batch exceeds the allocated anchor buffer")
        if n == 0:
            return self.input_ids[:0], None
        hidden = hidden_states.reshape(n, steps, -1)
        candidates, unary = self.model.compute_candidates(hidden_states)
        candidates = candidates.reshape(n, steps, self.prefix_top_k)
        unary = unary.reshape(n, steps, self.prefix_top_k)
        anchors = self.input_ids[self._prefix_anchor_indices[:n]]
        head = self.model.model.prefix_head
        prepare = getattr(self.model, "prepare_prefix_tables", None)
        if prepare is None:
            if getattr(head, "requires_token_embeddings", False):
                raise RuntimeError("Install the matching prefix-v2 vLLM adapter")
            tables = head.prepare_tables(hidden, candidates, unary, anchors)
        else:
            tables = prepare(hidden, candidates, unary, anchors)
        selected = self._walk(head, tables)
        return selected.reshape(-1), None

    def _walk(self, head, tables):
        if not self._prefix_walk_checked:
            # One-time check on real model features, outside the steady-state
            # benchmark. An explicit triton request fails on a bad toolchain;
            # auto logs the failure and keeps the portable torch path.
            try:
                from vllm_ascend.ops.triton.spec_decode.dflash_prefix import (
                    greedy_select_prefix,
                )

                expected = head.greedy_walk(tables)
                actual = greedy_select_prefix(tables)
                if not torch.equal(expected, actual):
                    raise RuntimeError(
                        "Fused prefix walk disagrees with the torch reference"
                    )
                self._prefix_fused_walk = greedy_select_prefix
                self._prefix_walk_checked = True
                logger.info(
                    "DFlashPrefix: fused Triton walk passed first-batch token parity"
                )
                return actual
            except Exception as exc:
                if self._prefix_walk_backend == "triton":
                    raise RuntimeError(
                        "Requested Triton prefix walk failed validation"
                    ) from exc
                logger.warning(
                    "DFlashPrefix: fused walk unavailable (%s); using torch", exc
                )
                self._prefix_walk_checked = True
        if self._prefix_fused_walk is not None:
            return self._prefix_fused_walk(tables)
        return head.greedy_walk(tables)
