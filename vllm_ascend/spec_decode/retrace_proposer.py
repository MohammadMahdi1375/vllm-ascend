# SPDX-License-Identifier: Apache-2.0
"""Request-scoped ReTrace state on the native DFlash execution path."""

import torch
from vllm.model_executor.models.retrace import ReTraceRequestCache
from vllm_ascend.ascend_config import get_ascend_config
from vllm_ascend.spec_decode.dflash_proposer import AscendDflashProposer


def is_retrace_draft(speculative_config):
    draft = getattr(speculative_config, "draft_model_config", None)
    return draft is not None and "ReTraceDraftModel" in getattr(
        draft.hf_config, "architectures", []
    )


class AscendReTraceProposer(AscendDflashProposer):
    def __init__(self, vllm_config, device, runner=None):
        super().__init__(vllm_config, device, runner)
        config = self.draft_model_config.hf_config
        parallel = vllm_config.parallel_config
        if vllm_config.model_config.hf_config.model_type != "qwen3":
            raise ValueError("ReTrace supports dense Qwen3 targets")
        if (
            not vllm_config.model_config.enforce_eager
            or vllm_config.scheduler_config.async_scheduling
        ):
            raise ValueError(
                "ReTrace requires enforce_eager and synchronous scheduling"
            )
        if vllm_config.quant_config is not None or getattr(
            self.draft_model_config, "quantization", None
        ):
            raise ValueError("Quantized ReTrace is not validated")
        if any(
            getattr(parallel, key, 1) != 1
            for key in (
                "tensor_parallel_size",
                "pipeline_parallel_size",
                "data_parallel_size",
                "decode_context_parallel_size",
                "prefill_context_parallel_size",
            )
        ):
            raise ValueError("ReTrace native inference currently supports one NPU")
        if config.draft_vocab_size != vllm_config.model_config.get_vocab_size():
            raise ValueError("ReTrace requires the full target vocabulary")
        if self.num_speculative_tokens != config.block_size - 1:
            raise ValueError("Use block_size-1 speculative tokens")
        if self.dynamic_spec is not None or get_ascend_config().enable_reduce_sample:
            raise ValueError(
                "ReTrace requires fixed proposal counts and enable_reduce_sample=false"
            )
        if vllm_config.speculative_config.disable_padded_drafter_batch:
            raise ValueError("ReTrace currently requires padded drafter batches")
        self.retrace_enabled = bool(config.retrace_enabled)
        self.num_query_per_req = config.block_size
        self.retrace_cache = ReTraceRequestCache(
            self.num_speculative_tokens, config.hidden_size, storage_dtype=torch.float16
        )

    def retrace_begin_step(
        self, req_ids, hidden_states, positions, metadata, sampled_token_ids
    ):
        if metadata is None:
            counts = [0] * len(req_ids)
            scores, scored_positions, tokens = (
                hidden_states[:0],
                positions[:0],
                positions[:0],
            )
        else:
            counts = metadata.num_draft_tokens
            rows = metadata.logits_indices[metadata.target_logits_indices.long()].long()
            scores = hidden_states[rows]
            scored_positions = positions[rows] + 1
            tokens = metadata.draft_token_ids
        if isinstance(sampled_token_ids, torch.Tensor):
            accepted = (sampled_token_ids >= 0).sum(-1) - 1
        else:
            accepted = torch.tensor(
                [len(tokens) - 1 for tokens in sampled_token_ids],
                device=hidden_states.device,
                dtype=torch.long,
            )
        self.retrace_cache.begin_step(
            req_ids, scores, scored_positions, tokens, counts, accepted
        )

    def retrace_model_kwargs(self, model_positions):
        if getattr(self.model, "retrace_profile_run", False):
            return {"retrace_runtime": True}
        positions = model_positions.reshape(-1, self.num_query_per_req)[:, 1:]
        return {
            "retrace_runtime": True,
            "retrace_memory": self.retrace_cache.inputs(positions, self.dtype),
        }

    def retrace_capture(self, hidden_states, token_indices):
        if getattr(self.model, "retrace_profile_run", False):
            return
        n = len(self.retrace_cache.req_ids)
        indices = token_indices[: n * self.num_speculative_tokens].long()
        hidden = hidden_states[indices].reshape(n, self.num_speculative_tokens, -1)
        positions = self.positions[indices].reshape(n, self.num_speculative_tokens)
        self.retrace_cache.capture(hidden, positions)

    def retrace_finish_step(self, token_ids):
        self.retrace_cache.finish_step(token_ids)

    def dummy_run(self, *args, **kwargs):
        self.model.retrace_profile_run = True
        try:
            return super().dummy_run(*args, **kwargs)
        finally:
            self.model.retrace_profile_run = False
