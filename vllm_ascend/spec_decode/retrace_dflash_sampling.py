# SPDX-License-Identifier: Apache-2.0
"""Opt-in DFlash proposal sampling for the Table 2 comparison protocol.

Implements temperature -> top-k -> top-p -> categorical sampling, retaining
the EXACT distribution q used for drawing proposals for rejection sampling.
Only the batch-one, full-vocabulary, synchronous DFlash path is supported.
No acceptance threshold or target probability is changed.
"""
import torch


def proposal_probs(logits, temperature, top_p=1.0, top_k=-1):
    if logits.ndim != 2 or temperature <= 0 or not 0 < top_p <= 1:
        raise ValueError("Expected [K,V] logits, temperature > 0 and 0 < top_p <= 1")
    scores = logits.float() / float(temperature)
    if 0 < top_k < scores.shape[-1]:
        threshold = scores.topk(int(top_k), dim=-1).values[:, -1:]
        scores = scores.masked_fill(scores < threshold, float("-inf"))
    probs = scores.softmax(-1)
    if top_p < 1:
        ordered, indices = probs.sort(dim=-1, descending=True)
        # Include the first token crossing the nucleus threshold.
        ordered = ordered.masked_fill(ordered.cumsum(-1) - ordered >= top_p, 0)
        ordered = ordered / ordered.sum(-1, keepdim=True)
        probs = torch.zeros_like(probs).scatter(-1, indices, ordered)
    return probs


def sample_dflash(logits, metadata, expected_proposals):
    if metadata is None or metadata.all_greedy:
        return logits.argmax(-1), None
    temperature = metadata.temperature
    if temperature is None or temperature.numel() != 1:
        raise ValueError("Matched DFlash sampling requires --max-num-seqs 1")
    if logits.ndim != 2 or logits.shape[0] != expected_proposals:
        raise ValueError("Unexpected proposal packing: require one full-vocabulary DFlash block")
    temp = float(temperature.reshape(-1)[0])
    if temp < 1e-5:
        return logits.argmax(-1), None

    def scalar(field, default):
        value = getattr(metadata, field, None)
        if value is None:
            return default
        if value.numel() != 1:
            raise ValueError(f"Expected one request's {field}")
        return value.reshape(-1)[0].item()

    probs = proposal_probs(logits, temp, scalar("top_p", 1.0), int(scalar("top_k", -1)))
    generators = getattr(metadata, "generators", {}) or {}
    generator = generators.get(0)
    tokens = torch.multinomial(probs, num_samples=1, generator=generator).squeeze(-1)
    # Returning logits, greedy tokens, or truncated vocab probabilities here
    # would invalidate the p/q rejection rule in the downstream verifier.
    return tokens, probs
