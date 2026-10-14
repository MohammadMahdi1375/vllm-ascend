"""Opt-in graph of the existing greedy DSpark Markov loop (no model change).

Fixed batch=1, TP=DP=1, full vocabulary, fixed K, no adaptive verification.
The LM head and draft attention remain outside this graph. Capture is deferred
until a real request; dummy/profile calls use the original proposer path.
"""
import torch


def greedy_markov(model, logits, hidden, tokens):
    """Same operation order, dtypes, in-place addition and argmax as the proposer.

    logits and tokens are private writable buffers. tokens[:, 0] is the anchor.
    The recurrent state is reset for EVERY block, including graph replays.
    """
    kind = getattr(model.config, "markov_head_type", "vanilla")
    if kind not in {"vanilla", "gated", "rnn"}:
        raise ValueError(f"Unsupported DSpark Markov head: {kind}")
    state = None
    for idx in range(logits.shape[1]):
        if kind == "vanilla":
            embedding = model.markov_embed(tokens[:, idx])
            bias = model.markov_bias(embedding)
        else:
            bias, _, state = model.markov_step(
                token_ids=tokens[:, idx], hidden_states=hidden[:, idx], state=state)
        logits[:, idx].add_(bias)
        tokens[:, idx + 1].copy_(logits[:, idx].argmax(dim=-1))
    return tokens


def signature(tensors):
    return tuple((tuple(t.shape), tuple(t.stride()), t.dtype, t.device) for t in tensors)


def execution(device):
    enabled = torch.is_autocast_enabled(device.type)
    return (torch.is_inference_mode_enabled(), enabled,
            torch.get_autocast_dtype(device.type) if enabled else None)


def check_parity(reference_tokens, actual_tokens, reference_scores, actual_scores):
    # Validation-only synchronizations. No CPU/device synchronization is used
    # by the steady-state fast path after all initial checks have passed.
    if not bool(torch.isfinite(reference_scores).all() & torch.isfinite(actual_scores).all()):
        raise RuntimeError("DSPARK_HEAD_PARITY_FAIL nonfinite scores")
    if not torch.equal(reference_scores, actual_scores) or not torch.equal(reference_tokens, actual_tokens):
        raise RuntimeError("DSPARK_HEAD_PARITY_FAIL scores or production tokens differ; no fallback")


class DSparkHeadGraph:
    def __init__(self, mode, logger, check_steps=32):
        if mode not in {"fast", "validate"} or check_steps < 1:
            raise ValueError("DSpark graph requires fast/validate mode and positive initial checks")
        self.mode, self.logger, self.check_steps = mode, logger, check_steps
        self.graph = None
        self.checked = 0
        self.failed = False

    def _capture(self, model, inputs):
        logits, hidden, anchor = inputs
        if logits.device.type != "npu":
            raise ValueError("DSpark head graph requires an Ascend NPU")
        if model.training or not torch.is_inference_mode_enabled():
            raise ValueError("DSpark head graph requires eval and inference_mode")
        if getattr(model, "draft_id_to_target_id", None) is not None:
            raise ValueError("DSpark head graph currently requires a full-vocabulary checkpoint")
        if (logits.ndim != 3 or logits.shape[0] != 1 or not 1 <= logits.shape[1] <= 15
                or hidden.ndim != 3 or hidden.shape[:2] != logits.shape[:2]
                or anchor.shape != (1,) or anchor.dtype != torch.int64
                or any(t.device != logits.device for t in inputs)):
            raise ValueError("DSpark head graph requires batch=1, K=1..15, matching hidden states and int64 anchor")
        self.model = model
        self.head = model.model.markov_head
        self.head_type = getattr(model.config, "markov_head_type", "vanilla")
        self.input_signature = signature(inputs)
        self.execution = execution(logits.device)
        self.static = tuple(t.detach().clone(memory_format=torch.preserve_format) for t in inputs)
        if signature(self.static) != self.input_signature:
            raise ValueError("DSpark head input cloning changed layout")
        self.tokens = torch.empty((1, logits.shape[1] + 1), dtype=torch.int64, device=logits.device)
        # The original loop mutates logits. Keep an immutable per-call source
        # and reset the working logits INSIDE the graph on every replay.
        self.scores = torch.empty_like(self.static[0])
        self.backend = torch.npu
        self.owner_stream = self.backend.current_stream()
        stream = self.backend.Stream()
        stream.wait_stream(self.owner_stream)

        def operation():
            self.scores.copy_(self.static[0])
            self.tokens[:, 0].copy_(self.static[2])
            return greedy_markov(model, self.scores, self.static[1], self.tokens)

        with self.backend.stream(stream):
            for _ in range(3):
                operation()
        self.backend.synchronize()
        graph = self.backend.NPUGraph()
        with self.backend.graph(graph, stream=stream):
            operation()
        self.owner_stream.wait_stream(stream)
        self.backend.synchronize()
        self.graph = graph
        self.logger.info("DSPARK_HEAD_GRAPH_CAPTURED batch=1 proposals=%d head=%s full_vocab=True; "
                         "LM head and draft attention remain eager", logits.shape[1], self.head_type)

    @torch.inference_mode()
    def select(self, model, logits, hidden, anchor):
        if self.failed:
            raise RuntimeError("DSpark head graph previously failed; restart serving")
        inputs = (logits, hidden, anchor)
        try:
            if self.graph is None:
                self._capture(model, inputs)
            if (model is not self.model or model.model.markov_head is not self.head or model.training
                    or getattr(model.config, "markov_head_type", "vanilla") != self.head_type
                    or signature(inputs) != self.input_signature or execution(logits.device) != self.execution):
                raise ValueError("DSpark graph model/input/execution changed; restart serving")
            if self.backend.current_stream() != self.owner_stream:
                raise ValueError("DSpark head graph requires one serialized caller stream")
            for destination, source in zip(self.static, inputs):
                destination.copy_(source)
            self.graph.replay()
            # vLLM can retain outputs beyond the next replay: return owned data.
            result = self.tokens.clone()
            if self.mode == "validate" or self.checked < self.check_steps:
                scores = logits.clone()
                tokens = torch.empty_like(self.tokens)
                tokens[:, 0].copy_(anchor)
                expected = greedy_markov(model, scores, hidden, tokens)
                check_parity(expected, result, scores, self.scores)
                self.checked += 1
                if self.checked == self.check_steps:
                    self.logger.info("DSPARK_HEAD_PARITY_PASS checked_blocks=%d max_score_error=0 mode=%s "
                                     "npu_graph=True production_tokens_checked=True; %s",
                                     self.checked, self.mode,
                                     "steady-state fast path enabled" if self.mode == "fast" else "validation continues")
            return result
        except Exception:
            self.failed = True
            raise
