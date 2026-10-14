"""Numerical CPU checks; real NPUGraph capture is validated by the A2 protocol."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


ROOT = Path(__file__).resolve().parents[3]
spec = importlib.util.spec_from_file_location("dspark_head_graph_tested", ROOT / "vllm_ascend/spec_decode/dspark_head_graph.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class ToyModel(torch.nn.Module):
    def __init__(self, kind):
        super().__init__()
        self.config = SimpleNamespace(markov_head_type=kind)
        self.embedding = torch.nn.Embedding(19, 4)
        self.projection = torch.nn.Linear(4, 19, bias=False)
        self.gate = torch.nn.Linear(9, 4)
        self.recurrent = torch.nn.Linear(13, 12)
        self.model = SimpleNamespace(markov_head=self.projection)
        self.eval()

    def markov_embed(self, token_ids):
        return self.embedding(token_ids)

    def markov_bias(self, emb):
        return self.projection(emb)

    def markov_step(self, token_ids, hidden_states, state):
        emb = self.embedding(token_ids)
        if self.config.markov_head_type == "gated":
            return self.projection(emb * self.gate(torch.cat([hidden_states, emb], -1)).sigmoid()), emb, None
        if state is None:
            state = torch.zeros_like(emb)
        gate, candidate, out = self.recurrent(torch.cat([state, emb, hidden_states], -1)).chunk(3, -1)
        state = gate.sigmoid()*state + (1-gate.sigmoid())*candidate.tanh()
        return self.projection(out.tanh()), emb, state


def original_loop(model, raw, hidden, anchor):
    # Extract and execute the UNMODIFIED fork loop, independently of new helper.
    package = ROOT.parents[1]
    path = package / "baselines/vllm-ascend/vllm_ascend/spec_decode/llm_base_proposer.py"
    if not path.is_file():
        pytest.skip("Original-loop baseline is shipped with comparison installer")
    import textwrap
    source = path.read_text()
    start = source.index("                draft_token_ids = self._dspark_draft_buffer[:num_blk]")
    end = source.index("                # Dynamic verify-length path, implemented in DynamicSpecScheduler", start)
    k = raw.shape[1]
    fake = SimpleNamespace(model=model, num_speculative_tokens=k,
                           _dspark_draft_buffer=torch.empty((1, k+1), dtype=torch.int64),
                           _dspark_seed_buffer=anchor)
    ns = dict(self=fake, logits=raw.clone(), num_blk=1, torch=torch,
              sample_hidden_states=hidden.reshape(k, -1), use_probabilistic=False)
    exec(textwrap.dedent(source[start:end]), ns)
    return ns["draft_token_ids"], ns["logits"]


@pytest.mark.parametrize("kind", ["vanilla", "gated", "rnn"])
@pytest.mark.parametrize("k", [1, 7, 15])
def test_operation_matches_original_fork(kind, k):
    torch.manual_seed(42)
    model = ToyModel(kind)
    for _ in range(4):
        raw, hidden, anchor = torch.randn(1, k, 19), torch.randn(1, k, 5), torch.tensor([3])
        expected, scores = original_loop(model, raw, hidden, anchor)
        result = torch.empty_like(expected)
        result[:, 0] = anchor
        actual = module.greedy_markov(model, raw, hidden, result)
        module.check_parity(expected, actual, scores, raw)


def test_strict_check_catches_same_tokens_different_scores():
    tokens = torch.tensor([[1, 2]])
    with pytest.raises(RuntimeError, match="PARITY_FAIL"):
        module.check_parity(tokens, tokens.clone(), torch.tensor([1.]), torch.tensor([1.001]))
    with pytest.raises(RuntimeError, match="nonfinite"):
        module.check_parity(tokens, tokens, torch.tensor([float("nan")]), torch.tensor([float("nan")]))


def test_graph_runtime_refuses_cpu():
    rt = module.DSparkHeadGraph("fast", SimpleNamespace(info=lambda *a: None))
    with pytest.raises(ValueError, match="Ascend NPU"):
        rt.select(ToyModel("rnn"), torch.randn(1, 7, 19), torch.randn(1, 7, 5), torch.tensor([1]))
    assert rt.failed


def test_replay_resets_state_and_returns_owned_output(monkeypatch):
    # Simulate graph replay with the captured operations; this tests buffer
    # lifecycle, not the vendor graph implementation or performance.
    rt = module.DSparkHeadGraph("fast", SimpleNamespace(info=lambda *a: None), check_steps=2)
    model = ToyModel("rnn")
    stream = object()
    def fake_capture(model, inputs):
        rt.model, rt.head, rt.head_type = model, model.model.markov_head, "rnn"
        rt.static = tuple(t.clone() for t in inputs)
        rt.input_signature = module.signature(inputs)
        rt.execution = module.execution(inputs[0].device)
        rt.tokens = torch.empty((1, 8), dtype=torch.int64)
        rt.scores = torch.empty_like(inputs[0])
        rt.owner_stream = stream
        rt.backend = SimpleNamespace(current_stream=lambda: stream)
        def replay():
            rt.scores.copy_(rt.static[0])
            rt.tokens[:, 0].copy_(rt.static[2])
            module.greedy_markov(model, rt.scores, rt.static[1], rt.tokens)
        rt.graph = SimpleNamespace(replay=replay)
    monkeypatch.setattr(rt, "_capture", fake_capture)
    first = rt.select(model, torch.randn(1, 7, 19), torch.randn(1, 7, 5), torch.tensor([1]))
    saved = first.clone()
    for anchor in (3, 8, 10):
        raw, hidden, seed = torch.randn(1, 7, 19), torch.randn(1, 7, 5), torch.tensor([anchor])
        tokens = rt.select(model, raw, hidden, seed)
        expected, _ = original_loop(model, raw, hidden, seed)
        assert torch.equal(tokens, expected)
    assert rt.checked == 2  # fast mode stops reference computation after warmup
    assert torch.equal(first, saved)
