"""CPU lifecycle regression; actual NPU replay is checked by check_target_graph.py.

Execute the real adapter class with a stream-sensitive runtime double. This
reproduces startup binding to a different stream without requiring an NPU.
"""
import ast
import logging
import math
import os
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest


class Tensor:
    def __init__(self, shape, label):
        self.shape, self.label = tuple(shape), label

    def reshape(self, *shape):
        if -1 in shape:
            size = math.prod(self.shape) // math.prod(n for n in shape if n != -1)
            shape = tuple(size if n == -1 else n for n in shape)
        return Tensor(shape, self.label)

    def __getitem__(self, index):
        if isinstance(index, slice):
            return Tensor((len(range(self.shape[0])[index]),), self.label)
        return Tensor(index.shape, self.label)

    def __mul__(self, other):
        return self


@pytest.fixture
def case(monkeypatch):
    for key in list(os.environ):
        if key.startswith("DFLASH_PREFIX_"):
            monkeypatch.delenv(key)
    monkeypatch.setenv("DFLASH_PREFIX_INFERENCE", "fast")
    state = SimpleNamespace(stream="startup", created=[], eager=0, depth=0, fail=False)

    class Head:
        def prepare_tables(self, hidden, candidates, unary, anchors, *, embedding_weight):
            assert hidden.shape == (1, 15, 2560)
            assert candidates.shape == unary.shape == (1, 15, 16)
            assert anchors.shape == (1,)
            assert embedding_weight == "target embeddings"
            state.eager += 1
            return candidates

        def greedy_walk(self, tables):
            return Tensor(tables.shape[:2], "selected")

    class Runtime:
        def __init__(self, head, logger):
            self.stream, self.checked = state.stream, 0
            state.created.append(self)

        def select(self, *args, **kwargs):
            if state.stream != self.stream:
                raise RuntimeError("caller stream changed")
            self.checked += 1
            return Tensor((1, 15), "selected")

    for name, symbol in (("compiled_inference", "CompiledPrefixInferenceRuntime"),
                         ("inference_v2", "PrefixInferenceRuntime")):
        module = ModuleType("speculators.models.dflash_prefix." + name)
        setattr(module, symbol, Runtime)
        monkeypatch.setitem(sys.modules, module.__name__, module)

    hidden = Tensor((15, 2560), "hidden")

    class Base:
        def __init__(self, config, device, runner=None):
            self.draft_model_config = config.speculative_config.draft_model_config
            self.num_speculative_tokens, self.max_batch_size, self.dynamic_spec = 15, 1, None
            self.input_ids = Tensor((16,), "anchors")
            self.model = SimpleNamespace(
                model=SimpleNamespace(prefix_head=Head(), embed_tokens=SimpleNamespace(weight="target embeddings")),
                compute_candidates=lambda h: (Tensor((15, 16), "ids"), Tensor((15, 16), "unary")),
                compute_unary_token_ids=lambda h: Tensor((15,), "unary only"),
            )

        def dummy_run(self, *, nested=False):
            assert self._prefix_in_dummy_run
            first = self.compute_draft_token_ids(hidden)
            if nested:
                assert self.dummy_run()[0].label == "selected"
                assert self._prefix_in_dummy_run
            if state.fail:
                raise RuntimeError("startup failed")
            return first

    path = Path(__file__).resolve().parents[3] / "vllm_ascend/spec_decode/dflash_prefix_proposer.py"
    cls = next(n for n in ast.parse(path.read_text()).body
               if isinstance(n, ast.ClassDef) and n.name == "AscendDflashPrefixProposer")
    namespace = {
        "AscendDflashProposer": Base, "os": os, "logger": logging.getLogger("prefix_lifecycle"),
        "torch": SimpleNamespace(arange=lambda n, **kw: Tensor((n,), "indices"), long="long"),
    }
    exec(compile(ast.Module(body=[cls], type_ignores=[]), str(path), "exec"), namespace)

    def make(kind="compiled_prefix_v3"):
        config = SimpleNamespace(speculative_config=SimpleNamespace(
            draft_sample_method="greedy", draft_model_config=SimpleNamespace(hf_config=SimpleNamespace(
                dflash_config={"prefix_block_size": 16, "prefix_top_k": 16, "prefix_selector_kind": kind},
            )),
        ))
        return namespace["AscendDflashPrefixProposer"](config, "fake_npu")
    return make, state, hidden


@pytest.mark.parametrize("kind", ["compiled_prefix_v3", "local_prefix_v2"])
def test_startup_on_other_stream_does_not_capture_or_consume_parity_checks(case, kind):
    make, state, hidden = case
    adapter = make(kind)
    assert adapter.dummy_run()[0].label == "selected"
    assert adapter.dummy_run()[0].shape == (15,)
    assert state.eager == 2 and not state.created
    assert adapter._prefix_inference_runtime is None
    assert not adapter._prefix_in_dummy_run

    state.stream = "serving"
    selected, probabilities = adapter.compute_draft_token_ids(hidden)
    assert selected.label == "selected" and probabilities is None
    runtime = adapter._prefix_inference_runtime
    assert len(state.created) == 1 and runtime.checked == 1
    adapter.compute_draft_token_ids(hidden)
    assert runtime.checked == 2

    # A later administrative dummy run must not replay or reset the real graph.
    state.stream = "startup"
    adapter.dummy_run()
    assert adapter._prefix_inference_runtime is runtime and runtime.checked == 2
    state.stream = "serving"
    adapter.compute_draft_token_ids(hidden)
    assert runtime.checked == 3
    state.stream = "unexpected stream"
    with pytest.raises(RuntimeError, match="caller stream changed"):
        adapter.compute_draft_token_ids(hidden)


def test_nested_dummy_scope_and_exception_restore_real_request_path(case):
    make, state, hidden = case
    adapter = make()
    adapter.dummy_run(nested=True)
    assert not adapter._prefix_in_dummy_run and not state.created
    state.fail = True
    with pytest.raises(RuntimeError, match="startup failed"):
        adapter.dummy_run()
    assert not adapter._prefix_in_dummy_run and not state.created
    state.stream = "serving"
    adapter.compute_draft_token_ids(hidden)
    assert len(state.created) == 1


def test_disabled_selector_still_uses_unary_path(case):
    make, state, hidden = case
    adapter = make()
    adapter._prefix_disable_selector = True
    assert adapter.dummy_run()[0].label == "unary only"
    assert adapter.compute_draft_token_ids(hidden)[0].label == "unary only"
    assert state.eager == 0 and not state.created
