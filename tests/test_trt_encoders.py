import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch

from radio_moge.trt_encoders import (
    MoGeTensorOutputs, RadioTensorOutputs, add_readonly_engine_cache, compare_outputs, explicit_output_casts,
    fp32_attention_decompositions, materialize_reduction_dtypes,
    preserve_sensitive_decompositions,
    retain_shape_tensor_storage,
)
from radio_moge.upstream import radio_input
from test_training import tiny_system


@pytest.mark.parametrize('shape', [(32, 48), (48, 32)])
def test_export_boundaries_preserve_all_features_and_downstream_gradients(shape):
    system, _ = tiny_system()
    image = torch.rand(4, 3, *shape)
    indices = (0, 1, 2, 3)
    reference, teacher = system.encode(image, indices)
    radio = RadioTensorOutputs(system.radio, indices)
    moge = MoGeTensorOutputs(system.encoder, reference.spatial.shape[-2:])
    with torch.no_grad():
        actual = radio(radio_input(image, system.radio))
        target = moge(image)
    for a, b in zip(actual, (reference.spatial, reference.summary, *reference.intermediates), strict=True):
        assert torch.equal(a, b)
    assert torch.equal(target[0], teacher.spatial)
    assert torch.equal(target[1], teacher.summary)
    # Frozen outputs must remain usable as inputs saved by trainable downstream ops.
    head = torch.nn.Conv2d(8, 6, 1)
    (head(actual[0]) - target[0]).square().mean().backward()
    assert head.weight.grad is not None and torch.isfinite(head.weight.grad).all()
    assert all(p.grad is None for module in (system.radio, system.encoder) for p in module.parameters())


def test_output_gate_checks_intermediates_magnitude_nonfinite_and_structure():
    outputs = tuple(torch.rand(2, 8, 2, 3) for _ in range(6))
    names = ['spatial', 'summary', 'layer7', 'layer15', 'layer23', 'layer31']
    assert compare_outputs(outputs, outputs, names)['passed']
    changed = (*outputs[:-1], outputs[-1] * 1.1)
    assert not compare_outputs(outputs, changed, names)['passed']
    changed = (*outputs[:-1], outputs[-1] * float('nan'))
    assert not compare_outputs(outputs, changed, names)['passed']
    with pytest.raises(ValueError, match='count'):
        compare_outputs(outputs, outputs[:-1], names)
    with pytest.raises(ValueError, match='shape/dtype'):
        compare_outputs(outputs, (*outputs[:-1], outputs[-1].double()), names)


def test_trainable_encoder_is_rejected():
    system, _ = tiny_system()
    system.radio.requires_grad_(True)
    with pytest.raises(ValueError, match='frozen'):
        RadioTensorOutputs(system.radio, (0, 1, 2, 3))


def test_strong_output_types_use_cast_and_restore_interpreter_on_failure(monkeypatch):
    class Tensor:
        def __init__(self, dtype):
            self.dtype, self.name = dtype, 'feature'

    class Interpreter:
        output_dtypes = [SimpleNamespace(to=lambda *a, **kw: 'float')]
        ctx = SimpleNamespace(net=SimpleNamespace(
            add_cast=lambda value, dtype: SimpleNamespace(get_output=lambda i: Tensor(dtype))))

        def output(self, target, args, kwargs):
            # Simulate a strongly typed network refusing metadata-only overrides.
            assert args[0][0].dtype == 'float'
            return args[0]

    for name, attributes in {
        'tensorrt': dict(ITensor=Tensor, DataType=object),
        'torch_tensorrt': dict(__version__='2.8.0'),
        'torch_tensorrt.dynamo.conversion._TRTInterpreter': dict(TRTInterpreter=Interpreter),
    }.items():
        module = ModuleType(name)
        module.__dict__.update(attributes)
        monkeypatch.setitem(sys.modules, name, module)
    original = Interpreter.output
    source = Tensor('bfloat16')
    with pytest.raises(RuntimeError, match='simulated build failure'):
        with explicit_output_casts() as casts:
            result = Interpreter().output(None, ((source,),), {})
            assert result[0].dtype == 'float' and source.dtype == 'bfloat16'
            assert casts == [dict(name='feature', source='bfloat16', target='float')]
            raise RuntimeError('simulated build failure')
    assert Interpreter.output is original


def test_attention_diagnostic_preserves_values_scale_and_output_type():
    def decomposed(query, key, value, *, scale):
        return (query @ key.transpose(-2, -1) * scale).softmax(-1) @ value

    torch.manual_seed(12)
    q, k, v = [torch.randn(2, 3, 5, 8).bfloat16() for _ in range(3)]
    def other(x):
        return x
    operations = dict(scaled_dot_product_attention=decomposed, unrelated=other)
    changed = fp32_attention_decompositions(operations)
    actual = changed['scaled_dot_product_attention'](q, k, v, scale=.2)
    expected = decomposed(q.float(), k.float(), v.float(), scale=.2).bfloat16()
    assert actual.dtype == q.dtype and torch.equal(actual, expected)
    assert operations['scaled_dot_product_attention'] is decomposed
    assert changed['unrelated'] is other


@pytest.mark.parametrize('dim', [None, 1])
def test_sum_dtype_is_preserved_before_reduction(dim):
    class Sum(torch.nn.Module):
        def forward(self, value):
            return torch.sum(value, dim=dim, dtype=torch.float32)

    value = torch.randn(4, 257).bfloat16()
    exported = torch.export.export(Sum(), (value,)).run_decompositions({}).module()
    expected = exported(value)
    changes = materialize_reduction_dtypes(exported)
    assert len(changes) == 1
    assert changes[0]['input_dtype'] == 'torch.bfloat16'
    assert changes[0]['reduction_dtype'] == 'torch.float32'
    assert torch.equal(exported(value), expected)
    assert all('dtype' not in n.kwargs for n in exported.graph.nodes
               if n.target in (torch.ops.aten.sum.default, torch.ops.aten.sum.dim_IntList))


def test_hybrid_lowering_keeps_native_attention_and_exact_output():
    class Attention(torch.nn.Module):
        def forward(self, q, k, v):
            return torch.nn.functional.scaled_dot_product_attention(q, k, v, scale=.2)

    inputs = tuple(torch.randn(2, 3, 5, 8) for _ in range(3))
    model = Attention()
    program = torch.export.export(model, inputs)
    table = preserve_sensitive_decompositions(torch.export.default_decompositions())
    lowered = program.run_decompositions(table).module()
    assert any(n.target == torch.ops.aten.scaled_dot_product_attention.default
               for n in lowered.graph.nodes)
    assert torch.equal(model(*inputs), lowered(*inputs))


def test_shape_input_buffers_survive_setup_and_preserve_multiple_binding_types():
    import ctypes
    import gc
    import numpy as np

    pointers = {}
    def bind(name, address):
        pointers[name] = address
        return True
    runtime = SimpleNamespace(
        input_names=['data', 'batch', 'size'],
        engine=SimpleNamespace(is_shape_inference_io=lambda name: name != 'data'),
        context=SimpleNamespace(set_tensor_address=bind),
        setup_input_tensors=lambda inputs, *flags: bind('data', inputs[0].data_ptr()),
    )
    inputs = [torch.ones(4, 3), torch.tensor([4], dtype=torch.int64),
              torch.tensor([3, 16, 80], dtype=torch.int32)]
    assert retain_shape_tensor_storage(runtime) == ['batch', 'size']
    for batch in (4, 16, 64, 4):
        inputs[1].fill_(batch)
        runtime.setup_input_tensors(inputs, False, False)
        inputs[1].zero_()  # Shape storage must not alias a caller's CPU tensor.
        gc.collect()
        churn = [np.full(3, -1, dtype=np.int64) for _ in range(100)]
        assert ctypes.c_int64.from_address(pointers['batch']).value == batch
        assert list((ctypes.c_int32 * 3).from_address(pointers['size'])) == [3, 16, 80]
        assert pointers['data'] == inputs[0].data_ptr()
        assert len(runtime._retained_shape_input_storage) == 2
        del churn
    with pytest.raises(RuntimeError, match='already installed'):
        retain_shape_tensor_storage(runtime)
    with pytest.raises(RuntimeError, match='CUDA graphs'):
        runtime.setup_input_tensors(inputs, True, True)
    runtime.context.set_tensor_address = lambda *args: False
    with pytest.raises(RuntimeError, match='Failed to bind'):
        runtime.setup_input_tensors(inputs, False, False)


def test_engine_reuse_preserves_prior_cache_and_writes_only_new_entries(tmp_path):
    prior, current = tmp_path / 'prior', tmp_path / 'current'
    (prior / 'saved').mkdir(parents=True)
    path = prior / 'saved' / 'blob.bin'
    path.write_bytes(b'previous-engine')
    identity = (path.stat().st_ino, path.stat().st_size, path.stat().st_mtime_ns)
    entries = {'current': b'current-engine'}
    def save(key, blob):
        (current / key).mkdir(parents=True)
        (current / key / 'blob.bin').write_bytes(blob)
        entries[key] = blob
    cache = SimpleNamespace(load=lambda key: entries.get(key), save=save)
    stats = add_readonly_engine_cache(cache, prior)
    assert cache.load('current') == b'current-engine'
    assert cache.load('saved') == b'previous-engine'
    assert cache.load('missing') is None
    cache.save('saved', b'new-engine')
    assert cache.load('saved') == b'new-engine'
    assert stats == dict(current_hits=2, reused_hits=1, misses=1)
    assert path.read_bytes() == b'previous-engine'
    assert identity == (path.stat().st_ino, path.stat().st_size, path.stat().st_mtime_ns)
