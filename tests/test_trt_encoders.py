import pytest
import torch

from radio_moge.trt_encoders import MoGeTensorOutputs, RadioTensorOutputs, compare_outputs
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
