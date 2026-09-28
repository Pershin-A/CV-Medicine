import torch
from types import SimpleNamespace
from dxa_project.geometry_ml.models import _empty_safe_roi_forward


def test_empty_negative_rois_have_finite_zero_loss_and_backward():
    heads = SimpleNamespace(training=True)
    feature = torch.ones(1, 8, 4, 4, requires_grad=True)
    result, losses = _empty_safe_roi_forward(
        heads, {'0': feature}, [torch.empty(0, 4)], [(64, 64)],
        [{'boxes': torch.empty(0, 4), 'labels': torch.empty(0, dtype=torch.long)}])
    assert result == [] and heads.empty_negative_batches == 1
    loss = sum(losses.values())
    assert torch.isfinite(loss) and loss.item() == 0
    loss.backward()
    assert torch.equal(feature.grad, torch.zeros_like(feature))
