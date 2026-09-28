"""Numerical safety checks with real autograd, requiring torch but no accelerator."""

from __future__ import annotations

import math

import pytest

from honing.backends.jev.trainer import backward_step, training_attention
from honing.core import ContractError

torch = pytest.importorskip("torch")


@pytest.mark.unit
@pytest.mark.parametrize("kind", ["nan_loss", "inf_gradient", "nan_gradient"])
def test_invalid_step_preserves_weights_and_optimizer_state(kind: str) -> None:
    parameter = torch.nn.Parameter(torch.tensor(0.0))
    optimizer = torch.optim.AdamW([parameter], lr=0.01)
    if kind == "nan_loss":
        loss = parameter * float("nan")
    elif kind == "inf_gradient":
        loss = parameter.sqrt()  # Finite zero loss, infinite derivative.
    else:
        loss = parameter.sqrt() * 0  # Finite zero loss, NaN derivative.
    with pytest.raises(ContractError, match=r"nonfinite training .*epoch=2, batch=3"):
        backward_step(loss, [parameter], optimizer, "epoch=2, batch=3")
    assert parameter.item() == 0.0
    assert not optimizer.state


@pytest.mark.unit
def test_finite_step_clips_gradient_and_updates_weights() -> None:
    parameter = torch.nn.Parameter(torch.tensor(2.0))
    optimizer = torch.optim.SGD([parameter], lr=0.1)
    norm = backward_step(parameter.square(), [parameter], optimizer, "epoch=1, batch=1")
    assert norm == pytest.approx(4.0)
    assert parameter.item() == pytest.approx(1.9)


@pytest.mark.unit
def test_attention_policy_excludes_cudnn_and_restores_callers_settings() -> None:
    from torch.nn.attention import SDPBackend, sdpa_kernel

    with sdpa_kernel([SDPBackend.CUDNN_ATTENTION]):
        with pytest.raises(ValueError, match="test failure"), training_attention():
            assert not torch.backends.cuda.cudnn_sdp_enabled()
            assert torch.backends.cuda.flash_sdp_enabled()
            assert torch.backends.cuda.mem_efficient_sdp_enabled()
            assert torch.backends.cuda.math_sdp_enabled()
            raise ValueError("test failure")
        assert torch.backends.cuda.cudnn_sdp_enabled()
        assert not torch.backends.cuda.math_sdp_enabled()


@pytest.mark.gpu
@pytest.mark.parametrize("length", [192, 320])
@pytest.mark.parametrize("dropout", [0.0, 0.1])
def test_masked_negative_attention_scores_have_finite_gradients(length: int, dropout: float) -> None:
    # Reduced reproducer from pytorch/pytorch#196678. A finite forward pass
    # alone does not detect this failure: cuDNN can return NaNs only in dQ.
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    torch.manual_seed(0)
    shape = (1, 1, length, 64)
    q = torch.randn(shape, device="cuda", dtype=torch.bfloat16) * 0.1
    k = torch.randn_like(q) * 0.1
    v = torch.randn_like(q)
    q[..., 0], k[..., 0] = 1.0, -120 * math.sqrt(64)
    q.requires_grad_()
    k.requires_grad_()
    v.requires_grad_()
    mask = torch.ones(length, length, device="cuda", dtype=torch.bool)
    with training_attention():
        out = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=dropout)
        out.backward(torch.randn_like(out))
    assert torch.isfinite(out).all()
    for tensor in (q, k, v):
        assert tensor.grad is not None and torch.isfinite(tensor.grad).all()
