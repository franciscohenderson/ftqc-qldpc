"""Tests for the TannerGNN neural-BP decoder and INT8 post-training quantization."""
from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")  # optional extra: pip install torch

from algebra.qldpc import hypergraph_product, repetition_check
from decoder.gnn import (
    QuantizedGNN,
    TannerGNN,
    compute_gnn_loss,
    fp32_agreement,
    quantize_int8,
    train_gnn,
)


def _sample_syndromes(h: np.ndarray, batch: int, p: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    h_bin = (np.asarray(h, dtype=np.uint8) & 1)
    errors = (rng.random((batch, h_bin.shape[1])) < p).astype(np.uint8)
    synd = ((errors.astype(np.int64) @ h_bin.T.astype(np.int64)) % 2).astype(np.uint8)
    return errors, synd


def test_gnn_training_quantization_and_serialization(tmp_path):
    code = hypergraph_product(repetition_check(4), repetition_check(4))
    h = code.hz
    m, n = h.shape

    # Evaluation batch for deterministic before/after loss check
    eval_err, eval_synd = _sample_syndromes(h, batch=128, p=0.04, seed=123)
    eval_err_t = torch.from_numpy(eval_err.astype(np.float32))
    eval_synd_t = torch.from_numpy(eval_synd.astype(np.float32))

    init_model = TannerGNN(h, hidden=16, layers=3, seed=42)
    init_model.eval()
    with torch.no_grad():
        init_logits = init_model(eval_synd_t)
        loss_before = float(compute_gnn_loss(init_model, init_logits, eval_err_t, eval_synd_t).item())

    # Train for 300 steps, batch 128, hidden 16, layers 3
    model = train_gnn(
        h,
        p_range=(0.01, 0.08),
        steps=300,
        batch=128,
        lr=2e-3,
        seed=42,
        hidden=16,
        layers=3,
    )

    with torch.no_grad():
        final_logits = model(eval_synd_t)
        loss_after = float(compute_gnn_loss(model, final_logits, eval_err_t, eval_synd_t).item())

    assert final_logits.shape == (128, n)
    assert model.loss_history[-1] < model.loss_history[0]
    assert loss_after < loss_before

    # Post-training INT8 quantization calibrated on syndromes
    _, calib_synd = _sample_syndromes(h, batch=256, p=0.05, seed=999)
    qmodel = quantize_int8(model, calib_synd)

    assert len(qmodel.weights) > 0
    for name, w in qmodel.weights.items():
        assert w.dtype == np.int8, f"Weight {name} has dtype {w.dtype}, expected int8"

    # Posterior LLR inference in pure NumPy
    B = 64
    _, test_synd = _sample_syndromes(h, batch=B, p=0.04, seed=2026)
    llr = qmodel.llr(test_synd)

    assert isinstance(llr, np.ndarray)
    assert llr.dtype == np.int16
    assert llr.shape == (B, n)
    assert int(llr.min()) >= -8000
    assert int(llr.max()) <= 8000

    # Sign agreement between FP32 and INT8 models
    agree = fp32_agreement(model, qmodel, test_synd)
    assert agree >= 0.9, f"Expected fp32_agreement >= 0.9, got {agree:.4f}"

    # Save and load must reproduce the output identically
    save_path = tmp_path / "quantized_tanner_gnn.npz"
    qmodel.save(save_path)
    assert save_path.exists()

    loaded_qmodel = QuantizedGNN.load(save_path)
    for name, w in loaded_qmodel.weights.items():
        assert w.dtype == np.int8
        assert np.array_equal(w, qmodel.weights[name])

    llr_loaded = loaded_qmodel.llr(test_synd)
    assert llr_loaded.dtype == np.int16
    assert llr_loaded.shape == (B, n)
    assert np.array_equal(llr, llr_loaded)
