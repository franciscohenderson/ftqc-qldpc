"""Neural Belief Propagation GNN decoder on the Tanner graph with INT8 post-training quantization.

LLR and Logit Sign Convention
-----------------------------
* ``TannerGNN.forward(synd)`` returns variable error logits ``logit`` of shape ``(B, n)`` where:
  ``logit = log(P(e_i = 1 | synd) / P(e_i = 0 | synd))``.
  Therefore, ``logit > 0`` indicates that an error on variable ``i`` is probable (``P(e_i = 1) > 0.5``),
  and ``logit < 0`` indicates no error is probable.
* ``FastDecoder`` (and ``FastDecoder.osd_external``) uses the standard channel/posterior LLR
  convention in fixed-point ``int16`` with scale 16:
  ``LLR_i = log(P(e_i = 0 | synd) / P(e_i = 1 | synd)) = -logit_i``,
  quantized as ``llr_int16 = clip(round(-16 * logit), -8000, 8000)``.
  In this convention, a **more likely error** has a **lower (more negative) LLR**, matching the
  ascending LLR ordering used by OSD-CS in ``FastDecoder``.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F


def _extract_tanner_graph(h) -> tuple[int, int, np.ndarray, np.ndarray, np.ndarray]:
    """Extract binary parity-check matrix and sorted Tanner graph edge arrays (check_idx, var_idx)."""
    if sp.issparse(h):
        csr = sp.csr_matrix(h)
        csr.sum_duplicates()
        csr.sort_indices()
        h_bin = (csr.toarray().astype(np.uint8)) & 1
    else:
        h_bin = np.asarray(h, dtype=np.uint8) & 1
    if h_bin.ndim != 2:
        raise ValueError(f"Parity-check matrix h must be 2D, got shape {h_bin.shape}")
    m, n = h_bin.shape
    rows, cols = np.nonzero(h_bin)
    return m, n, h_bin, rows.astype(np.int64), cols.astype(np.int64)


def _build_check_neighborhoods(m: int, edge_check: np.ndarray, edge_var: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Build padded check-to-variable index and mask arrays of shape (m, max_deg) for soft syndrome loss."""
    deg = np.bincount(edge_check, minlength=m) if edge_check.size > 0 else np.zeros(m, dtype=np.int64)
    max_deg = max(int(deg.max(initial=0)), 1)
    nbr_vars = np.zeros((m, max_deg), dtype=np.int64)
    nbr_mask = np.zeros((m, max_deg), dtype=np.float32)
    cursors = np.zeros(m, dtype=np.int64)
    for c, v in zip(edge_check, edge_var):
        pos = cursors[c]
        nbr_vars[c, pos] = v
        nbr_mask[c, pos] = 1.0
        cursors[c] = pos + 1
    return nbr_vars, nbr_mask


class _TannerLayer(nn.Module):
    """Single neural-BP message-passing layer over variable, check, and edge states."""

    def __init__(self, hidden: int):
        super().__init__()
        self.v2c = nn.Linear(2 * hidden, hidden)
        self.check_upd = nn.Linear(2 * hidden, hidden)
        self.c2v = nn.Linear(3 * hidden, hidden)
        self.var_upd = nn.Linear(2 * hidden, hidden)


class TannerGNN(nn.Module):
    """Neural-BP Graph Neural Network over the bipartite Tanner graph of parity-check matrix ``h``.

    Parameters
    ----------
    h : scipy.sparse.spmatrix or np.ndarray
        Binary parity-check matrix of shape ``(m, n)``.
    hidden : int, default=32
        Hidden dimension for edge, variable, and check representations.
    layers : int, default=6
        Number of neural message-passing iterations.
    prior_llr : float, default=3.0
        Default channel prior LLR scalar used as the input feature for variable nodes.
    seed : int, default=0
        Deterministic seed for parameter initialization.
    """

    def __init__(
        self,
        h,
        hidden: int = 32,
        layers: int = 6,
        prior_llr: float = 3.0,
        seed: int = 0,
    ):
        super().__init__()
        if seed is not None:
            torch.manual_seed(int(seed))

        m, n, h_bin, edge_check, edge_var = _extract_tanner_graph(h)
        nbr_vars, nbr_mask = _build_check_neighborhoods(m, edge_check, edge_var)

        self.m = int(m)
        self.n = int(n)
        self.num_edges = int(edge_check.shape[0])
        self.hidden = int(hidden)
        self.num_layers = int(layers)
        self.prior_llr = float(prior_llr)

        self.register_buffer("edge_check", torch.from_numpy(edge_check).long())
        self.register_buffer("edge_var", torch.from_numpy(edge_var).long())
        self.register_buffer("h_bin", torch.from_numpy(h_bin.astype(np.float32)))
        self.register_buffer("check_nbr_vars", torch.from_numpy(nbr_vars).long())
        self.register_buffer("check_nbr_mask", torch.from_numpy(nbr_mask).float())

        self.check_in = nn.Linear(1, self.hidden)
        self.var_in = nn.Linear(1, self.hidden)
        self.edge_in = nn.Linear(2 * self.hidden, self.hidden)

        self.layers = nn.ModuleList([_TannerLayer(self.hidden) for _ in range(self.num_layers)])

        self.readout_mlp = nn.Linear(2 * self.hidden, self.hidden)
        self.readout_out = nn.Linear(self.hidden, 1)

        self.loss_history: list[float] = []

    def named_linear_modules(self) -> list[tuple[str, nn.Linear]]:
        """Return ordered ``(name, module)`` pairs for all ``nn.Linear`` layers in the network."""
        items: list[tuple[str, nn.Linear]] = [
            ("check_in", self.check_in),
            ("var_in", self.var_in),
            ("edge_in", self.edge_in),
        ]
        for idx, layer in enumerate(self.layers):
            items.append((f"L{idx}_v2c", layer.v2c))
            items.append((f"L{idx}_check_upd", layer.check_upd))
            items.append((f"L{idx}_c2v", layer.c2v))
            items.append((f"L{idx}_var_upd", layer.var_upd))
        items.append(("readout_mlp", self.readout_mlp))
        items.append(("readout_out", self.readout_out))
        return items

    def forward(
        self,
        synd: torch.Tensor,
        prior_llr: float | torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute per-variable error logits ``(B, n)`` from syndrome batch ``(B, m)``.

        ``logit > 0`` indicates an error is probable on that variable.
        """
        if not isinstance(synd, torch.Tensor):
            synd = torch.as_tensor(synd, dtype=torch.float32)
        else:
            synd = synd.to(dtype=torch.float32)

        squeeze_batch = False
        if synd.ndim == 1:
            synd = synd.unsqueeze(0)
            squeeze_batch = True
        if synd.shape[1] != self.m:
            raise ValueError(f"Expected syndrome width {self.m}, got {synd.shape[1]}")

        batch_size = synd.shape[0]
        device = synd.device
        dtype = synd.dtype

        # Check feature in {+1, -1}: +1 for satisfied check (synd=0), -1 for violated check (synd=1)
        c_feat = (1.0 - 2.0 * synd).unsqueeze(-1)

        # Variable feature: channel LLR prior
        p_llr = self.prior_llr if prior_llr is None else prior_llr
        if isinstance(p_llr, torch.Tensor):
            v_feat = p_llr.to(device=device, dtype=dtype)
            if v_feat.ndim == 1:
                v_feat = v_feat.unsqueeze(0).expand(batch_size, self.n)
            if v_feat.ndim == 2:
                v_feat = v_feat.unsqueeze(-1)
        else:
            v_feat = torch.full((batch_size, self.n, 1), float(p_llr), dtype=dtype, device=device)

        h_c = F.relu(self.check_in(c_feat))
        h_v = F.relu(self.var_in(v_feat))
        h_e = F.relu(
            self.edge_in(torch.cat([h_c[:, self.edge_check], h_v[:, self.edge_var]], dim=-1))
        )

        agg_v = torch.zeros(batch_size, self.n, self.hidden, dtype=dtype, device=device)

        for layer in self.layers:
            # 1. Variable -> Check message along edges
            v2c_in = torch.cat([h_v[:, self.edge_var], h_e], dim=-1)
            m_v2c = F.relu(layer.v2c(v2c_in))

            # 2. Aggregate variable messages into check nodes via index_add_
            agg_c = torch.zeros(batch_size, self.m, self.hidden, dtype=dtype, device=device)
            agg_c.index_add_(1, self.edge_check, m_v2c)

            # 3. Check state update
            c_upd_in = torch.cat([h_c, agg_c], dim=-1)
            h_c = h_c + F.relu(layer.check_upd(c_upd_in))

            # 4. Check -> Variable message along edges (using extrinsic check sum)
            ext_c = agg_c[:, self.edge_check] - m_v2c
            c2v_in = torch.cat([h_c[:, self.edge_check], ext_c, h_e], dim=-1)
            m_c2v = F.relu(layer.c2v(c2v_in))
            h_e = m_c2v

            # 5. Aggregate check messages into variable nodes via index_add_
            agg_v = torch.zeros(batch_size, self.n, self.hidden, dtype=dtype, device=device)
            agg_v.index_add_(1, self.edge_var, m_c2v)

            # 6. Variable state update
            v_upd_in = torch.cat([h_v, agg_v], dim=-1)
            h_v = h_v + F.relu(layer.var_upd(v_upd_in))

        readout_in = torch.cat([h_v, agg_v], dim=-1)
        readout_h = F.relu(self.readout_mlp(readout_in))
        logits = self.readout_out(readout_h).squeeze(-1)

        return logits.squeeze(0) if squeeze_batch else logits


def _syndrome_parity_penalty(model: TannerGNN, logits: torch.Tensor, synd: torch.Tensor) -> torch.Tensor:
    """Stable soft syndrome parity penalty using product of ``tanh(-logit / 2)`` per check.

    Under posterior error probabilities ``p_v = sigmoid(logit_v)``, the expected parity spin is
    ``E[(-1)^e_v] = 1 - 2 * p_v = tanh(-logit_v / 2)``. For each check ``c``, the product over its
    incident variables should match ``(-1)^{synd_c} = 1 - 2 * synd_c``.
    """
    var_spin = torch.tanh(-0.5 * logits)  # (B, n) in (-1, 1)
    nbr_spins = var_spin[:, model.check_nbr_vars]  # (B, m, max_deg)
    mask = model.check_nbr_mask.unsqueeze(0)  # (1, m, max_deg)
    padded_spins = nbr_spins * mask + (1.0 - mask)
    check_prod = torch.prod(padded_spins, dim=-1)  # (B, m)
    target_spin = 1.0 - 2.0 * synd
    return torch.mean(0.5 * (1.0 - target_spin * check_prod))


def compute_gnn_loss(model: TannerGNN, logits: torch.Tensor, errors: torch.Tensor, synd: torch.Tensor) -> torch.Tensor:
    """Combined BCE loss + 0.1 * soft syndrome parity penalty."""
    bce = F.binary_cross_entropy_with_logits(logits, errors)
    parity_pen = _syndrome_parity_penalty(model, logits, synd)
    return bce + 0.1 * parity_pen


def train_gnn(
    h,
    p_range: tuple[float, float] = (0.01, 0.08),
    steps: int = 2000,
    batch: int = 256,
    lr: float = 2e-3,
    seed: int = 0,
    hidden: int = 32,
    layers: int = 6,
    log_every: int = 0,
) -> TannerGNN:
    """Train ``TannerGNN`` on i.i.d. Bernoulli(p) errors with ``p ~ U(p_range)`` per sample.

    Loss is ``BCE(logits, e) + 0.1 * syndrome_parity_penalty``. Uses AdamW and gradient clipping.
    """
    torch.manual_seed(int(seed))
    rng = np.random.default_rng(int(seed))

    p_low, p_high = float(p_range[0]), float(p_range[1])
    p_mean = float(np.clip(0.5 * (p_low + p_high), 1e-4, 1.0 - 1e-4))
    prior_llr = float(np.log((1.0 - p_mean) / p_mean))

    model = TannerGNN(h, hidden=hidden, layers=layers, prior_llr=prior_llr, seed=seed)
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    h_t = model.h_bin.t().contiguous()  # (n, m)
    n = model.n

    # Fixed evaluation batch for deterministic progress tracking
    eval_p = rng.uniform(p_low, p_high, size=(batch, 1)).astype(np.float32)
    eval_e_np = (rng.random((batch, n), dtype=np.float32) < eval_p).astype(np.float32)
    eval_e = torch.from_numpy(eval_e_np)
    eval_s = torch.remainder(eval_e @ h_t, 2.0)

    with torch.no_grad():
        init_logits = model(eval_s)
        init_loss = float(compute_gnn_loss(model, init_logits, eval_e, eval_s).item())
    model.loss_history = [init_loss]

    for step in range(1, int(steps) + 1):
        p_sample = rng.uniform(p_low, p_high, size=(batch, 1)).astype(np.float32)
        e_np = (rng.random((batch, n), dtype=np.float32) < p_sample).astype(np.float32)
        e_t = torch.from_numpy(e_np)
        s_t = torch.remainder(e_t @ h_t, 2.0)

        optimizer.zero_grad(set_to_none=True)
        logits = model(s_t)
        loss = compute_gnn_loss(model, logits, e_t, s_t)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        loss_val = float(loss.item())
        model.loss_history.append(loss_val)

        if log_every and step % int(log_every) == 0:
            print(f"[train_gnn] step {step}/{steps} loss={loss_val:.4f}")

    model.eval()
    return model


class QuantizedGNN:
    """Post-training INT8 quantized Tanner GNN running pure NumPy inference.

    Weights are stored in symmetric per-output-channel ``int8`` with ``float32`` per-channel scales.
    Activations are quantized dynamically to ``int8`` at every layer using symmetric per-tensor
    scales calibrated from the 99.9th percentile of absolute activations on ``calib_synd``.
    Matrix multiplications execute on ``int8`` values accumulated in ``int32`` via ``np.matmul``.

    Posterior LLR convention:
    ``llr = clip(round(-16 * logit), -8000, 8000)`` in ``int16`` (higher logit => more likely error
    => lower / more negative LLR, matching ``FastDecoder.osd_external``).
    """

    def __init__(
        self,
        m: int,
        n: int,
        hidden: int,
        num_layers: int,
        prior_llr: float,
        edge_check: np.ndarray,
        edge_var: np.ndarray,
        weights: dict[str, np.ndarray],
        w_scales: dict[str, np.ndarray],
        biases: dict[str, np.ndarray],
        act_scales: dict[str, float],
    ):
        self.m = int(m)
        self.n = int(n)
        self.hidden = int(hidden)
        self.num_layers = int(num_layers)
        self.prior_llr = float(prior_llr)
        self.edge_check = np.asarray(edge_check, dtype=np.int64)
        self.edge_var = np.asarray(edge_var, dtype=np.int64)
        self.weights = {k: np.asarray(v, dtype=np.int8) for k, v in weights.items()}
        self.w_scales = {k: np.asarray(v, dtype=np.float32) for k, v in w_scales.items()}
        self.biases = {k: np.asarray(v, dtype=np.float32) for k, v in biases.items()}
        self.act_scales = {k: float(v) for k, v in act_scales.items()}

    @staticmethod
    def _layer_names(num_layers: int) -> list[str]:
        names = ["check_in", "var_in", "edge_in"]
        for idx in range(num_layers):
            names.extend([f"L{idx}_v2c", f"L{idx}_check_upd", f"L{idx}_c2v", f"L{idx}_var_upd"])
        names.extend(["readout_mlp", "readout_out"])
        return names

    def _qlinear(self, x: np.ndarray, name: str, relu: bool = True) -> np.ndarray:
        """Execute quantized Linear layer in pure NumPy (int8 inputs/weights, int32 accumulation)."""
        s_in = self.act_scales[name]
        w_q = self.weights[name]     # (out_features, in_features) int8
        s_w = self.w_scales[name]    # (out_features,) float32
        b = self.biases[name]        # (out_features,) float32

        x_q = np.clip(np.rint(x / s_in), -127.0, 127.0).astype(np.int8)
        acc_i32 = np.matmul(x_q.astype(np.int32), w_q.T.astype(np.int32))
        out = acc_i32.astype(np.float32) * (s_in * s_w) + b
        if relu:
            np.maximum(out, 0.0, out=out)
        return out

    def logits(self, synd: np.ndarray) -> np.ndarray:
        """Compute posterior error logits ``(B, n)`` in ``float32`` via INT8 layer-by-layer inference."""
        s = np.asarray(synd, dtype=np.float32)
        if s.ndim == 1:
            s = s[None, :]
        if s.shape[1] != self.m:
            raise ValueError(f"Expected syndrome width {self.m}, got {s.shape[1]}")

        batch_size = s.shape[0]
        c_feat = (1.0 - 2.0 * s)[:, :, None]  # (B, m, 1)
        v_feat = np.full((batch_size, self.n, 1), self.prior_llr, dtype=np.float32)

        h_c = self._qlinear(c_feat, "check_in", relu=True)
        h_v = self._qlinear(v_feat, "var_in", relu=True)
        edge_init_in = np.concatenate([h_c[:, self.edge_check, :], h_v[:, self.edge_var, :]], axis=-1)
        h_e = self._qlinear(edge_init_in, "edge_in", relu=True)

        agg_v = np.zeros((batch_size, self.n, self.hidden), dtype=np.float32)

        for idx in range(self.num_layers):
            v2c_in = np.concatenate([h_v[:, self.edge_var, :], h_e], axis=-1)
            m_v2c = self._qlinear(v2c_in, f"L{idx}_v2c", relu=True)

            agg_c = np.zeros((batch_size, self.m, self.hidden), dtype=np.float32)
            np.add.at(agg_c, (slice(None), self.edge_check), m_v2c)

            c_upd_in = np.concatenate([h_c, agg_c], axis=-1)
            h_c = h_c + self._qlinear(c_upd_in, f"L{idx}_check_upd", relu=True)

            ext_c = agg_c[:, self.edge_check, :] - m_v2c
            c2v_in = np.concatenate([h_c[:, self.edge_check, :], ext_c, h_e], axis=-1)
            m_c2v = self._qlinear(c2v_in, f"L{idx}_c2v", relu=True)
            h_e = m_c2v

            agg_v = np.zeros((batch_size, self.n, self.hidden), dtype=np.float32)
            np.add.at(agg_v, (slice(None), self.edge_var), m_c2v)

            v_upd_in = np.concatenate([h_v, agg_v], axis=-1)
            h_v = h_v + self._qlinear(v_upd_in, f"L{idx}_var_upd", relu=True)

        readout_in = np.concatenate([h_v, agg_v], axis=-1)
        readout_h = self._qlinear(readout_in, "readout_mlp", relu=True)
        out_logits = self._qlinear(readout_h, "readout_out", relu=False)
        return out_logits.squeeze(-1)

    def llr(self, synd: np.ndarray) -> np.ndarray:
        """Compute posterior LLRs ``(B, n)`` in ``int16`` using ``clip(round(-16 * logit), -8000, 8000)``."""
        s = np.asarray(synd, dtype=np.uint8)
        if s.ndim == 1:
            s = s[None, :]
        raw_logits = self.logits(s)
        q_llr = np.clip(np.rint(-16.0 * raw_logits), -8000.0, 8000.0).astype(np.int16)
        return q_llr

    def save(self, path: str | Path) -> Path:
        """Serialize INT8 weights, scales, biases, and Tanner graph to an ``.npz`` archive."""
        p = Path(path)
        payload: dict[str, np.ndarray] = {
            "meta": np.array([self.m, self.n, self.hidden, self.num_layers], dtype=np.int64),
            "prior_llr": np.array(self.prior_llr, dtype=np.float32),
            "edge_check": self.edge_check.astype(np.int64),
            "edge_var": self.edge_var.astype(np.int64),
        }
        for name in self._layer_names(self.num_layers):
            payload[f"w_{name}"] = self.weights[name].astype(np.int8)
            payload[f"ws_{name}"] = self.w_scales[name].astype(np.float32)
            payload[f"b_{name}"] = self.biases[name].astype(np.float32)
            payload[f"as_{name}"] = np.array(self.act_scales[name], dtype=np.float32)
        with open(p, "wb") as f:
            np.savez(f, **payload)
        return p

    @classmethod
    def load(cls, path: str | Path) -> QuantizedGNN:
        """Load a ``QuantizedGNN`` instance from an ``.npz`` archive produced by ``save``."""
        p = Path(path)
        if not p.exists() and Path(f"{p}.npz").exists():
            p = Path(f"{p}.npz")
        with np.load(p, allow_pickle=False) as data:
            meta = data["meta"]
            m, n, hidden, num_layers = (int(meta[0]), int(meta[1]), int(meta[2]), int(meta[3]))
            prior_llr = float(data["prior_llr"])
            edge_check = data["edge_check"].astype(np.int64)
            edge_var = data["edge_var"].astype(np.int64)
            weights: dict[str, np.ndarray] = {}
            w_scales: dict[str, np.ndarray] = {}
            biases: dict[str, np.ndarray] = {}
            act_scales: dict[str, float] = {}
            for name in cls._layer_names(num_layers):
                weights[name] = data[f"w_{name}"].astype(np.int8)
                w_scales[name] = data[f"ws_{name}"].astype(np.float32)
                biases[name] = data[f"b_{name}"].astype(np.float32)
                act_scales[name] = float(data[f"as_{name}"])
        return cls(
            m=m,
            n=n,
            hidden=hidden,
            num_layers=num_layers,
            prior_llr=prior_llr,
            edge_check=edge_check,
            edge_var=edge_var,
            weights=weights,
            w_scales=w_scales,
            biases=biases,
            act_scales=act_scales,
        )


def quantize_int8(model: TannerGNN, calib_synd: np.ndarray) -> QuantizedGNN:
    """Post-training INT8 quantization of ``model`` calibrated on ``calib_synd``.

    * Linear weights: symmetric per-output-channel ``int8`` in ``[-127, 127]`` with ``float32`` scale.
    * Input activations: symmetric per-tensor scale calibrated at the 99.9th percentile of ``|x|``.
    """
    model.eval()
    calib = np.asarray(calib_synd, dtype=np.float32)
    if calib.ndim == 1:
        calib = calib[None, :]

    captured_inputs: dict[str, list[np.ndarray]] = {}
    hooks = []

    for name, mod in model.named_linear_modules():
        captured_inputs[name] = []

        def _make_hook(key: str):
            def _hook(_module, inputs, _output):
                x = inputs[0].detach().cpu().numpy().astype(np.float32)
                captured_inputs[key].append(x)
            return _hook

        hooks.append(mod.register_forward_hook(_make_hook(name)))

    try:
        with torch.no_grad():
            calib_t = torch.from_numpy(calib)
            model(calib_t)
    finally:
        for h in hooks:
            h.remove()

    weights: dict[str, np.ndarray] = {}
    w_scales: dict[str, np.ndarray] = {}
    biases: dict[str, np.ndarray] = {}
    act_scales: dict[str, float] = {}

    for name, mod in model.named_linear_modules():
        w_fp = mod.weight.detach().cpu().numpy().astype(np.float32)
        b_fp = mod.bias.detach().cpu().numpy().astype(np.float32)

        max_per_channel = np.max(np.abs(w_fp), axis=1)
        scale_w = np.where(max_per_channel > 1e-8, max_per_channel / 127.0, 1.0 / 127.0).astype(np.float32)
        w_q = np.clip(np.rint(w_fp / scale_w[:, None]), -127.0, 127.0).astype(np.int8)

        xs = captured_inputs.get(name, [])
        if xs:
            flat_abs = np.abs(np.concatenate([x.ravel() for x in xs]))
            amax = float(np.percentile(flat_abs, 99.9)) if flat_abs.size > 0 else 1.0
        else:
            amax = 1.0
        scale_a = float(max(amax / 127.0, 1e-8))

        weights[name] = w_q
        w_scales[name] = scale_w
        biases[name] = b_fp
        act_scales[name] = scale_a

    edge_check = model.edge_check.detach().cpu().numpy().astype(np.int64)
    edge_var = model.edge_var.detach().cpu().numpy().astype(np.int64)

    return QuantizedGNN(
        m=model.m,
        n=model.n,
        hidden=model.hidden,
        num_layers=model.num_layers,
        prior_llr=model.prior_llr,
        edge_check=edge_check,
        edge_var=edge_var,
        weights=weights,
        w_scales=w_scales,
        biases=biases,
        act_scales=act_scales,
    )


def fp32_agreement(model: TannerGNN, qmodel: QuantizedGNN, synd: np.ndarray | torch.Tensor) -> float:
    """Return fraction of variables whose hard-decision sign agrees between FP32 and INT8 models.

    In FP32, ``logit > 0`` means error probable; in INT8, ``llr < 0`` (or ``q_logit > 0``) means
    error probable.
    """
    model.eval()
    if isinstance(synd, torch.Tensor):
        s_np = synd.detach().cpu().numpy().astype(np.uint8)
        s_t = synd.detach().to(dtype=torch.float32)
    else:
        s_np = np.asarray(synd, dtype=np.uint8)
        s_t = torch.from_numpy(s_np.astype(np.float32))

    if s_np.ndim == 1:
        s_np = s_np[None, :]
        s_t = s_t.unsqueeze(0)

    with torch.no_grad():
        fp32_logits = model(s_t).detach().cpu().numpy()

    int8_llr = qmodel.llr(s_np)
    fp32_sign = fp32_logits > 0.0
    int8_sign = int8_llr < 0
    return float(np.mean(fp32_sign == int8_sign))
