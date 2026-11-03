"""Detector error model -> decoding problem (check matrix, observable matrix, priors)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
import stim


@dataclass
class DecodingProblem:
    h: sp.csr_matrix       # (detectors, mechanisms)
    obs: sp.csr_matrix     # (observables, mechanisms)
    priors: np.ndarray     # (mechanisms,) probability each mechanism fires

    @property
    def shape(self):
        return self.h.shape


def dem_to_problem(dem: stim.DetectorErrorModel) -> DecodingProblem:
    """Flatten a DEM (no decomposition required) into sparse matrices.

    Mechanisms with identical (detectors, observables) symptoms are merged with XOR-probability
    combination p = p1 (1 - p2) + p2 (1 - p1) so the decoder sees one column per distinct symptom.
    """
    nd, no = dem.num_detectors, dem.num_observables
    merged: dict[tuple, float] = {}
    for inst in dem.flattened():
        if inst.type != "error":
            continue
        p = inst.args_copy()[0]
        dets, obss = [], []
        for t in inst.targets_copy():
            if t.is_relative_detector_id():
                dets.append(t.val)
            elif t.is_logical_observable_id():
                obss.append(t.val)
        # symptoms are XORs: repeated targets cancel
        key_d = tuple(sorted(x for x in set(dets) if dets.count(x) % 2))
        key_o = tuple(sorted(x for x in set(obss) if obss.count(x) % 2))
        if not key_d:      # undetectable mechanism: no column to decode (all-zero in H)
            continue
        key = (key_d, key_o)
        q = merged.get(key, 0.0)
        merged[key] = q * (1 - p) + p * (1 - q)
    keys = list(merged)
    rows_h, cols_h, rows_o, cols_o = [], [], [], []
    for j, (kd, ko) in enumerate(keys):
        rows_h += kd; cols_h += [j] * len(kd)
        rows_o += ko; cols_o += [j] * len(ko)
    h = sp.csr_matrix((np.ones(len(rows_h), np.uint8), (rows_h, cols_h)), shape=(nd, len(keys)))
    o = sp.csr_matrix((np.ones(len(rows_o), np.uint8), (rows_o, cols_o)), shape=(no, len(keys)))
    return DecodingProblem(h, o, np.array([merged[k] for k in keys]))
