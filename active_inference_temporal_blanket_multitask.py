#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
active_inference_temporal_blanket_multitask.py

Baseline plus one-at-a-time sensitivity script with multi-target delayed/future evaluation for:

Same generative process:
    E -> S -> I -> A -> E
with the only explicit delay in:
    S(t - τ) -> I(t)

Compared recognition models:
    1. instantaneous
    2. generalized-coordinate comparator
    3. history-extended
    4. oracle-delay
    5. random-history control
    6. shuffled-history control

This script performs, in one execution:
    - simulation
    - resume/checkpointing
    - model comparison
    - regime classification
    - tau / mu / history / generalized-order summaries
    - delay-dominated summaries
    - top condition extraction
    - consolidated final report
    - basic figures

Run examples:
    python3 -u active_inference_temporal_blanket_multitask.py --mode smoke --resume
    python3 -u active_inference_temporal_blanket_multitask.py --mode baseline --seeds 20 --n_jobs 4 --resume

Recommended:
    caffeinate -dimsu python3 -u ~/Downloads/active_inference_temporal_blanket_multitask.py \
      --mode baseline \
      --seeds 20 \
      --n_jobs 4 \
      --resume \
      --outdir ~/Desktop/ai_blanket_balanced_main \
      2>&1 | tee ~/Desktop/ai_blanket_balanced_main.log
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =============================================================================
# Logging
# =============================================================================

T0_GLOBAL = time.time()


def elapsed() -> str:
    return f"{time.time() - T0_GLOBAL:8.1f}s"


def log(msg: str) -> None:
    print(f"[{elapsed()}] {msg}", flush=True)


def safe_mean(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return float("nan")
    return float(np.nanmean(x))


# =============================================================================
# Config
# =============================================================================

MODEL_NAMES = ["instant", "generalized", "history", "oracle", "random", "shuffled"]


@dataclass(frozen=True)
class SimConfig:
    # simulation
    T_units: float = 160.0
    dt: float = 0.02
    burn_frac: float = 0.20
    train_frac: float = 0.50

    # generative process
    # E -> S -> I -> A -> E, delay only in S -> I.
    mu: float = 0.20
    lambda_S: float = 0.60
    gamma: float = 0.40
    lambda_A: float = 0.60
    kappa: float = 1.00
    alpha: float = 1.00
    rho: float = 0.80
    chi: float = 0.10
    sigma_E: float = 0.05
    sigma_S: float = 0.10
    sigma_I: float = 0.10
    sigma_A: float = 0.10
    beta: float = 0.10

    # active inference recognition model
    c_gain: float = 1.00
    kappa_AI: float = 0.20
    chi_AI: float = 0.20
    Pi_s: float = 1.00
    Pi_mu: float = 0.20
    Pi_p: float = 1.00
    eta_mu: float = 5e-4
    eta_a: float = 1e-3
    max_mu_step: float = 0.02
    max_a_step: float = 0.02
    mu_clip: float = 1e4
    a_clip: float = 1e4

    # control
    E_target: float = 0.0
    eta_action: float = 0.01

    # recognition fitting
    ridge_alpha: float = 1e-5
    lambda_dim: float = 1e-4

    # recognition-vector settings
    H: float = 1.20
    r: float = 0.10
    gen_K: int = 2
    deriv_smooth_steps: int = 5

    # response
    response: str = "linear"

    # runtime identity
    seed: int = 0
    condition_id: int = 0


@dataclass(frozen=True)
class GridSpec:
    taus: Tuple[float, ...]
    Hs: Tuple[float, ...]
    rs: Tuple[float, ...]
    gen_Ks: Tuple[int, ...]
    mus: Tuple[float, ...]
    sigma_Es: Tuple[float, ...]
    sigma_pairs: Tuple[Tuple[float, float], ...]
    lambda_Ss: Tuple[float, ...]
    lambda_As: Tuple[float, ...]
    gammas: Tuple[float, ...]
    chis: Tuple[float, ...]
    responses: Tuple[str, ...]
    seeds: Tuple[int, ...]
    T_units: float
    dt: float


def make_grid(mode: str, seeds_override: Optional[int] = None) -> GridSpec:
    """
    Smart design:
    - baseline: main story; only τ × H × model at standard environment.
    - oat_*: one-at-a-time sensitivity analyses around the same baseline.
    - robustness_nonlinear: small nonlinear check; not full factorial.
    """
    def seeds(default: int) -> Tuple[int, ...]:
        return tuple(range(seeds_override if seeds_override is not None else default))

    # Baseline constants
    base_taus = (0.0, 0.2, 0.8, 1.6, 3.2, 6.4)
    base_Hs = (0.0, 0.4, 1.2, 3.2, 6.4)
    base_rs = (0.10,)
    base_gen_Ks = (2,)
    base_mus = (0.20,)
    base_sigma_Es = (0.05,)
    base_sigma_pairs = ((0.10, 0.10),)
    base_lambda_Ss = (0.60,)
    base_lambda_As = (0.60,)
    base_gammas = (0.40,)
    base_chis = (0.10,)
    base_responses = ("linear",)

    if mode == "smoke":
        return GridSpec(
            taus=(0.0, 0.8),
            Hs=(0.0, 1.2),
            rs=base_rs,
            gen_Ks=base_gen_Ks,
            mus=base_mus,
            sigma_Es=base_sigma_Es,
            sigma_pairs=base_sigma_pairs,
            lambda_Ss=base_lambda_Ss,
            lambda_As=base_lambda_As,
            gammas=base_gammas,
            chis=base_chis,
            responses=base_responses,
            seeds=seeds(2),
            T_units=60.0,
            dt=0.02,
        )

    if mode == "baseline":
        return GridSpec(
            taus=base_taus,
            Hs=base_Hs,
            rs=base_rs,
            gen_Ks=base_gen_Ks,
            mus=base_mus,
            sigma_Es=base_sigma_Es,
            sigma_pairs=base_sigma_pairs,
            lambda_Ss=base_lambda_Ss,
            lambda_As=base_lambda_As,
            gammas=base_gammas,
            chis=base_chis,
            responses=base_responses,
            seeds=seeds(20),
            T_units=160.0,
            dt=0.02,
        )

    if mode == "baseline_both_response":
        return GridSpec(
            taus=base_taus,
            Hs=base_Hs,
            rs=base_rs,
            gen_Ks=base_gen_Ks,
            mus=base_mus,
            sigma_Es=base_sigma_Es,
            sigma_pairs=base_sigma_pairs,
            lambda_Ss=base_lambda_Ss,
            lambda_As=base_lambda_As,
            gammas=base_gammas,
            chis=base_chis,
            responses=("linear", "tanh"),
            seeds=seeds(20),
            T_units=160.0,
            dt=0.02,
        )

    if mode == "oat_mu":
        return GridSpec(
            taus=base_taus,
            Hs=base_Hs,
            rs=base_rs,
            gen_Ks=base_gen_Ks,
            mus=(0.05, 0.10, 0.20, 0.40),
            sigma_Es=base_sigma_Es,
            sigma_pairs=base_sigma_pairs,
            lambda_Ss=base_lambda_Ss,
            lambda_As=base_lambda_As,
            gammas=base_gammas,
            chis=base_chis,
            responses=base_responses,
            seeds=seeds(10),
            T_units=160.0,
            dt=0.02,
        )

    if mode == "oat_chi":
        return GridSpec(
            taus=base_taus,
            Hs=base_Hs,
            rs=base_rs,
            gen_Ks=base_gen_Ks,
            mus=base_mus,
            sigma_Es=base_sigma_Es,
            sigma_pairs=base_sigma_pairs,
            lambda_Ss=base_lambda_Ss,
            lambda_As=base_lambda_As,
            gammas=base_gammas,
            chis=(0.00, 0.10, 0.30, 0.60),
            responses=base_responses,
            seeds=seeds(10),
            T_units=160.0,
            dt=0.02,
        )

    if mode == "oat_sigmaE":
        return GridSpec(
            taus=base_taus,
            Hs=base_Hs,
            rs=base_rs,
            gen_Ks=base_gen_Ks,
            mus=base_mus,
            sigma_Es=(0.02, 0.05, 0.10),
            sigma_pairs=base_sigma_pairs,
            lambda_Ss=base_lambda_Ss,
            lambda_As=base_lambda_As,
            gammas=base_gammas,
            chis=base_chis,
            responses=base_responses,
            seeds=seeds(10),
            T_units=160.0,
            dt=0.02,
        )

    if mode == "oat_noise":
        return GridSpec(
            taus=base_taus,
            Hs=base_Hs,
            rs=base_rs,
            gen_Ks=base_gen_Ks,
            mus=base_mus,
            sigma_Es=base_sigma_Es,
            sigma_pairs=((0.05, 0.05), (0.10, 0.10), (0.20, 0.20)),
            lambda_Ss=base_lambda_Ss,
            lambda_As=base_lambda_As,
            gammas=base_gammas,
            chis=base_chis,
            responses=base_responses,
            seeds=seeds(10),
            T_units=160.0,
            dt=0.02,
        )

    if mode == "oat_lambdaS":
        return GridSpec(
            taus=base_taus,
            Hs=base_Hs,
            rs=base_rs,
            gen_Ks=base_gen_Ks,
            mus=base_mus,
            sigma_Es=base_sigma_Es,
            sigma_pairs=base_sigma_pairs,
            lambda_Ss=(0.30, 0.60, 1.20),
            lambda_As=base_lambda_As,
            gammas=base_gammas,
            chis=base_chis,
            responses=base_responses,
            seeds=seeds(10),
            T_units=160.0,
            dt=0.02,
        )

    if mode == "oat_genK":
        return GridSpec(
            taus=base_taus,
            Hs=base_Hs,
            rs=base_rs,
            gen_Ks=(1, 2, 3),
            mus=base_mus,
            sigma_Es=base_sigma_Es,
            sigma_pairs=base_sigma_pairs,
            lambda_Ss=base_lambda_Ss,
            lambda_As=base_lambda_As,
            gammas=base_gammas,
            chis=base_chis,
            responses=base_responses,
            seeds=seeds(10),
            T_units=160.0,
            dt=0.02,
        )

    if mode == "robustness_nonlinear":
        # Small robustness grid: selected delay/history combinations only.
        return GridSpec(
            taus=(0.0, 0.8, 3.2, 6.4),
            Hs=(0.0, 1.2, 6.4),
            rs=base_rs,
            gen_Ks=base_gen_Ks,
            mus=base_mus,
            sigma_Es=base_sigma_Es,
            sigma_pairs=base_sigma_pairs,
            lambda_Ss=base_lambda_Ss,
            lambda_As=base_lambda_As,
            gammas=base_gammas,
            chis=base_chis,
            responses=("linear", "tanh"),
            seeds=seeds(10),
            T_units=160.0,
            dt=0.02,
        )

    raise ValueError(f"Unknown mode: {mode}")


# =============================================================================
# Core math
# =============================================================================

def phi(x: np.ndarray | float, response: str) -> np.ndarray | float:
    if response == "linear":
        return x
    if response == "tanh":
        return np.tanh(x)
    raise ValueError(f"Unknown response: {response}")


def history_offsets(H: float, r: float, dt: float) -> List[int]:
    if H <= 0:
        return []
    vals = []
    m = int(math.floor(H / r))
    for j in range(1, m + 1):
        vals.append(j * r)
    vals.append(H)
    vals = sorted(set(round(v, 10) for v in vals if v > 0))
    return sorted(set(max(1, int(round(v / dt))) for v in vals))


def causal_moving_average(x: np.ndarray, window: int) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    window = max(1, int(window))
    if window <= 1:
        return x.copy()
    out = np.zeros_like(x)
    csum = np.cumsum(np.insert(x, 0, 0.0))
    for i in range(len(x)):
        j0 = max(0, i - window + 1)
        out[i] = (csum[i + 1] - csum[j0]) / (i - j0 + 1)
    return out


def standardize_column(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    sd = float(np.nanstd(x))
    if not np.isfinite(sd) or sd < eps:
        return np.zeros_like(x)
    return (x - float(np.nanmean(x))) / sd


def generalized_coordinate_matrix(
    S: np.ndarray,
    A: np.ndarray,
    K: int,
    dt: float,
    smooth_steps: int = 5,
) -> np.ndarray:
    """
    G_n^K = {S_n, A_n, DS_n, D²S_n, ..., D^K S_n}.
    This is a Friston-style generalized-coordinate comparator, not a full
    implementation of generalized filtering.
    """
    S0 = causal_moving_average(S, smooth_steps)
    cols = [np.asarray(S, dtype=float), np.asarray(A, dtype=float)]
    current = S0.copy()
    for _ in range(1, int(K) + 1):
        d = np.zeros_like(current)
        d[1:] = (current[1:] - current[:-1]) / dt
        d[0] = d[1] if len(d) > 1 else 0.0
        d_smooth = causal_moving_average(d, smooth_steps)
        cols.append(standardize_column(d_smooth))
        current = d_smooth
    return np.column_stack(cols)


def ridge_fit(X: np.ndarray, y: np.ndarray, alpha: float) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    X_aug = np.column_stack([np.ones(len(X)), X])
    p = X_aug.shape[1]
    R = np.eye(p)
    R[0, 0] = 0.0
    A = X_aug.T @ X_aug + alpha * R
    b = X_aug.T @ y
    try:
        return np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        return np.linalg.pinv(A) @ b


def ridge_predict(X: np.ndarray, theta: np.ndarray) -> np.ndarray:
    X_aug = np.column_stack([np.ones(len(X)), X])
    return X_aug @ theta


def ridge_target_mse(
    X: np.ndarray,
    target: np.ndarray,
    split_eff: int,
    cfg: SimConfig,
) -> float:
    """
    Fit a ridge map from the same recognition vector X_n to an arbitrary
    target y_n and return held-out MSE. This is used only for evaluation
    metrics; it does not change the active-inference rollout.
    """
    X = np.asarray(X, dtype=float)
    target = np.asarray(target, dtype=float)
    N = min(len(X), len(target))
    if N < 120:
        return float("nan")
    X = X[:N]
    target = target[:N]
    split_eff = int(min(max(50, split_eff), N - 50))
    if split_eff < 50 or N - split_eff < 50:
        return float("nan")
    theta = ridge_fit(X[:split_eff], target[:split_eff], cfg.ridge_alpha)
    pred = ridge_predict(X[split_eff:], theta)
    return safe_mean((target[split_eff:] - pred) ** 2)


def target_variance_reference(
    target: np.ndarray,
    split_eff: int,
) -> float:
    """
    Baseline no-information reference: held-out variance around training mean.
    Useful for interpreting whether a prediction target is non-trivial.
    """
    target = np.asarray(target, dtype=float)
    N = len(target)
    if N < 120:
        return float("nan")
    split_eff = int(min(max(50, split_eff), N - 50))
    train_mean = float(np.nanmean(target[:split_eff]))
    return safe_mean((target[split_eff:] - train_mean) ** 2)


def future_target_mse(
    X: np.ndarray,
    target_series: np.ndarray,
    max_lag: int,
    split_eff: int,
    horizon_units: float,
    cfg: SimConfig,
) -> tuple[float, float]:
    """
    Predict target_series[n + q] from M_n, where X rows correspond to absolute
    indices n = max_lag, ..., max_lag + len(X)-1.
    Returns (MSE, variance-reference MSE).
    """
    q = int(round(float(horizon_units) / cfg.dt))
    if q <= 0:
        idx = np.arange(max_lag, max_lag + len(X))
    else:
        idx = np.arange(max_lag, max_lag + len(X)) + q

    valid = idx < len(target_series)
    if valid.sum() < 120:
        return float("nan"), float("nan")

    Xv = X[valid]
    yv = np.asarray(target_series, dtype=float)[idx[valid]]
    mse = ridge_target_mse(Xv, yv, split_eff=min(split_eff, len(Xv)-50), cfg=cfg)
    ref = target_variance_reference(yv, split_eff=min(split_eff, len(Xv)-50))
    return mse, ref


def delayed_sensory_target_mse(
    X: np.ndarray,
    S: np.ndarray,
    max_lag: int,
    split_eff: int,
    tau: float,
    cfg: SimConfig,
) -> tuple[float, float]:
    """
    Predict the delay-aligned sensory state S_{n-d} from M_n.
    This exposes the actual delayed variable driving I_n.
    """
    d = int(round(float(tau) / cfg.dt))
    idx = np.arange(max_lag, max_lag + len(X)) - d
    idx = np.maximum(idx, 0)
    yv = np.asarray(S, dtype=float)[idx]
    mse = ridge_target_mse(X, yv, split_eff=split_eff, cfg=cfg)
    ref = target_variance_reference(yv, split_eff=split_eff)
    return mse, ref


def add_multitarget_metrics_to_row(
    row: dict,
    X: np.ndarray,
    E: np.ndarray,
    S: np.ndarray,
    I: np.ndarray,
    A: np.ndarray,
    max_lag: int,
    split_eff: int,
    tau: float,
    cfg: SimConfig,
) -> dict:
    """
    Add evaluation tasks that are closer to delayed/action-relevant settings:
      - current internal-state reconstruction: I_n
      - delayed sensory reconstruction: S_{n-d}
      - future external prediction: E_{n+q}
      - future active-boundary prediction: A_{n+q}
    These metrics use the same recognition vector as each model. They do not
    change the generative process or the active-inference update.
    """
    abs_idx = np.arange(max_lag, max_lag + len(X))
    I_aligned = np.asarray(I, dtype=float)[abs_idx]
    A_aligned = np.asarray(A, dtype=float)[abs_idx]

    mse_I = ridge_target_mse(X, I_aligned, split_eff, cfg)
    ref_I = target_variance_reference(I_aligned, split_eff)

    mse_Sdelay, ref_Sdelay = delayed_sensory_target_mse(X, S, max_lag, split_eff, tau, cfg)

    row["MSE_I_current"] = mse_I
    row["REF_I_current"] = ref_I
    row["RI_I_current"] = (ref_I - mse_I) / ref_I if ref_I and np.isfinite(ref_I) and abs(ref_I) > 1e-12 else np.nan

    row["MSE_S_delay"] = mse_Sdelay
    row["REF_S_delay"] = ref_Sdelay
    row["RI_S_delay"] = (ref_Sdelay - mse_Sdelay) / ref_Sdelay if ref_Sdelay and np.isfinite(ref_Sdelay) and abs(ref_Sdelay) > 1e-12 else np.nan

    # Future horizons. These are fixed to keep the main interpretation stable.
    for h in (0.2, 0.8, 1.6):
        tag = str(h).replace(".", "p")
        mse_Ef, ref_Ef = future_target_mse(X, E, max_lag, split_eff, h, cfg)
        mse_Af, ref_Af = future_target_mse(X, A, max_lag, split_eff, h, cfg)

        row[f"MSE_E_future_{tag}"] = mse_Ef
        row[f"REF_E_future_{tag}"] = ref_Ef
        row[f"RI_E_future_{tag}"] = (ref_Ef - mse_Ef) / ref_Ef if ref_Ef and np.isfinite(ref_Ef) and abs(ref_Ef) > 1e-12 else np.nan

        row[f"MSE_A_future_{tag}"] = mse_Af
        row[f"REF_A_future_{tag}"] = ref_Af
        row[f"RI_A_future_{tag}"] = (ref_Af - mse_Af) / ref_Af if ref_Af and np.isfinite(ref_Af) and abs(ref_Af) > 1e-12 else np.nan

    return row



def gaussian_cmi_scalar(x: np.ndarray, y: np.ndarray, Z: np.ndarray, ridge: float = 1e-8) -> float:
    """
    Gaussian CMI I(x; y | Z) for scalar x,y and multivariate Z.
    """
    x = np.asarray(x, dtype=float).reshape(-1, 1)
    y = np.asarray(y, dtype=float).reshape(-1, 1)
    Z = np.asarray(Z, dtype=float)
    if Z.ndim == 1:
        Z = Z.reshape(-1, 1)

    N = min(len(x), len(y), len(Z))
    if N < 20:
        return float("nan")

    X = np.column_stack([x[:N, 0], y[:N, 0]])
    Z = Z[:N]

    X = X - X.mean(axis=0, keepdims=True)
    Z = Z - Z.mean(axis=0, keepdims=True)

    S_xx = np.cov(X, rowvar=False, bias=False)
    S_zz = np.cov(Z, rowvar=False, bias=False)
    if np.ndim(S_zz) == 0:
        S_zz = np.array([[float(S_zz)]])
    scale = float(np.trace(S_zz) / max(1, S_zz.shape[0])) if np.trace(S_zz) > 0 else 1.0
    S_zz = S_zz + ridge * scale * np.eye(S_zz.shape[0])
    S_xz = (X.T @ Z) / (N - 1)

    try:
        inv_zz = np.linalg.inv(S_zz)
    except np.linalg.LinAlgError:
        inv_zz = np.linalg.pinv(S_zz)

    S_cond = S_xx - S_xz @ inv_zz @ S_xz.T
    v1 = max(float(S_cond[0, 0]), 1e-12)
    v2 = max(float(S_cond[1, 1]), 1e-12)
    det = max(float(np.linalg.det(S_cond)), 1e-24)
    val = 0.5 * math.log((v1 * v2) / det)
    if not np.isfinite(val):
        return float("nan")
    return max(0.0, float(val))


# =============================================================================
# Boundary vectors
# =============================================================================

def build_boundary_matrix(
    S: np.ndarray,
    A: np.ndarray,
    model: str,
    H: float,
    r: float,
    tau: float,
    dt: float,
    rng: np.random.Generator,
    circular_shift: Optional[int] = None,
    gen_K: int = 2,
    deriv_smooth_steps: int = 5,
) -> Tuple[np.ndarray, int]:
    d = int(round(tau / dt))
    H_lags = history_offsets(H, r, dt)
    n = len(S)

    if model == "generalized":
        return generalized_coordinate_matrix(S, A, gen_K, dt, deriv_smooth_steps), 0

    if model == "instant":
        lags = []
        source = S
    elif model == "history":
        lags = H_lags
        source = S
    elif model == "oracle":
        lags = [] if d <= 0 else [d]
        source = S
    elif model == "random":
        count = len(H_lags)
        max_admissible = max(max(H_lags) if H_lags else 1, d + 1, 1)
        pool = [z for z in range(1, max_admissible + count + 10) if z != d]
        lags = sorted(rng.choice(pool, size=count, replace=False).tolist()) if count > 0 else []
        source = S
    elif model == "shuffled":
        lags = H_lags
        shift = circular_shift
        if shift is None:
            shift = int(rng.integers(low=max(1, n // 10), high=max(2, n - 1)))
        source = np.roll(S, shift)
    else:
        raise ValueError(f"Unknown model: {model}")

    max_lag = max(lags) if lags else 0
    rows = []
    for i in range(max_lag, n):
        row = [S[i], A[i]]
        for lag in lags:
            row.append(source[i - lag])
        rows.append(row)
    return np.asarray(rows, dtype=float), max_lag


# =============================================================================
# Simulation and active inference
# =============================================================================

def simulate_trajectory(cfg: SimConfig, tau: float, rng: np.random.Generator) -> Dict[str, np.ndarray]:
    """
    Same generative process for all models:
      E -> S -> I -> A -> E
    with delay only in S -> I.
    """
    N = int(round(cfg.T_units / cfg.dt))
    E = np.zeros(N, dtype=float)
    S = np.zeros(N, dtype=float)
    I = np.zeros(N, dtype=float)
    A = np.zeros(N, dtype=float)

    # Primary fixed-trajectory analysis: no model-specific online action is used
    # to generate the shared environment trajectory.
    a_env = np.zeros(N, dtype=float)

    d = int(round(tau / cfg.dt))
    sqrt_dt = math.sqrt(cfg.dt)

    for n in range(N - 1):
        S_delay = S[n - d] if n - d >= 0 else S[0]
        eps = rng.standard_normal(4)

        E[n + 1] = E[n] + (-cfg.mu * E[n] + cfg.chi * np.tanh(A[n])) * cfg.dt + cfg.sigma_E * sqrt_dt * eps[0]
        S[n + 1] = S[n] + (-cfg.lambda_S * S[n] + cfg.kappa * E[n]) * cfg.dt + cfg.sigma_S * sqrt_dt * eps[1]
        I[n + 1] = I[n] + (-cfg.gamma * I[n] + cfg.alpha * phi(S_delay, cfg.response)) * cfg.dt + cfg.sigma_I * sqrt_dt * eps[2]
        A[n + 1] = A[n] + (-cfg.lambda_A * A[n] + cfg.rho * phi(I[n], cfg.response) + cfg.beta * a_env[n]) * cfg.dt + cfg.sigma_A * sqrt_dt * eps[3]

        if not np.isfinite(E[n + 1] + S[n + 1] + I[n + 1] + A[n + 1]):
            E[n + 1:] = np.nan
            S[n + 1:] = np.nan
            I[n + 1:] = np.nan
            A[n + 1:] = np.nan
            break

        if max(abs(E[n + 1]), abs(S[n + 1]), abs(I[n + 1]), abs(A[n + 1])) > 1e8:
            E[n + 1:] = np.nan
            S[n + 1:] = np.nan
            I[n + 1:] = np.nan
            A[n + 1:] = np.nan
            break

    return {"E": E, "S": S, "I": I, "A": A}


def active_inference_rollout(
    E_test: np.ndarray,
    S_test: np.ndarray,
    theta: np.ndarray,
    X_test: np.ndarray,
    cfg: SimConfig,
) -> Dict[str, np.ndarray]:
    n = len(X_test)
    mu_prior = ridge_predict(X_test, theta)
    mu = np.zeros(n, dtype=float)
    a = np.zeros(n, dtype=float)
    F = np.zeros(n, dtype=float)

    if n > 0:
        mu[0] = mu_prior[0]

    for t in range(1, n):
        mu_t = mu[t - 1]
        mu_prev = mu[t - 2] if t >= 2 else mu[t - 1]
        a_t = a[t - 1]

        Dmu = (mu_t - mu_prev) / cfg.dt
        s_hat = cfg.c_gain * mu_t
        eps_s = S_test[t - 1] - s_hat
        f_ai = -cfg.kappa_AI * mu_t + cfg.chi_AI * a_t
        eps_mu = Dmu - f_ai
        eps_p = mu_t - mu_prior[t - 1]

        F[t - 1] = (
            0.5 * cfg.Pi_s * eps_s**2
            + 0.5 * cfg.Pi_mu * eps_mu**2
            + 0.5 * cfg.Pi_p * eps_p**2
        )

        dF_dmu = (
            cfg.Pi_s * eps_s * (-cfg.c_gain)
            + cfg.Pi_mu * eps_mu * (1.0 / cfg.dt + cfg.kappa_AI)
            + cfg.Pi_p * eps_p
        )
        dF_da = cfg.Pi_mu * eps_mu * (-cfg.chi_AI)

        dmu_step = float(np.clip(-cfg.eta_mu * dF_dmu, -cfg.max_mu_step, cfg.max_mu_step))
        da_step = float(np.clip(-cfg.eta_a * dF_da, -cfg.max_a_step, cfg.max_a_step))

        mu[t] = float(np.clip(mu_t + dmu_step, -cfg.mu_clip, cfg.mu_clip))
        a[t] = float(np.clip(a_t + da_step, -cfg.a_clip, cfg.a_clip))

    if n > 1:
        F[-1] = F[-2]

    return {"mu": mu, "a": a, "F": F, "mu_prior": mu_prior}


def make_cfg_from_task(task: Dict[str, Any]) -> SimConfig:
    return SimConfig(
        T_units=float(task["T_units"]),
        dt=float(task["dt"]),
        seed=int(task["seed"]),
        condition_id=int(task["condition_id"]),
        mu=float(task["mu"]),
        lambda_S=float(task["lambda_S"]),
        lambda_A=float(task["lambda_A"]),
        gamma=float(task["gamma"]),
        chi=float(task["chi"]),
        sigma_E=float(task["sigma_E"]),
        sigma_S=float(task["sigma_S"]),
        sigma_I=float(task["sigma_I"]),
        response=str(task["response"]),
        H=float(task["H"]),
        r=float(task["r"]),
        gen_K=int(task["gen_K"]),
    )


def failure_row(task: Dict[str, Any], model: str, reason: str) -> Dict[str, Any]:
    row = {
        **task,
        "model": model,
        "boundary_dim": np.nan,
        "max_lag_steps": np.nan,
        "status": reason,
        "MSE_E": np.nan,
        "MSE_prior": np.nan,
        "F_mean": np.nan,
        "C_homeo": np.nan,
        "C_action": np.nan,
        "J_control": np.nan,
        "J_pen": np.nan,
        "CMI_delta": np.nan,
        "mean_abs_E": np.nan,
        "max_abs_E": np.nan,
        "mean_abs_mu": np.nan,
        "max_abs_mu": np.nan,
        "mean_abs_a": np.nan,
        "max_abs_a": np.nan,
        "clip_mu_fraction": np.nan,
        "clip_a_fraction": np.nan,
        "mean_abs_prior": np.nan,
        "max_abs_prior": np.nan,
        "MSE_I_current": np.nan,
        "REF_I_current": np.nan,
        "RI_I_current": np.nan,
        "MSE_S_delay": np.nan,
        "REF_S_delay": np.nan,
        "RI_S_delay": np.nan,
        "MSE_E_future_0p2": np.nan,
        "REF_E_future_0p2": np.nan,
        "RI_E_future_0p2": np.nan,
        "MSE_E_future_0p8": np.nan,
        "REF_E_future_0p8": np.nan,
        "RI_E_future_0p8": np.nan,
        "MSE_E_future_1p6": np.nan,
        "REF_E_future_1p6": np.nan,
        "RI_E_future_1p6": np.nan,
        "MSE_A_future_0p2": np.nan,
        "REF_A_future_0p2": np.nan,
        "RI_A_future_0p2": np.nan,
        "MSE_A_future_0p8": np.nan,
        "REF_A_future_0p8": np.nan,
        "RI_A_future_0p8": np.nan,
        "MSE_A_future_1p6": np.nan,
        "REF_A_future_1p6": np.nan,
        "RI_A_future_1p6": np.nan,
    }
    return row


def evaluate_one_task(task: Dict[str, Any]) -> List[Dict[str, Any]]:
    try:
        cfg = make_cfg_from_task(task)
        tau = float(task["tau"])
        rng = np.random.default_rng(int(task["seed"]) + 1000003 * int(task["condition_id"]))

        traj = simulate_trajectory(cfg, tau, rng)
        E, S, I, A = traj["E"], traj["S"], traj["I"], traj["A"]

        if np.any(~np.isfinite(E)) or np.any(~np.isfinite(S)) or np.any(~np.isfinite(I)) or np.any(~np.isfinite(A)):
            return [failure_row(task, model, "trajectory_unstable") for model in MODEL_NAMES]

        burn = int(round(cfg.burn_frac * len(E)))
        E = E[burn:]
        S = S[burn:]
        I = I[burn:]
        A = A[burn:]

        n = len(E)
        split = int(round(cfg.train_frac * n))
        if split < 100 or n - split < 100:
            return [failure_row(task, model, "too_short") for model in MODEL_NAMES]

        shift = int(rng.integers(low=max(1, n // 10), high=max(2, n - 1)))
        rows = []

        for model in MODEL_NAMES:
            try:
                X, max_lag = build_boundary_matrix(
                    S=S,
                    A=A,
                    model=model,
                    H=cfg.H,
                    r=cfg.r,
                    tau=tau,
                    dt=cfg.dt,
                    rng=rng,
                    circular_shift=shift,
                    gen_K=cfg.gen_K,
                    deriv_smooth_steps=cfg.deriv_smooth_steps,
                )

                y = E[max_lag:max_lag + len(X)]
                I_eff = I[max_lag:max_lag + len(X)]
                S_eff = S[max_lag:max_lag + len(X)]

                if len(X) < 200:
                    rows.append(failure_row(task, model, "boundary_too_short"))
                    continue

                split_eff = max(50, min(len(X) - 50, split - max_lag))
                if len(X) - split_eff < 50:
                    rows.append(failure_row(task, model, "test_too_short"))
                    continue

                X_train = X[:split_eff]
                y_train = y[:split_eff]
                X_test = X[split_eff:]
                E_test = y[split_eff:]
                S_test = S_eff[split_eff:]
                I_test = I_eff[split_eff:]

                theta = ridge_fit(X_train, y_train, cfg.ridge_alpha)

                ai = active_inference_rollout(
                    E_test=E_test,
                    S_test=S_test,
                    theta=theta,
                    X_test=X_test,
                    cfg=cfg,
                )

                mu = ai["mu"]
                a = ai["a"]
                F = ai["F"]
                prior = ai["mu_prior"]

                L = min(len(E_test), len(mu), len(a), len(F), len(prior), len(X_test), len(I_test))
                E_eval = E_test[:L]
                I_eval = I_test[:L]
                X_eval = X_test[:L]
                mu = mu[:L]
                a = a[:L]
                F = F[:L]
                prior = prior[:L]

                mse_E = safe_mean((E_eval - mu) ** 2)
                mse_prior = safe_mean((E_eval - prior) ** 2)
                F_mean = safe_mean(F)
                C_homeo = safe_mean((E_eval - cfg.E_target) ** 2)
                C_action = safe_mean(a ** 2)
                J_control = C_homeo + cfg.eta_action * C_action
                J_pen = J_control + cfg.lambda_dim * X_eval.shape[1]
                cmi = gaussian_cmi_scalar(I_eval, E_eval, X_eval)

                mu_clip_threshold = 0.99 * cfg.mu_clip
                a_clip_threshold = 0.99 * cfg.a_clip

                result_row = {
                    **task,
                    "model": model,
                    "boundary_dim": int(X_eval.shape[1]),
                    "max_lag_steps": int(max_lag),
                    "status": "ok",
                    "MSE_E": mse_E,
                    "MSE_prior": mse_prior,
                    "F_mean": F_mean,
                    "C_homeo": C_homeo,
                    "C_action": C_action,
                    "J_control": J_control,
                    "J_pen": J_pen,
                    "CMI_delta": cmi,
                    "mean_abs_E": safe_mean(np.abs(E_eval)),
                    "max_abs_E": float(np.nanmax(np.abs(E_eval))) if L else np.nan,
                    "mean_abs_mu": safe_mean(np.abs(mu)),
                    "max_abs_mu": float(np.nanmax(np.abs(mu))) if L else np.nan,
                    "mean_abs_a": safe_mean(np.abs(a)),
                    "max_abs_a": float(np.nanmax(np.abs(a))) if L else np.nan,
                    "clip_mu_fraction": safe_mean(np.abs(mu) >= mu_clip_threshold),
                    "clip_a_fraction": safe_mean(np.abs(a) >= a_clip_threshold),
                    "mean_abs_prior": safe_mean(np.abs(prior)),
                    "max_abs_prior": float(np.nanmax(np.abs(prior))) if L else np.nan,
                }

                result_row = add_multitarget_metrics_to_row(
                    row=result_row,
                    X=X,
                    E=E,
                    S=S,
                    I=I,
                    A=A,
                    max_lag=max_lag,
                    split_eff=split_eff,
                    tau=tau,
                    cfg=cfg,
                )
                rows.append(result_row)
            except Exception as exc:
                r = failure_row(task, model, f"model_error:{exc}")
                r["traceback"] = traceback.format_exc()
                rows.append(r)

        return rows

    except Exception as exc:
        return [{**task, "model": m, "status": f"task_error:{exc}", "traceback": traceback.format_exc()} for m in MODEL_NAMES]


# =============================================================================
# Resume/checkpoint
# =============================================================================

def task_uid(task: Dict[str, Any]) -> str:
    fields = [
        "tau", "H", "r", "gen_K", "mu", "sigma_E", "sigma_S", "sigma_I",
        "lambda_S", "lambda_A", "gamma", "chi", "response", "seed"
    ]
    return "|".join(str(task.get(k)) for k in fields)


def load_completed_uids(checkpoint_path: Path) -> set:
    if not checkpoint_path.exists():
        return set()
    try:
        df = pd.read_csv(checkpoint_path, usecols=["task_uid", "model"])
        if df.empty:
            return set()
        counts = df.groupby("task_uid")["model"].nunique()
        return set(counts[counts >= len(MODEL_NAMES)].index.tolist())
    except Exception:
        return set()


def append_rows_to_checkpoint(checkpoint_path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    exists = checkpoint_path.exists()

    first = [
        "task_uid", "condition_id", "seed", "tau", "H", "r", "gen_K",
        "mu", "sigma_E", "sigma_S", "sigma_I", "lambda_S", "lambda_A",
        "gamma", "chi", "response", "model", "status"
    ]

    for row in rows:
        if "task_uid" not in row:
            row["task_uid"] = task_uid(row)

    fields = sorted(set().union(*(row.keys() for row in rows)))
    fields = first + [f for f in fields if f not in first]

    with checkpoint_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        if not exists:
            writer.writeheader()
        for row in rows:
            writer.writerow(row)
        f.flush()


# =============================================================================
# Summaries
# =============================================================================

def add_relative_metrics(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    keys = [
        "condition_id", "seed", "tau", "H", "r", "gen_K", "mu", "sigma_E",
        "sigma_S", "sigma_I", "lambda_S", "lambda_A", "gamma", "chi", "response"
    ]

    base = df[df["model"] == "instant"][keys + ["MSE_E", "F_mean", "J_control", "CMI_delta"]].copy()
    base = base.rename(columns={
        "MSE_E": "MSE_E_instant",
        "F_mean": "F_mean_instant",
        "J_control": "J_control_instant",
        "CMI_delta": "CMI_delta_instant",
    })
    out = df.merge(base, on=keys, how="left")

    for col in ["MSE_E_instant", "F_mean_instant", "J_control_instant", "CMI_delta_instant"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    out["RI_E"] = (out["MSE_E_instant"] - out["MSE_E"]) / out["MSE_E_instant"].where(out["MSE_E_instant"].abs() > 1e-12, np.nan)
    out["RI_F"] = (out["F_mean_instant"] - out["F_mean"]) / out["F_mean_instant"].where(out["F_mean_instant"].abs() > 1e-12, np.nan)
    out["RI_J"] = (out["J_control_instant"] - out["J_control"]) / out["J_control_instant"].where(out["J_control_instant"].abs() > 1e-12, np.nan)
    out["RI_CMI"] = (out["CMI_delta_instant"] - out["CMI_delta"]) / out["CMI_delta_instant"].where(out["CMI_delta_instant"].abs() > 1e-10, np.nan)
    return out


def classify_regimes(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    group_keys = [
        "tau", "H", "r", "gen_K", "mu", "sigma_E", "sigma_S", "sigma_I",
        "lambda_S", "lambda_A", "gamma", "chi", "response"
    ]
    margin = 0.02

    ok = df[df["status"] == "ok"].copy()
    for key, g in ok.groupby(group_keys):
        means = g.groupby("model")[[
            "MSE_E", "F_mean", "J_control", "CMI_delta",
            "RI_E", "RI_F", "RI_J", "RI_CMI",
            "clip_mu_fraction", "clip_a_fraction"
        ]].mean(numeric_only=True)

        if not all(m in means.index for m in MODEL_NAMES):
            region = "D_incomplete"
            reason = "missing_model"
        else:
            inst = means.loc["instant"]
            hist = means.loc["history"]
            oracle = means.loc["oracle"]
            rand = means.loc["random"]
            shuf = means.loc["shuffled"]

            unstable = bool((means["clip_mu_fraction"].max() > 0) or (means["clip_a_fraction"].max() > 0))

            hist_gain = max(hist["RI_E"], hist["RI_F"], hist["RI_J"], hist["RI_CMI"] if np.isfinite(hist["RI_CMI"]) else -np.inf)
            rand_gain = max(rand["RI_E"], rand["RI_F"], rand["RI_J"], rand["RI_CMI"] if np.isfinite(rand["RI_CMI"]) else -np.inf)
            shuf_gain = max(shuf["RI_E"], shuf["RI_F"], shuf["RI_J"], shuf["RI_CMI"] if np.isfinite(shuf["RI_CMI"]) else -np.inf)
            oracle_gain = max(oracle["RI_E"], oracle["RI_F"], oracle["RI_J"], oracle["RI_CMI"] if np.isfinite(oracle["RI_CMI"]) else -np.inf)

            hist_beats_controls = (hist_gain > rand_gain + margin) and (hist_gain > shuf_gain + margin)
            hist_screening_ok = hist["CMI_delta"] < inst["CMI_delta"]

            instant_close = True
            for metric in ["MSE_E", "F_mean", "J_control", "CMI_delta"]:
                best = means[metric].min()
                if inst[metric] > best * (1.0 + margin):
                    instant_close = False

            if unstable:
                region = "D_numerically_unstable"
                reason = "clip_detected"
            elif instant_close:
                region = "A_instantaneous_sufficient"
                reason = "instant_close_to_best"
            elif hist_gain > margin and hist_beats_controls and hist_screening_ok:
                region = "B_history_superior"
                reason = "history_beats_instant_and_controls"
            elif oracle_gain > margin and hist_gain <= margin:
                region = "C_oracle_superior_generic_history_insufficient"
                reason = "oracle_only"
            else:
                region = "D_no_clear_advantage"
                reason = "no_clear_winner"

        row = dict(zip(group_keys, key))
        row.update({"region": region, "reason": reason})
        for model in MODEL_NAMES:
            if model in means.index:
                for metric in means.columns:
                    row[f"{model}_{metric}"] = float(means.loc[model, metric])
        rows.append(row)

    return pd.DataFrame(rows)


def summarize_and_write(df: pd.DataFrame, outdir: Path, args: argparse.Namespace, grid: GridSpec) -> None:
    if not df.empty:
        df = add_relative_metrics(df)

    df.to_csv(outdir / "all_results_long.csv", index=False)

    ok = df[df["status"] == "ok"].copy()

    model_summary = ok.groupby("model", as_index=False).agg(
        n=("MSE_E", "count"),
        MSE_E_mean=("MSE_E", "mean"),
        F_mean_mean=("F_mean", "mean"),
        J_control_mean=("J_control", "mean"),
        CMI_delta_mean=("CMI_delta", "mean"),
        RI_E_mean=("RI_E", "mean"),
        RI_F_mean=("RI_F", "mean"),
        RI_J_mean=("RI_J", "mean"),
        RI_CMI_mean=("RI_CMI", "mean"),
        mean_abs_E_mean=("mean_abs_E", "mean"),
        mean_abs_mu_mean=("mean_abs_mu", "mean"),
        clip_mu_fraction_mean=("clip_mu_fraction", "mean"),
        clip_a_fraction_mean=("clip_a_fraction", "mean"),
        MSE_I_current_mean=("MSE_I_current", "mean"),
        MSE_S_delay_mean=("MSE_S_delay", "mean"),
        MSE_E_future_0p8_mean=("MSE_E_future_0p8", "mean"),
        MSE_A_future_0p8_mean=("MSE_A_future_0p8", "mean"),
        RI_I_current_mean=("RI_I_current", "mean"),
        RI_S_delay_mean=("RI_S_delay", "mean"),
        RI_E_future_0p8_mean=("RI_E_future_0p8", "mean"),
        RI_A_future_0p8_mean=("RI_A_future_0p8", "mean"),
    ) if not ok.empty else pd.DataFrame()
    model_summary.to_csv(outdir / "model_comparison_summary.csv", index=False)

    regime = classify_regimes(df) if not df.empty else pd.DataFrame()
    regime.to_csv(outdir / "regime_summary.csv", index=False)

    group_metrics = [
        "MSE_E", "F_mean", "J_control", "CMI_delta", "RI_E", "RI_F", "RI_J", "RI_CMI",
        "MSE_I_current", "RI_I_current", "MSE_S_delay", "RI_S_delay",
        "MSE_E_future_0p2", "RI_E_future_0p2",
        "MSE_E_future_0p8", "RI_E_future_0p8",
        "MSE_E_future_1p6", "RI_E_future_1p6",
        "MSE_A_future_0p2", "RI_A_future_0p2",
        "MSE_A_future_0p8", "RI_A_future_0p8",
        "MSE_A_future_1p6", "RI_A_future_1p6",
    ]
    diagnostics = ["clip_mu_fraction", "clip_a_fraction", "mean_abs_E", "mean_abs_mu"]

    def grouped(name: str, keys: List[str]) -> pd.DataFrame:
        if ok.empty:
            out = pd.DataFrame()
        else:
            out = ok.groupby(keys, as_index=False).agg(
                **{f"{m}_mean": (m, "mean") for m in group_metrics},
                **{f"{m}_mean": (m, "mean") for m in diagnostics},
                n=("MSE_E", "count"),
            )
        out.to_csv(outdir / name, index=False)
        return out

    grouped("tau_dependence_summary.csv", ["tau", "model"])
    grouped("mu_tau_regime_summary.csv", ["mu", "tau", "model"])
    grouped("history_depth_summary.csv", ["H", "model"])
    grouped("generalized_order_summary.csv", ["gen_K", "model"])
    grouped("parameter_condition_summary.csv", [
        "tau", "H", "gen_K", "mu", "sigma_E", "sigma_S", "sigma_I",
        "lambda_S", "chi", "response", "model"
    ])

    multitarget_cols = [
        "MSE_I_current", "RI_I_current",
        "MSE_S_delay", "RI_S_delay",
        "MSE_E_future_0p2", "RI_E_future_0p2",
        "MSE_E_future_0p8", "RI_E_future_0p8",
        "MSE_E_future_1p6", "RI_E_future_1p6",
        "MSE_A_future_0p2", "RI_A_future_0p2",
        "MSE_A_future_0p8", "RI_A_future_0p8",
        "MSE_A_future_1p6", "RI_A_future_1p6",
    ]
    if not ok.empty:
        mt_cols = [c for c in multitarget_cols if c in ok.columns]
        # The script usually runs one analysis mode per output directory.
        # If a combined dataframe has analysis_mode, group by both; otherwise group by model only.
        if "analysis_mode" in ok.columns:
            multitarget_summary = ok.groupby(["analysis_mode", "model"], as_index=False)[mt_cols].mean(numeric_only=True)
        else:
            multitarget_summary = ok.groupby(["model"], as_index=False)[mt_cols].mean(numeric_only=True)
        multitarget_summary.to_csv(outdir / "multitarget_model_summary.csv", index=False)

        multitarget_by_tau = ok.groupby(["tau", "H", "model"], as_index=False)[mt_cols].mean(numeric_only=True)
        multitarget_by_tau.to_csv(outdir / "multitarget_tau_H_model_summary.csv", index=False)

    delay = ok[(ok["tau"] >= 0.8) & (ok["H"] >= 0.8)].copy()
    if not delay.empty:
        delay_summary = delay.groupby(["tau", "H", "mu", "chi", "sigma_E", "model"], as_index=False).agg(
            n=("MSE_E", "count"),
            MSE_E_mean=("MSE_E", "mean"),
            F_mean_mean=("F_mean", "mean"),
            J_control_mean=("J_control", "mean"),
            CMI_delta_mean=("CMI_delta", "mean"),
            RI_E_mean=("RI_E", "mean"),
            RI_F_mean=("RI_F", "mean"),
            RI_J_mean=("RI_J", "mean"),
            RI_CMI_mean=("RI_CMI", "mean"),
        )
    else:
        delay_summary = pd.DataFrame()
    delay_summary.to_csv(outdir / "delay_dominated_summary.csv", index=False)

    hist = ok[ok["model"] == "history"].copy()
    if not hist.empty:
        top_history = hist.sort_values(["RI_E", "RI_F", "RI_CMI"], ascending=False).head(200)
        cols = [
            "tau", "H", "gen_K", "mu", "sigma_E", "sigma_S", "sigma_I",
            "lambda_S", "chi", "response", "seed",
            "RI_E", "RI_F", "RI_J", "RI_CMI",
            "MSE_E", "F_mean", "J_control", "CMI_delta",
            "clip_mu_fraction", "clip_a_fraction"
        ]
        top_history = top_history[[c for c in cols if c in top_history.columns]]
    else:
        top_history = pd.DataFrame()
    top_history.to_csv(outdir / "top_history_conditions.csv", index=False)

    write_figures(df, regime, outdir)
    write_final_report(df, model_summary, regime, outdir, args, grid)


def write_figures(df: pd.DataFrame, regime: pd.DataFrame, outdir: Path) -> None:
    figdir = outdir / "figures"
    figdir.mkdir(parents=True, exist_ok=True)
    ok = df[df["status"] == "ok"].copy()
    if ok.empty:
        return

    # Fig 1: regime counts by tau
    if not regime.empty:
        counts = regime.groupby(["tau", "region"]).size().reset_index(name="count")
        piv = counts.pivot(index="tau", columns="region", values="count").fillna(0)
        plt.figure(figsize=(10, 5))
        bottom = np.zeros(len(piv))
        x = np.arange(len(piv.index))
        for col in piv.columns:
            vals = piv[col].values
            plt.bar(x, vals, bottom=bottom, label=col)
            bottom += vals
        plt.xticks(x, [str(v) for v in piv.index], rotation=45)
        plt.xlabel("Delay τ")
        plt.ylabel("Number of regimes")
        plt.title("Regime classification by delay")
        plt.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(figdir / "fig1_regime_map_by_delay.png", dpi=200)
        plt.close()

    def line_metric(metric: str, fname: str, ylabel: str):
        g = ok.groupby(["tau", "model"], as_index=False)[metric].mean()
        plt.figure(figsize=(9, 5))
        for model in MODEL_NAMES:
            gm = g[g["model"] == model].sort_values("tau")
            if len(gm):
                plt.plot(gm["tau"], gm[metric], marker="o", label=model)
        plt.xlabel("Delay τ")
        plt.ylabel(ylabel)
        plt.title(ylabel + " by delay")
        plt.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(figdir / fname, dpi=200)
        plt.close()

    line_metric("MSE_E", "fig2_mse_by_delay.png", "External-state inference error")
    line_metric("F_mean", "fig3_free_energy_by_delay.png", "Mean free energy")
    line_metric("J_control", "fig4_control_cost_by_delay.png", "Control cost")
    line_metric("CMI_delta", "fig5_cmi_by_delay.png", "Residual CMI")

    # Fig 6: heatmap-like tau x mu instantaneous sufficient fraction
    if not regime.empty:
        reg2 = regime.copy()
        reg2["is_instant"] = (reg2["region"] == "A_instantaneous_sufficient").astype(float)
        hm = reg2.groupby(["mu", "tau"], as_index=False)["is_instant"].mean()
        piv = hm.pivot(index="mu", columns="tau", values="is_instant")
        plt.figure(figsize=(8, 5))
        plt.imshow(piv.values, aspect="auto", origin="lower")
        plt.xticks(range(len(piv.columns)), [str(c) for c in piv.columns], rotation=45)
        plt.yticks(range(len(piv.index)), [str(i) for i in piv.index])
        plt.xlabel("Delay τ")
        plt.ylabel("External dissipation μ")
        plt.title("Fraction classified as instantaneous sufficient")
        plt.colorbar(label="fraction")
        plt.tight_layout()
        plt.savefig(figdir / "fig6_mu_tau_instantaneous_sufficiency.png", dpi=200)
        plt.close()


def write_final_report(
    df: pd.DataFrame,
    model_summary: pd.DataFrame,
    regime: pd.DataFrame,
    outdir: Path,
    args: argparse.Namespace,
    grid: GridSpec,
) -> None:
    lines = []
    lines.append("Balanced active-inference blanket comparison")
    lines.append("============================================")
    lines.append("")
    lines.append(f"Mode: {args.mode}")
    lines.append(f"Created: {time.ctime()}")
    lines.append(f"Elapsed seconds: {time.time() - T0_GLOBAL:.1f}")
    lines.append(f"Rows: {len(df)}")
    lines.append(f"OK rows: {(df['status'] == 'ok').sum() if 'status' in df.columns else 0}")
    lines.append("")
    lines.append("Grid")
    lines.append("----")
    lines.append(json.dumps(asdict(grid), indent=2))
    lines.append("")
    lines.append("Status counts")
    lines.append("-------------")
    lines.append(df["status"].value_counts(dropna=False).to_string() if not df.empty else "No rows")
    lines.append("")
    lines.append("Model summary")
    lines.append("-------------")
    lines.append(model_summary.to_string(index=False) if not model_summary.empty else "No model summary")
    lines.append("")
    lines.append("Regime counts")
    lines.append("-------------")
    lines.append(regime["region"].value_counts().to_string() if not regime.empty else "No regime summary")
    lines.append("")
    lines.append("Numerical diagnostics")
    lines.append("---------------------")
    diag_cols = [
        "MSE_E", "F_mean", "J_control", "CMI_delta",
        "mean_abs_E", "max_abs_E", "mean_abs_mu", "max_abs_mu",
        "clip_mu_fraction", "clip_a_fraction"
    ]
    if not df.empty:
        lines.append(df[[c for c in diag_cols if c in df.columns]].describe().T.to_string())
    lines.append("")
    lines.append("Files")
    lines.append("-----")
    for f in sorted(outdir.rglob("*")):
        if f.is_file():
            lines.append(str(f.relative_to(outdir)))

    (outdir / "final_console_report.txt").write_text("\n".join(lines), encoding="utf-8")
    (outdir / "consolidated_report.txt").write_text("\n".join(lines), encoding="utf-8")


# =============================================================================
# Task generation and main
# =============================================================================

def generate_tasks(grid: GridSpec, max_tasks: Optional[int] = None) -> List[Dict[str, Any]]:
    tasks = []
    cid = 0
    iterable = itertools.product(
        grid.taus,
        grid.Hs,
        grid.rs,
        grid.gen_Ks,
        grid.mus,
        grid.sigma_Es,
        grid.sigma_pairs,
        grid.lambda_Ss,
        grid.lambda_As,
        grid.gammas,
        grid.chis,
        grid.responses,
        grid.seeds,
    )

    for tau, H, r, gen_K, mu, sigma_E, sigpair, lambda_S, lambda_A, gamma, chi, response, seed in iterable:
        sigma_S, sigma_I = sigpair
        task = {
            "condition_id": cid,
            "tau": float(tau),
            "H": float(H),
            "r": float(r),
            "gen_K": int(gen_K),
            "mu": float(mu),
            "sigma_E": float(sigma_E),
            "sigma_S": float(sigma_S),
            "sigma_I": float(sigma_I),
            "lambda_S": float(lambda_S),
            "lambda_A": float(lambda_A),
            "gamma": float(gamma),
            "chi": float(chi),
            "response": response,
            "seed": int(seed),
            "T_units": float(grid.T_units),
            "dt": float(grid.dt),
        }
        task["task_uid"] = task_uid(task)
        tasks.append(task)
        cid += 1
        if max_tasks is not None and len(tasks) >= max_tasks:
            break

    return tasks


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["smoke", "baseline", "baseline_both_response", "oat_mu", "oat_chi", "oat_sigmaE", "oat_noise", "oat_lambdaS", "oat_genK", "robustness_nonlinear"], default="smoke")
    ap.add_argument("--outdir", type=str, default=None)
    ap.add_argument("--n_jobs", type=int, default=1)
    ap.add_argument("--seeds", type=int, default=None)
    ap.add_argument("--max_tasks", type=int, default=None)
    ap.add_argument("--progress_every", type=int, default=50)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--checkpoint_name", type=str, default="checkpoint_rows.csv")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    outdir = Path(args.outdir) if args.outdir else Path.home() / "Desktop" / f"ai_blanket_{args.mode}"
    outdir.mkdir(parents=True, exist_ok=True)

    grid = make_grid(args.mode, args.seeds)
    tasks_all = generate_tasks(grid, args.max_tasks)

    checkpoint_path = outdir / args.checkpoint_name
    completed = load_completed_uids(checkpoint_path) if args.resume else set()
    tasks = [t for t in tasks_all if t["task_uid"] not in completed]

    manifest = {
        "script": "active_inference_temporal_blanket_multitask.py",
        "mode": args.mode,
        "resume": bool(args.resume),
        "checkpoint_path": str(checkpoint_path),
        "n_jobs": args.n_jobs,
        "n_tasks_total": len(tasks_all),
        "n_tasks_completed_at_start": len(completed),
        "n_tasks_to_run": len(tasks),
        "models": MODEL_NAMES,
        "grid": asdict(grid),
        "central_design": {
            "generative_process": "E -> S -> I -> A -> E",
            "explicit_delay": "S(t - tau) -> I(t)",
            "comparison": "recognition model only",
            "models": MODEL_NAMES,
        },
    }
    (outdir / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    log(f"Output directory: {outdir}")
    log(f"Mode: {args.mode}")
    log(f"Generated total tasks: {len(tasks_all)}")
    log(f"Completed at start: {len(completed)}")
    log(f"Remaining tasks: {len(tasks)}")
    log(f"Models per task: {len(MODEL_NAMES)}")
    log(f"Checkpoint: {checkpoint_path}")

    new_rows: List[Dict[str, Any]] = []
    start = time.time()

    if len(tasks) == 0:
        log("No remaining tasks. Rebuilding outputs from checkpoint.")
    elif args.n_jobs <= 1:
        for i, task in enumerate(tasks, 1):
            rows = evaluate_one_task(task)
            for row in rows:
                row["task_uid"] = task["task_uid"]
            append_rows_to_checkpoint(checkpoint_path, rows)
            new_rows.extend(rows)
            if i == 1 or i % args.progress_every == 0 or i == len(tasks):
                rate = i / max(1e-9, time.time() - start)
                log(f"Progress {i}/{len(tasks)} remaining tasks; new_rows={len(new_rows)}; rate={rate:.2f} tasks/s")
    else:
        with ProcessPoolExecutor(max_workers=args.n_jobs) as ex:
            future_to_task = {ex.submit(evaluate_one_task, task): task for task in tasks}
            for i, fut in enumerate(as_completed(future_to_task), 1):
                task = future_to_task[fut]
                rows = fut.result()
                for row in rows:
                    row["task_uid"] = task["task_uid"]
                append_rows_to_checkpoint(checkpoint_path, rows)
                new_rows.extend(rows)
                if i == 1 or i % args.progress_every == 0 or i == len(tasks):
                    rate = i / max(1e-9, time.time() - start)
                    log(f"Progress {i}/{len(tasks)} remaining tasks; new_rows={len(new_rows)}; rate={rate:.2f} tasks/s")

    log("Loading checkpoint and writing final outputs...")
    if checkpoint_path.exists():
        df = pd.read_csv(checkpoint_path)
        if "task_uid" in df.columns and "model" in df.columns:
            df = df.drop_duplicates(subset=["task_uid", "model"], keep="last")
    else:
        df = pd.DataFrame(new_rows)

    summarize_and_write(df, outdir, args, grid)

    log("Done.")
    log(f"Final report: {outdir / 'final_console_report.txt'}")


if __name__ == "__main__":
    main()
