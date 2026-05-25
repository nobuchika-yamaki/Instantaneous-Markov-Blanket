#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
analyze_multitask_results_complete.py

Complete post-hoc analysis for the multi-target active-inference blanket results.

This script produces:
- global quality checks
- model summaries by analysis mode
- baseline model summaries
- paired model-minus-instant differences if raw results are available
- taskwise winners for each target metric
- regime counts and top history/oracle conditions
- OAT sensitivity summaries
- final figures
- one consolidated RESULTS_COMPLETE_REPORT.txt

Usage:
  python3 -u analyze_multitask_results_complete.py \
    --input_dir ~/Desktop/ai_blanket_multitask_analysis/_combined \
    --outdir ~/Desktop/ai_blanket_multitask_analysis/_complete_analysis \
    --n_boot 5000
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


T0 = time.time()

MODEL_ORDER = ["instant", "generalized", "history", "oracle", "random", "shuffled"]
MODEL_LABEL = {
    "instant": "Instantaneous",
    "generalized": "Generalized",
    "history": "History",
    "oracle": "Oracle",
    "random": "Random",
    "shuffled": "Shuffled",
}

REGION_ORDER = [
    "A_instantaneous_sufficient",
    "B_history_superior",
    "C_oracle_superior_generic_history_insufficient",
    "D_no_clear_advantage",
    "D_incomplete",
    "D_numerically_unstable",
]

REGION_LABEL = {
    "A_instantaneous_sufficient": "Instantaneous sufficient",
    "B_history_superior": "History superior",
    "C_oracle_superior_generic_history_insufficient": "Oracle superior",
    "D_no_clear_advantage": "No clear advantage",
    "D_incomplete": "Incomplete",
    "D_numerically_unstable": "Numerically unstable",
}

PRIMARY_METRICS = [
    "MSE_E",
    "CMI_delta",
    "MSE_I_current",
    "MSE_S_delay",
    "MSE_E_future_0p2",
    "MSE_E_future_0p8",
    "MSE_E_future_1p6",
    "MSE_A_future_0p2",
    "MSE_A_future_0p8",
    "MSE_A_future_1p6",
]

RI_METRICS = [
    "RI_I_current",
    "RI_S_delay",
    "RI_E_future_0p2",
    "RI_E_future_0p8",
    "RI_E_future_1p6",
    "RI_A_future_0p2",
    "RI_A_future_0p8",
    "RI_A_future_1p6",
]

DIAG_METRICS = [
    "clip_mu_fraction",
    "clip_a_fraction",
    "mean_abs_E",
    "max_abs_E",
    "mean_abs_mu",
    "max_abs_mu",
]


class Tee:
    def __init__(self, *files):
        self.files = files
    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()
    def flush(self):
        for f in self.files:
            f.flush()


def log(msg: str) -> None:
    print(f"[{time.time() - T0:8.1f}s] {msg}", flush=True)


def expand(p: str | Path) -> Path:
    return Path(os.path.expanduser(str(p))).resolve()


def first_existing(input_dir: Path, names: List[str]) -> Optional[Path]:
    for name in names:
        p = input_dir / name
        if p.exists():
            return p
    return None


def numericize(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def bootstrap_ci(x: np.ndarray, n_boot: int, rng: np.random.Generator):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return np.nan, np.nan, np.nan, np.nan
    mean = float(np.mean(x))
    sd = float(np.std(x, ddof=1)) if len(x) > 1 else np.nan
    if len(x) < 2 or n_boot <= 0:
        return mean, np.nan, np.nan, sd
    idx = rng.integers(0, len(x), size=(n_boot, len(x)))
    boot = np.mean(x[idx], axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return mean, float(lo), float(hi), sd


def mean_ci_table(df: pd.DataFrame, group_cols: List[str], value_cols: List[str], n_boot: int, seed: int) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    rng = np.random.default_rng(seed)
    rows = []
    for key, g in df.groupby(group_cols, dropna=False):
        if not isinstance(key, tuple):
            key = (key,)
        row = dict(zip(group_cols, key))
        row["n_rows"] = len(g)
        if "seed" in g.columns:
            row["n_seeds"] = g["seed"].nunique()
        for col in value_cols:
            if col not in g.columns:
                continue
            mean, lo, hi, sd = bootstrap_ci(pd.to_numeric(g[col], errors="coerce").to_numpy(), n_boot, rng)
            row[f"{col}_mean"] = mean
            row[f"{col}_ci95_low"] = lo
            row[f"{col}_ci95_high"] = hi
            row[f"{col}_sd"] = sd
        rows.append(row)
    return pd.DataFrame(rows)


def paired_vs_instant(df: pd.DataFrame, metrics: List[str], n_boot: int) -> pd.DataFrame:
    if df.empty or "task_uid" not in df.columns:
        return pd.DataFrame()
    keys = ["analysis_mode", "task_uid"]
    inst = df[df["model"] == "instant"][keys + [m for m in metrics if m in df.columns]].copy()
    inst = inst.rename(columns={m: f"{m}_instant" for m in metrics if m in inst.columns})
    comp = df[df["model"] != "instant"].copy()
    comp = comp.drop(columns=[c for c in comp.columns if c.endswith("_instant")], errors="ignore")
    merged = comp.merge(inst, on=keys, how="inner", validate="many_to_one")
    diff_cols = []
    for m in metrics:
        if m in merged.columns and f"{m}_instant" in merged.columns:
            col = f"d_{m}_model_minus_instant"
            merged[col] = merged[m] - merged[f"{m}_instant"]
            diff_cols.append(col)
    return mean_ci_table(merged, ["model"], diff_cols, n_boot, seed=123)


def taskwise_winners(df: pd.DataFrame, group_cols: List[str], metrics: List[str]) -> pd.DataFrame:
    rows = []
    if df.empty:
        return pd.DataFrame()
    for metric in metrics:
        if metric not in df.columns:
            continue
        for key, g in df.groupby(group_cols, dropna=False):
            if not isinstance(key, tuple):
                key = (key,)
            means = g.groupby("model")[metric].mean(numeric_only=True).dropna().sort_values()
            if means.empty:
                continue
            best_model = means.index[0]
            best_val = float(means.iloc[0])
            inst_val = float(means.loc["instant"]) if "instant" in means.index else np.nan
            row = dict(zip(group_cols, key))
            row.update({
                "metric": metric,
                "best_model": best_model,
                "best_value": best_val,
                "instant_value": inst_val,
                "best_minus_instant": best_val - inst_val if np.isfinite(inst_val) else np.nan,
                "relative_gain_vs_instant": (inst_val - best_val) / inst_val if np.isfinite(inst_val) and abs(inst_val) > 1e-12 else np.nan,
            })
            rows.append(row)
    return pd.DataFrame(rows)


def regime_counts(reg: pd.DataFrame) -> pd.DataFrame:
    if reg is None or reg.empty or "region" not in reg.columns:
        return pd.DataFrame()
    if "analysis_mode" not in reg.columns:
        reg = reg.copy()
        reg["analysis_mode"] = "unknown"
    rows = []
    for mode, g in reg.groupby("analysis_mode"):
        total = len(g)
        for region, count in g["region"].value_counts().items():
            rows.append({
                "analysis_mode": mode,
                "region": region,
                "count": int(count),
                "percent": 100.0 * count / total if total else np.nan,
                "total_conditions": total,
            })
    return pd.DataFrame(rows).sort_values(["analysis_mode", "region"])


def top_conditions(reg: pd.DataFrame, region: str, top_n: int) -> pd.DataFrame:
    if reg is None or reg.empty or "region" not in reg.columns:
        return pd.DataFrame()
    sub = reg[reg["region"] == region].copy()
    if sub.empty:
        return pd.DataFrame()
    if region == "B_history_superior":
        sub["rank_score"] = (
            (sub.get("instant_CMI_delta", np.nan) - sub.get("history_CMI_delta", np.nan)).fillna(0)
            + (sub.get("instant_MSE_E", np.nan) - sub.get("history_MSE_E", np.nan)).fillna(0)
            + (sub.get("instant_F_mean", np.nan) - sub.get("history_F_mean", np.nan)).fillna(0)
        )
    elif region == "C_oracle_superior_generic_history_insufficient":
        sub["rank_score"] = (
            (sub.get("instant_CMI_delta", np.nan) - sub.get("oracle_CMI_delta", np.nan)).fillna(0)
            + (sub.get("instant_MSE_E", np.nan) - sub.get("oracle_MSE_E", np.nan)).fillna(0)
            + (sub.get("instant_F_mean", np.nan) - sub.get("oracle_F_mean", np.nan)).fillna(0)
        )
    else:
        sub["rank_score"] = 0.0
    preferred = [
        "analysis_mode", "region", "reason", "tau", "H", "gen_K", "mu", "sigma_E", "sigma_S", "sigma_I",
        "lambda_S", "chi", "response", "rank_score",
        "instant_MSE_E", "generalized_MSE_E", "history_MSE_E", "oracle_MSE_E",
        "instant_CMI_delta", "generalized_CMI_delta", "history_CMI_delta", "oracle_CMI_delta",
        "instant_F_mean", "generalized_F_mean", "history_F_mean", "oracle_F_mean",
    ]
    cols = [c for c in preferred if c in sub.columns]
    return sub.sort_values("rank_score", ascending=False)[cols].head(top_n)


def oat_parameter_effects(df: pd.DataFrame) -> pd.DataFrame:
    mode_to_param = {
        "oat_mu": "mu",
        "oat_chi": "chi",
        "oat_sigmaE": "sigma_E",
        "oat_noise": "sigma_S",
        "oat_lambdaS": "lambda_S",
        "oat_genK": "gen_K",
        "robustness_nonlinear": "response",
    }
    rows = []
    for mode, param in mode_to_param.items():
        sub = df[df["analysis_mode"] == mode].copy()
        if sub.empty or param not in sub.columns:
            continue
        metrics = [c for c in PRIMARY_METRICS + RI_METRICS if c in sub.columns]
        tab = sub.groupby(["analysis_mode", param, "model"], as_index=False)[metrics].mean(numeric_only=True)
        tab = tab.rename(columns={param: "varied_parameter_value"})
        tab["varied_parameter"] = param
        rows.append(tab)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def savefig(path: Path) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


def plot_tau(df: pd.DataFrame, metric: str, out: Path, title: str) -> None:
    if metric not in df.columns:
        return
    base = df[(df["analysis_mode"] == "baseline") & (df["status"].isin(["ok", "summary"]))].copy()
    if base.empty or "tau" not in base.columns:
        return
    stat = base.groupby(["tau", "model"], as_index=False).agg(
        mean=(metric, "mean"),
        sem=(metric, lambda x: np.nanstd(x, ddof=1) / np.sqrt(np.sum(np.isfinite(x))) if np.sum(np.isfinite(x)) > 1 else 0.0),
    )
    plt.figure(figsize=(8.5, 5.2))
    for model in MODEL_ORDER:
        g = stat[stat["model"] == model].sort_values("tau")
        if len(g):
            plt.errorbar(g["tau"], g["mean"], yerr=g["sem"], marker="o", capsize=3, label=MODEL_LABEL[model])
    plt.xlabel("Delay τ")
    plt.ylabel(metric)
    plt.title(title)
    plt.legend(fontsize=8)
    savefig(out)


def make_figures(df: pd.DataFrame, reg: pd.DataFrame, outdir: Path) -> None:
    figdir = outdir / "figures"
    figdir.mkdir(exist_ok=True)
    log("Creating figures")

    if reg is not None and not reg.empty and "region" in reg.columns:
        base_reg = reg[reg["analysis_mode"] == "baseline"].copy() if "analysis_mode" in reg.columns else reg.copy()
        if "tau" in base_reg.columns and not base_reg.empty:
            counts = base_reg.groupby(["tau", "region"]).size().reset_index(name="count")
            piv = counts.pivot(index="tau", columns="region", values="count").fillna(0)
            for r in REGION_ORDER:
                if r not in piv.columns:
                    piv[r] = 0
            piv = piv[REGION_ORDER]
            x = np.arange(len(piv.index))
            bottom = np.zeros(len(piv.index))
            plt.figure(figsize=(8.5, 5.2))
            for r in REGION_ORDER:
                vals = piv[r].values
                if np.any(vals > 0):
                    plt.bar(x, vals, bottom=bottom, label=REGION_LABEL.get(r, r))
                    bottom += vals
            plt.xticks(x, [str(v) for v in piv.index])
            plt.xlabel("Delay τ")
            plt.ylabel("Number of baseline conditions")
            plt.title("Baseline regime classification by delay")
            plt.legend(fontsize=8)
            savefig(figdir / "Fig1_baseline_regime_by_delay.png")

    plot_tau(df, "MSE_E", figdir / "Fig2_current_E_MSE_by_delay.png", "Current external-state inference")
    plot_tau(df, "CMI_delta", figdir / "Fig3_CMI_by_delay.png", "Residual CMI")
    plot_tau(df, "MSE_I_current", figdir / "Fig4_internal_state_MSE_by_delay.png", "Internal-state reconstruction")
    plot_tau(df, "MSE_S_delay", figdir / "Fig5_delayed_sensory_MSE_by_delay.png", "Delay-aligned sensory reconstruction")
    plot_tau(df, "MSE_A_future_1p6", figdir / "Fig6_future_A_1p6_MSE_by_delay.png", "Future active-boundary prediction")

    if reg is not None and not reg.empty and "region" in reg.columns and "analysis_mode" in reg.columns:
        counts = reg.groupby(["analysis_mode", "region"]).size().reset_index(name="count")
        piv = counts.pivot(index="analysis_mode", columns="region", values="count").fillna(0)
        for r in REGION_ORDER:
            if r not in piv.columns:
                piv[r] = 0
        piv = piv[REGION_ORDER]
        x = np.arange(len(piv.index))
        bottom = np.zeros(len(piv.index))
        plt.figure(figsize=(10, 5.5))
        for r in REGION_ORDER:
            vals = piv[r].values
            if np.any(vals > 0):
                plt.bar(x, vals, bottom=bottom, label=REGION_LABEL.get(r, r))
                bottom += vals
        plt.xticks(x, list(piv.index), rotation=35, ha="right")
        plt.xlabel("Analysis mode")
        plt.ylabel("Number of conditions")
        plt.title("Regime counts across analyses")
        plt.legend(fontsize=8)
        savefig(figdir / "FigS1_regime_counts_all_modes.png")


def write_report(outdir: Path, df: pd.DataFrame, tables: dict, raw_available: bool) -> None:
    lines = []
    lines.append("COMPLETE MULTI-TARGET RESULTS ANALYSIS")
    lines.append("======================================")
    lines.append("")
    lines.append(f"Created: {time.ctime()}")
    lines.append(f"Raw combined_all_results_long.csv available: {raw_available}")
    lines.append(f"Total rows analyzed: {len(df):,}")
    lines.append("")

    lines.append("1. Global quality")
    lines.append("-----------------")
    if "status" in df.columns:
        lines.append(df["status"].value_counts(dropna=False).to_string())
    if "model" in df.columns:
        lines.append("")
        lines.append("Rows per model:")
        lines.append(df["model"].value_counts().to_string())
    if {"clip_mu_fraction", "clip_a_fraction"}.issubset(df.columns):
        lines.append("")
        lines.append("Maximum clipping:")
        lines.append(df[["clip_mu_fraction", "clip_a_fraction"]].max().to_string())
    lines.append("")

    lines.append("2. Regime counts")
    lines.append("----------------")
    rc = tables.get("regime_counts_percent", pd.DataFrame())
    lines.append(rc.to_string(index=False) if not rc.empty else "Not available.")
    lines.append("")

    lines.append("3. Baseline model summary")
    lines.append("-------------------------")
    base = tables.get("baseline_model_summary", pd.DataFrame())
    if not base.empty:
        # Raw tables have *_mean columns; summary tables already have raw metric names.
        preferred = [
            "model", "n_rows", "n_seeds",
            "MSE_E_mean", "MSE_E",
            "CMI_delta_mean", "CMI_delta",
            "MSE_I_current_mean", "MSE_I_current",
            "MSE_S_delay_mean", "MSE_S_delay",
            "MSE_E_future_0p8_mean", "MSE_E_future_0p8",
            "MSE_A_future_1p6_mean", "MSE_A_future_1p6",
            "clip_mu_fraction_mean", "clip_mu_fraction",
            "clip_a_fraction_mean", "clip_a_fraction",
        ]
        cols = [c for c in preferred if c in base.columns]
        lines.append(base[cols].to_string(index=False))
    else:
        lines.append("Not available.")
    lines.append("")

    lines.append("4. Paired model-minus-instant differences")
    lines.append("-----------------------------------------")
    pair = tables.get("baseline_paired_vs_instant", pd.DataFrame())
    if not pair.empty:
        cols = [c for c in [
            "model", "n_rows", "n_seeds",
            "d_MSE_E_model_minus_instant_mean",
            "d_CMI_delta_model_minus_instant_mean",
            "d_MSE_I_current_model_minus_instant_mean",
            "d_MSE_S_delay_model_minus_instant_mean",
            "d_MSE_A_future_1p6_model_minus_instant_mean",
        ] if c in pair.columns]
        lines.append(pair[cols].to_string(index=False))
    else:
        lines.append("Not available. Raw task-level combined_all_results_long.csv is required.")
    lines.append("")

    lines.append("5. Baseline taskwise winner counts")
    lines.append("----------------------------------")
    bwc = tables.get("baseline_taskwise_winner_counts", pd.DataFrame())
    lines.append(bwc.to_string(index=False) if not bwc.empty else "Not available.")
    lines.append("")

    lines.append("6. OAT parameter effects")
    lines.append("------------------------")
    oat = tables.get("oat_parameter_effects", pd.DataFrame())
    if not oat.empty:
        cols = [c for c in [
            "analysis_mode", "varied_parameter", "varied_parameter_value", "model",
            "MSE_E", "CMI_delta", "MSE_I_current", "MSE_S_delay", "MSE_A_future_1p6"
        ] if c in oat.columns]
        lines.append(oat[cols].head(300).to_string(index=False))
        if len(oat) > 300:
            lines.append(f"... truncated; full CSV has {len(oat)} rows.")
    else:
        lines.append("Not available.")
    lines.append("")

    lines.append("7. Interpretation")
    lines.append("-----------------")
    lines.append("- Current E inference alone is Markov-friendly because S_n is directly informative about E_n.")
    lines.append("- The key multi-target metrics are MSE_I_current, MSE_S_delay, and MSE_A_future_*.")
    lines.append("- Use CMI_delta and model-minus-instant differences, not RI_CMI, for blanket screening.")
    lines.append("- Control cost remains secondary under fixed-trajectory evaluation.")
    lines.append("")

    lines.append("8. Files written")
    lines.append("----------------")
    for p in sorted(outdir.rglob("*")):
        if p.is_file():
            lines.append(str(p.relative_to(outdir)))

    (outdir / "RESULTS_COMPLETE_REPORT.txt").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, default=".")
    parser.add_argument("--outdir", type=str, default=None)
    parser.add_argument("--n_boot", type=int, default=5000)
    parser.add_argument("--top_n", type=int, default=100)
    args = parser.parse_args()

    input_dir = expand(args.input_dir)
    outdir = expand(args.outdir) if args.outdir else input_dir / "complete_analysis"
    outdir.mkdir(parents=True, exist_ok=True)

    log_handle = (outdir / "00_RUN_LOG.txt").open("w", encoding="utf-8")
    sys.stdout = Tee(sys.__stdout__, log_handle)

    log(f"Input directory: {input_dir}")
    log(f"Output directory: {outdir}")
    log(f"Bootstrap resamples: {args.n_boot}")

    raw_path = first_existing(input_dir, ["combined_all_results_long.csv", "all_results_long.csv"])
    summary_path = first_existing(input_dir, ["combined_multitarget_results_summary.csv", "combined_multitarget_model_summary.csv"])
    reg_path = first_existing(input_dir, ["combined_regime_summary.csv", "combined_regime_summary(1).csv", "regime_summary.csv"])
    reg_counts_path = first_existing(input_dir, ["combined_regime_counts.csv", "combined_regime_counts(1).csv"])

    # Use raw data only if it is actually the multi-target raw output.
    # Older current-state-only combined_all_results_long.csv files may coexist
    # in the same folder; these must not be mistaken for multi-target results.
    raw_available = False
    if raw_path is not None:
        log(f"Checking raw results: {raw_path}")
        raw_head = pd.read_csv(raw_path, nrows=5)
        required_multitarget_cols = {"MSE_I_current", "MSE_S_delay", "MSE_A_future_1p6"}
        if required_multitarget_cols.issubset(set(raw_head.columns)):
            raw_available = True
            log(f"Reading multi-target raw results: {raw_path}")
            df = pd.read_csv(raw_path)
        else:
            log("Raw file exists but lacks multi-target columns; ignoring it for this analysis.")
            df = None
    else:
        df = None

    if not raw_available:
        if summary_path is not None:
            log(f"Reading multi-target summary results: {summary_path}")
            df = pd.read_csv(summary_path)
        else:
            raise FileNotFoundError("No multi-target raw file or combined_multitarget_results_summary.csv found.")

    df = numericize(df, PRIMARY_METRICS + RI_METRICS + DIAG_METRICS)
    if "status" not in df.columns:
        df["status"] = "summary"
    if "analysis_mode" not in df.columns:
        df["analysis_mode"] = "unknown"

    reg = pd.DataFrame()
    if reg_path is not None:
        log(f"Reading regime summary: {reg_path}")
        reg = pd.read_csv(reg_path)
    elif reg_counts_path is not None:
        log(f"Reading regime counts: {reg_counts_path}")
        reg = pd.read_csv(reg_counts_path)

    ok = df[df["status"].isin(["ok", "summary"])].copy()
    value_cols = [c for c in PRIMARY_METRICS + RI_METRICS + DIAG_METRICS if c in ok.columns]

    tables = {}

    log("Computing global quality.")
    q_agg = {"rows": ("model", "count"), "n_models": ("model", "nunique")}
    if "task_uid" in ok.columns:
        q_agg["n_task_uid"] = ("task_uid", "nunique")
    if "seed" in ok.columns:
        q_agg["n_seeds"] = ("seed", "nunique")
    for c in ["clip_mu_fraction", "clip_a_fraction", "max_abs_E", "max_abs_mu"]:
        if c in ok.columns:
            q_agg[f"{c}_max"] = (c, "max")
    tables["global_quality"] = ok.groupby("analysis_mode", as_index=False).agg(**q_agg)

    log("Computing model summaries.")
    if raw_available:
        tables["model_summary_by_mode"] = mean_ci_table(ok, ["analysis_mode", "model"], value_cols, args.n_boot, seed=1)
        baseline = ok[ok["analysis_mode"] == "baseline"].copy()
        tables["baseline_model_summary"] = mean_ci_table(baseline, ["model"], value_cols, args.n_boot, seed=2)
        tables["baseline_paired_vs_instant"] = paired_vs_instant(baseline, [c for c in PRIMARY_METRICS if c in baseline.columns], args.n_boot)
    else:
        tables["model_summary_by_mode"] = ok.copy()
        tables["baseline_model_summary"] = ok[ok["analysis_mode"] == "baseline"].copy()
        tables["baseline_paired_vs_instant"] = pd.DataFrame()

    log("Computing taskwise winners.")
    if raw_available:
        group_cols = [c for c in ["analysis_mode", "tau", "H", "mu", "chi", "sigma_E", "sigma_S", "lambda_S", "response"] if c in ok.columns]
        win = taskwise_winners(ok, group_cols, [c for c in PRIMARY_METRICS if c in ok.columns])
        tables["taskwise_winners"] = win
        if not win.empty:
            tables["taskwise_winner_counts"] = win.groupby(["analysis_mode", "metric", "best_model"], as_index=False).size().rename(columns={"size": "count"})
            bwin = win[win["analysis_mode"] == "baseline"].copy() if "analysis_mode" in win.columns else pd.DataFrame()
            tables["baseline_taskwise_winners"] = bwin
            tables["baseline_taskwise_winner_counts"] = bwin.groupby(["metric", "best_model"], as_index=False).size().rename(columns={"size": "count"}) if not bwin.empty else pd.DataFrame()
            tables["oat_taskwise_winners"] = win[win["analysis_mode"] != "baseline"].copy()
        else:
            tables["taskwise_winner_counts"] = pd.DataFrame()
            tables["baseline_taskwise_winners"] = pd.DataFrame()
            tables["baseline_taskwise_winner_counts"] = pd.DataFrame()
            tables["oat_taskwise_winners"] = pd.DataFrame()
    else:
        for name in ["taskwise_winners", "taskwise_winner_counts", "baseline_taskwise_winners", "baseline_taskwise_winner_counts", "oat_taskwise_winners"]:
            tables[name] = pd.DataFrame()

    log("Computing regime tables.")
    if not reg.empty and "region" in reg.columns:
        tables["regime_counts_percent"] = regime_counts(reg)
        tables["top_history_conditions"] = top_conditions(reg, "B_history_superior", args.top_n)
        tables["top_oracle_conditions"] = top_conditions(reg, "C_oracle_superior_generic_history_insufficient", args.top_n)
    elif not reg.empty and {"analysis_mode", "region", "count"}.issubset(reg.columns):
        reg2 = reg.copy()
        total = reg2.groupby("analysis_mode")["count"].transform("sum")
        reg2["percent"] = 100 * reg2["count"] / total
        tables["regime_counts_percent"] = reg2
        tables["top_history_conditions"] = pd.DataFrame()
        tables["top_oracle_conditions"] = pd.DataFrame()
    else:
        tables["regime_counts_percent"] = pd.DataFrame()
        tables["top_history_conditions"] = pd.DataFrame()
        tables["top_oracle_conditions"] = pd.DataFrame()

    log("Computing OAT parameter effects.")
    tables["oat_parameter_effects"] = oat_parameter_effects(ok) if raw_available else ok[ok["analysis_mode"].astype(str).str.startswith("oat_")].copy()

    log("Writing CSV tables.")
    ordered = [
        "global_quality",
        "model_summary_by_mode",
        "baseline_model_summary",
        "baseline_paired_vs_instant",
        "taskwise_winners",
        "taskwise_winner_counts",
        "baseline_taskwise_winners",
        "baseline_taskwise_winner_counts",
        "oat_taskwise_winners",
        "regime_counts_percent",
        "oat_parameter_effects",
        "top_history_conditions",
        "top_oracle_conditions",
    ]
    for i, name in enumerate(ordered, 1):
        tables.get(name, pd.DataFrame()).to_csv(outdir / f"{i:02d}_{name}.csv", index=False)

    if raw_available:
        make_figures(ok, reg, outdir)
    else:
        log("Raw data unavailable; skipping task-level figures.")

    log("Writing report.")
    write_report(outdir, ok, tables, raw_available=raw_available)

    log("Done.")
    log(f"Report: {outdir / 'RESULTS_COMPLETE_REPORT.txt'}")


if __name__ == "__main__":
    main()
