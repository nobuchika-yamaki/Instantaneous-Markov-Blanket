#!/bin/bash
set -euo pipefail

PY_SCRIPT="${1:-$HOME/Downloads/active_inference_temporal_blanket_multitask.py}"
ROOT_OUT="${2:-$HOME/Desktop/ai_blanket_multitask_analysis}"
N_JOBS="${N_JOBS:-4}"

mkdir -p "$ROOT_OUT"

MODES=(
  baseline
  oat_mu
  oat_chi
  oat_sigmaE
  oat_noise
  oat_lambdaS
  oat_genK
  robustness_nonlinear
)

echo "=========================================="
echo "MULTI-TARGET BASELINE + OAT RUN"
echo "=========================================="
echo "PY_SCRIPT=$PY_SCRIPT"
echo "ROOT_OUT=$ROOT_OUT"
echo "N_JOBS=$N_JOBS"
echo

for MODE in "${MODES[@]}"; do
  OUTDIR="$ROOT_OUT/$MODE"
  mkdir -p "$OUTDIR"
  echo
  echo "------------------------------------------"
  echo "Running mode: $MODE"
  echo "Output: $OUTDIR"
  echo "------------------------------------------"

  caffeinate -dimsu python3 -u "$PY_SCRIPT" \
    --mode "$MODE" \
    --n_jobs "$N_JOBS" \
    --resume \
    --outdir "$OUTDIR" \
    2>&1 | tee "$OUTDIR/${MODE}.log"
done

echo
echo "------------------------------------------"
echo "Combining multi-target summaries"
echo "------------------------------------------"

ROOT_OUT_FOR_PY="$ROOT_OUT" python3 - <<'PY'
import os
import pandas as pd

root = os.path.expanduser(os.environ["ROOT_OUT_FOR_PY"])
modes = [
    "baseline",
    "oat_mu",
    "oat_chi",
    "oat_sigmaE",
    "oat_noise",
    "oat_lambdaS",
    "oat_genK",
    "robustness_nonlinear",
]

frames = []
regs = []
models = []
multi = []

for mode in modes:
    out = os.path.join(root, mode)
    for fname, store in [
        ("all_results_long.csv", frames),
        ("regime_summary.csv", regs),
        ("model_comparison_summary.csv", models),
        ("multitarget_model_summary.csv", multi),
    ]:
        p = os.path.join(out, fname)
        if os.path.exists(p):
            df = pd.read_csv(p)
            df["analysis_mode"] = mode
            store.append(df)

combined = os.path.join(root, "_combined")
os.makedirs(combined, exist_ok=True)

if frames:
    all_df = pd.concat(frames, ignore_index=True)
    all_df.to_csv(os.path.join(combined, "combined_all_results_long.csv"), index=False)

if regs:
    reg_df = pd.concat(regs, ignore_index=True)
    reg_df.to_csv(os.path.join(combined, "combined_regime_summary.csv"), index=False)
    reg_counts = reg_df.groupby(["analysis_mode", "region"]).size().reset_index(name="count")
    reg_counts.to_csv(os.path.join(combined, "combined_regime_counts.csv"), index=False)

if models:
    model_df = pd.concat(models, ignore_index=True)
    model_df.to_csv(os.path.join(combined, "combined_model_comparison_summary.csv"), index=False)

if multi:
    mt_df = pd.concat(multi, ignore_index=True)
    mt_df.to_csv(os.path.join(combined, "combined_multitarget_model_summary.csv"), index=False)

if frames:
    ok = all_df[all_df["status"] == "ok"].copy()
    metric_cols = [
        "MSE_E", "CMI_delta",
        "MSE_I_current", "RI_I_current",
        "MSE_S_delay", "RI_S_delay",
        "MSE_E_future_0p2", "RI_E_future_0p2",
        "MSE_E_future_0p8", "RI_E_future_0p8",
        "MSE_E_future_1p6", "RI_E_future_1p6",
        "MSE_A_future_0p2", "RI_A_future_0p2",
        "MSE_A_future_0p8", "RI_A_future_0p8",
        "MSE_A_future_1p6", "RI_A_future_1p6",
        "clip_mu_fraction", "clip_a_fraction",
    ]
    metric_cols = [c for c in metric_cols if c in ok.columns]
    summary = ok.groupby(["analysis_mode", "model"], as_index=False)[metric_cols].mean(numeric_only=True)
    summary.to_csv(os.path.join(combined, "combined_multitarget_results_summary.csv"), index=False)

    report = os.path.join(combined, "combined_multitarget_report.txt")
    with open(report, "w") as f:
        f.write("Combined multi-target active-inference blanket report\n")
        f.write("====================================================\n\n")
        f.write("Status counts\n")
        f.write("-------------\n")
        f.write(all_df["status"].value_counts(dropna=False).to_string())
        f.write("\n\nRows per model\n")
        f.write("--------------\n")
        f.write(all_df["model"].value_counts().to_string())
        f.write("\n\nMulti-target model summary\n")
        f.write("--------------------------\n")
        f.write(summary.to_string(index=False))
        if regs:
            f.write("\n\nRegime counts\n")
            f.write("-------------\n")
            f.write(reg_counts.to_string(index=False))
        f.write("\n")
    print("combined report:", report)

print("combined dir:", combined)
PY

echo
echo "DONE"
echo "Root output:"
echo "$ROOT_OUT"
echo "Combined report:"
echo "$ROOT_OUT/_combined/combined_multitarget_report.txt"
