"""
End-to-end demonstration of the four-layer DIW control system.

Run:  python main.py

Produces, on synthetic data calibrated to Chapter 3 ranges:
    1. trains the BiLSTM rheological-drift predictor + LR baseline (Ch.6)
    2. reports RMSE / R^2 / directional accuracy (Section 6.3)
    3. runs the closed-loop controller in predictive vs reactive-only modes
    4. saves figures to ./outputs/
"""
import os
import argparse
from dataclasses import replace
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from diw.config import BILSTM, MEAS
from diw.synthetic import make_dataset
from diw.preprocess import build_supervised
from diw.bilstm import train_bilstm, linear_regression_baseline
from diw.closed_loop import run_closed_loop

OUT = os.path.join(os.path.dirname(__file__), "outputs")
os.makedirs(OUT, exist_ok=True)


def main(n_runs=8, epochs=None, stride=10):
    cfg = BILSTM if epochs is None else replace(BILSTM, epochs=epochs)

    print("=" * 64)
    print("AI-ENABLED CLOSED-LOOP DIW PROCESS CONTROL — demo run")
    print("=" * 64)

    # ----- 1. data ----------------------------------------------------- #
    print("\n[1] Generating synthetic multivariate sensor dataset ...")
    ds = make_dataset(n_runs=n_runs, duration_s=120.0, dt_s=0.1)
    print(f"    total samples: {len(ds['width']):,}  "
          f"(pressure, nozzle_temp, humidity -> width)")

    data = build_supervised(ds, cfg, stride=stride)
    print(f"    train windows: {len(data['X_train']):,}  "
          f"val windows: {len(data['X_val']):,}  "
          f"window={cfg.window_len} samples (30 s)")

    # ----- 2. BiLSTM vs baseline -------------------------------------- #
    print("\n[2] Training BiLSTM rheological-drift predictor ...")
    model, history, metrics, y_pred = train_bilstm(data, cfg)

    print("\n[3] Linear-regression baseline ...")
    lr_metrics = linear_regression_baseline(data)

    print("\n--- Model comparison (30 s-ahead filament-width forecast) ---")
    print(f"{'model':<20}{'RMSE(um)':>10}{'R2':>10}{'DirAcc':>10}")
    print(f"{'Linear Regression':<20}{lr_metrics['rmse_um']:>10.3f}"
          f"{lr_metrics['r2']:>10.3f}{lr_metrics['directional_accuracy']:>10.3f}")
    print(f"{'BiLSTM (selected)':<20}{metrics['rmse_um']:>10.3f}"
          f"{metrics['r2']:>10.3f}{metrics['directional_accuracy']:>10.3f}")

    _plot_training(history)
    _plot_forecast(data["y_val"], y_pred)
    _plot_metric_bars(lr_metrics, metrics)

    # ----- 3. closed-loop: predictive vs reactive --------------------- #
    print("\n[4] Closed-loop simulation (predictive vs reactive-only) ...")
    log_pred = run_closed_loop(model, data["scaler"],
                               duration_s=120.0, predictive=True)
    log_react = run_closed_loop(None, None,
                                duration_s=120.0, predictive=False)

    nom = MEAS.filament_width_nominal_um
    err_pred = np.sqrt(np.mean((log_pred["width"] - nom) ** 2))
    err_react = np.sqrt(np.mean((log_react["width"] - nom) ** 2))
    print(f"    width RMSE vs nominal — predictive: {err_pred:6.2f} um | "
          f"reactive-only: {err_react:6.2f} um")
    improvement = 100 * (err_react - err_pred) / err_react
    print(f"    predictive control reduces width error by {improvement:.1f}%")

    _plot_closed_loop(log_pred, log_react)

    print(f"\nDone. Figures written to: {OUT}")
    print("Files:", ", ".join(sorted(os.listdir(OUT))))


# --------------------------------------------------------------------------- #
# plotting helpers
# --------------------------------------------------------------------------- #
def _plot_training(history):
    plt.figure(figsize=(7, 4))
    plt.plot(history["train_loss"], label="train MSE")
    plt.plot(history["val_loss"], label="val MSE")
    plt.xlabel("epoch"); plt.ylabel("MSE loss")
    plt.title("BiLSTM training curve (early stopping, patience 8)")
    plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(os.path.join(OUT, "bilstm_training.png"), dpi=130)
    plt.close()


def _plot_forecast(y_true, y_pred):
    plt.figure(figsize=(9, 4))
    plt.plot(y_true, label="actual", color="k", lw=1)
    plt.plot(y_pred, label="BiLSTM predicted", color="tab:blue", ls="--", lw=1.3)
    plt.xlabel("validation sample index"); plt.ylabel("filament width (um)")
    plt.title("30 s-ahead filament-width forecast — BiLSTM on validation set")
    plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(os.path.join(OUT, "forecast_vs_actual.png"), dpi=130)
    plt.close()


def _plot_metric_bars(lr, bilstm):
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6))
    names = ["Linear Reg.", "BiLSTM"]
    colours = ["tab:green", "tab:blue"]
    for ax, key, title, better in [
        (axes[0], "rmse_um", "RMSE (um)\n(lower better)", "low"),
        (axes[1], "r2", "R2\n(higher better)", "high"),
        (axes[2], "directional_accuracy", "Directional Acc.\n(higher better)", "high"),
    ]:
        vals = [lr[key], bilstm[key]]
        bars = ax.bar(names, vals, color=colours)
        best = np.argmin(vals) if better == "low" else np.argmax(vals)
        bars[best].set_edgecolor("gold"); bars[best].set_linewidth(3)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.3f}",
                    ha="center", va="bottom", fontsize=9)
        ax.set_title(title); ax.grid(alpha=0.3, axis="y")
    fig.suptitle("Model performance comparison (gold = best per metric)")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "metric_comparison.png"), dpi=130)
    plt.close()


def _plot_closed_loop(pred, react):
    nom = MEAS.filament_width_nominal_um
    tol = nom * MEAS.filament_width_tol_frac
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)

    ax = axes[0]
    ax.plot(react["t"], react["width"], color="tab:red", alpha=0.8,
            label="reactive-only width")
    ax.plot(pred["t"], pred["width"], color="tab:blue", alpha=0.9,
            label="predictive (BiLSTM+MPC) width")
    ax.axhline(nom, color="k", ls="--", lw=1, label="nominal 250 um")
    ax.axhspan(nom - tol, nom + tol, color="green", alpha=0.08,
               label="+/-5% tolerance")
    ax.set_ylabel("filament width (um)")
    ax.set_title("Closed-loop control: predictive vs reactive-only")
    ax.legend(loc="upper left", fontsize=8); ax.grid(alpha=0.3)

    ax = axes[1]
    ax.plot(pred["t"], pred["humidity"], color="tab:cyan",
            label="humidity (%RH) — leading driver")
    ax2 = ax.twinx()
    ax2.plot(pred["t"], pred["print_speed"], color="tab:purple",
             label="print speed (mm/s) — MPC")
    ax2.plot(pred["t"], pred["substrate_temp"], color="tab:orange",
             label="substrate temp (degC) — MPC")
    ax.set_xlabel("time (s)"); ax.set_ylabel("humidity (%RH)")
    ax2.set_ylabel("MPC setpoints")
    ax.grid(alpha=0.3)
    lines1, lab1 = ax.get_legend_handles_labels()
    lines2, lab2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, lab1 + lab2, loc="upper left", fontsize=8)

    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "closed_loop.png"), dpi=130)
    plt.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="DIW four-layer closed-loop control demo")
    ap.add_argument("--quick", action="store_true",
                    help="fast run: fewer epochs/runs for a quick check")
    ap.add_argument("--epochs", type=int, default=None,
                    help="override BiLSTM epochs (default 40 from config)")
    ap.add_argument("--runs", type=int, default=8,
                    help="number of synthetic print runs to generate")
    ap.add_argument("--stride", type=int, default=10,
                    help="sliding-window stride (1 = report's max overlap)")
    args = ap.parse_args()

    if args.quick:
        main(n_runs=4, epochs=12, stride=15)
    else:
        main(n_runs=args.runs, epochs=args.epochs, stride=args.stride)
