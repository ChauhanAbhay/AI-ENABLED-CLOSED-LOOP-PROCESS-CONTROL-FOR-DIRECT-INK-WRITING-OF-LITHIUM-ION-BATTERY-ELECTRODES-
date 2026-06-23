# AI-Enabled Closed-Loop Process Control for Direct Ink Writing of Li-ion Battery Electrodes

Runnable Python implementation of the four-layer control architecture from the
CE52002 coursework (Singh, Ilobinso & Lucchina, 2026). Everything runs on
synthetic data calibrated to the operational ranges in Chapter 3 — no hardware,
no external dataset, no GPU required.

## The four layers

| Layer | Role | Module | Report section |
|-------|------|--------|----------------|
| 1. YOLOv8-nano visual inspector | reactive defect detection + rule-based correction | `diw/yolo.py` | 6.1.1, 7.2 |
| 2. BiLSTM rheological-drift predictor | 30 s-ahead filament-width forecast | `diw/bilstm.py` | 6.1.2 |
| 3. MPC setpoint solver | constrained actuation (print speed, substrate temp) | `diw/mpc.py` | 6.1.3, 7.3 |
| 4. Digital twin / closed loop | coordinates reactive + predictive paths | `diw/closed_loop.py` | 7.1, Fig 16 |

Supporting modules: `diw/config.py` (all constants traced to report tables),
`diw/synthetic.py` (rheology process model + dataset generator),
`diw/preprocess.py` (outlier clip, scaling, sliding windows).

## Quick start

```bash
pip install -r requirements.txt
python main.py --quick      # fast demo (~2-3 min on CPU)
python main.py              # faithful run: 40 epochs, more data
```

This trains the BiLSTM, compares it against a linear-regression baseline,
runs the closed loop in predictive vs reactive-only mode, and writes four
figures to `outputs/`:

- `bilstm_training.png` — training/validation loss with early stopping
- `forecast_vs_actual.png` — 30 s-ahead width forecast on the validation set
- `metric_comparison.png` — RMSE / R² / directional accuracy (gold = best)
- `closed_loop.png` — predictive vs reactive width control under a humidity ramp

## How the synthetic physics works

The process model (`DIWProcess`) reproduces the causal structure the report
relies on so the forecasting task is genuinely learnable:

- **humidity rise → leading indicator** of viscosity/width drift
- **pressure creep → lagging indicator** (follows humidity with its own lag)
- **temperature** → small thermal drift, large viscosity effect at high solids
- **width** responds to all three through a first-order transport lag (~4 s)

A deliberate mid-run humidity ramp drives a width excursion. The predictive
path (BiLSTM → MPC feed-forward) pre-empts it; the reactive-only path can only
respond after the bead has already drifted — which is exactly the gap the
report identifies in prior DIW work.

## Running the REAL vision model

`diw/yolo.py` contains `two_stage_train()`, a faithful reproduction of the
two-stage training in Appendix B (Stage 1 pre-train on real FDM, Stage 2
fine-tune on the hybrid dataset with the backbone frozen, layers 0-9). It
needs `pip install ultralytics` and the image datasets on disk. For the
dataset-free closed-loop demo, a `MockDetector` stands in, classifying the
bead from its width deviation the same way the detector's bounding boxes would.

## Notes / honest caveats

- On a small dataset and few epochs (`--quick`), the BiLSTM can underperform
  the linear baseline on held-out metrics — the same "indicative rather than
  conclusive" caveat the report makes (Section 6.1.2). More data + epochs
  improve it; the architecture and pipeline are the deliverable here.
- The MPC internal model is a transparent linear surrogate (the report uses an
  MLP/NN surrogate). Gains mirror the plant so the controller's world-model is
  consistent. Swap in a learned surrogate by replacing `b_speed`/`b_subtemp`
  in `mpc.py` with a trained model.
- All numeric constants live in `config.py` and are annotated with the table or
  section they come from, so the code stays faithful to the design.

## Layout

```
diw_control/
├── main.py              # end-to-end demo + figure generation
├── requirements.txt
├── diw/
│   ├── config.py        # constants (Tables 1-3)
│   ├── synthetic.py     # rheology process model + dataset
│   ├── preprocess.py    # clip / scale / window (Section 5.3)
│   ├── bilstm.py        # BiLSTM + baselines + metrics (6.1.2, 6.3)
│   ├── mpc.py           # receding-horizon QP solver (6.1.3)
│   ├── yolo.py          # two-stage training + mock detector + rules
│   └── closed_loop.py   # digital twin orchestration (Chapter 7)
└── outputs/             # generated figures
```
