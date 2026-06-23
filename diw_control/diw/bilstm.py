"""
BiLSTM rheological-drift predictor  (Section 6.1.2, Appendix B Fig 18).

Architecture (faithful to the report):
    - two Bidirectional LSTM layers, 64 units each
    - Dropout 0.2 between layers
    - single Dense output node (filament-width deviation 30 s ahead)
    - MSE loss, Adam, early stopping (patience 8)

Metrics reported: RMSE (um), R^2, directional accuracy (Section 6.3).
"""
from __future__ import annotations
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

torch.set_num_threads(4)   # stable CPU performance in the sandbox

from .config import BILSTM, SEED


class BiLSTMRegressor(nn.Module):
    def __init__(self, cfg: BILSTM = BILSTM):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=cfg.input_features,
            hidden_size=cfg.hidden_units,
            num_layers=cfg.num_layers,
            batch_first=True,
            dropout=cfg.dropout,
            bidirectional=True,           # the "Bi" in BiLSTM
        )
        # *2 because bidirectional concatenates forward + backward states
        self.dropout = nn.Dropout(cfg.dropout)
        self.head = nn.Linear(cfg.hidden_units * 2, 1)

    def forward(self, x):
        # x: (batch, window, n_feat)
        out, _ = self.lstm(x)
        last = out[:, -1, :]              # final time-step representation
        return self.head(self.dropout(last)).squeeze(-1)


# --------------------------------------------------------------------------- #
# metrics
# --------------------------------------------------------------------------- #
def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def r2_score(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    return float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0


def directional_accuracy(y_true, y_pred, reference):
    """
    Fraction of forecasts that correctly predict the SIGN of the deviation
    from a reference (the last observed width). This is the operationally
    important metric: did we get the direction of drift right?
    """
    true_dir = np.sign(y_true - reference)
    pred_dir = np.sign(y_pred - reference)
    return float(np.mean(true_dir == pred_dir))


# --------------------------------------------------------------------------- #
# training
# --------------------------------------------------------------------------- #
def train_bilstm(data: dict, cfg: BILSTM = BILSTM, verbose: bool = True):
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- standardise the target (width ~250 um) so the loss is well-scaled.
    #     metrics are computed back in micron units after de-standardising.
    y_mean = float(data["y_train"].mean())
    y_std = float(data["y_train"].std()) or 1.0

    Xtr = torch.tensor(data["X_train"])
    ytr = torch.tensor((data["y_train"] - y_mean) / y_std)
    Xval = torch.tensor(data["X_val"]).to(device)
    yval_um = data["y_val"]

    loader = DataLoader(
        TensorDataset(Xtr, ytr),
        batch_size=cfg.batch_size, shuffle=True,
    )

    model = BiLSTMRegressor(cfg).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)
    loss_fn = nn.MSELoss()

    best_val = float("inf")
    best_state = None
    patience = 0
    history = {"train_loss": [], "val_loss": []}

    yval_std = torch.tensor((yval_um - y_mean) / y_std).to(device)

    for epoch in range(cfg.epochs):
        model.train()
        tr_losses = []
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()
            tr_losses.append(loss.item())

        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(Xval), yval_std).item()
        tr_loss = float(np.mean(tr_losses))
        history["train_loss"].append(tr_loss)
        history["val_loss"].append(val_loss)

        if val_loss < best_val - 1e-6:
            best_val = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1

        if verbose and (epoch % 5 == 0 or epoch == cfg.epochs - 1):
            print(f"  epoch {epoch:3d}  train_mse={tr_loss:7.4f}  "
                  f"val_mse={val_loss:7.4f}  patience={patience}")

        if patience >= cfg.early_stop_patience:
            if verbose:
                print(f"  early stop at epoch {epoch}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    # final evaluation — de-standardise predictions back to microns
    model.eval()
    with torch.no_grad():
        y_pred = model(Xval).cpu().numpy() * y_std + y_mean
    y_true = yval_um
    reference = np.full_like(y_true, y_true.mean())

    # stash the target scaler on the model for inference in the closed loop
    model._y_mean, model._y_std = y_mean, y_std

    metrics = {
        "rmse_um": rmse(y_true, y_pred),
        "r2": r2_score(y_true, y_pred),
        "directional_accuracy": directional_accuracy(y_true, y_pred, reference),
    }
    return model, history, metrics, y_pred


# --------------------------------------------------------------------------- #
# baselines for the three-way comparison (Fig 12-13)
# --------------------------------------------------------------------------- #
def linear_regression_baseline(data: dict):
    """Flatten windows and fit ordinary least squares — the report's baseline."""
    Xtr = data["X_train"].reshape(len(data["X_train"]), -1)
    Xval = data["X_val"].reshape(len(data["X_val"]), -1)
    ytr, yval = data["y_train"], data["y_val"]

    # closed-form OLS with bias term
    A = np.column_stack([np.ones(len(Xtr)), Xtr])
    coef, *_ = np.linalg.lstsq(A, ytr, rcond=None)
    Aval = np.column_stack([np.ones(len(Xval)), Xval])
    pred = Aval @ coef

    ref = np.full_like(yval, yval.mean())
    return {
        "rmse_um": rmse(yval, pred),
        "r2": r2_score(yval, pred),
        "directional_accuracy": directional_accuracy(yval, pred, ref),
    }
