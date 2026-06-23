"""
Digital twin + closed-loop controller  (Chapter 7, Figure 16).

Coordinates the four layers each control tick:

    1. read sensors from the (synthetic) plant
    2. YOLO/MockDetector inspects the current bead   -> REACTIVE rule deltas
    3. BiLSTM forecasts width 30 s ahead             -> PREDICTIVE disturbance
    4. MPC solves for print-speed / substrate-temp   -> PREEMPTIVE setpoints
    5. apply both reactive + predictive actions to the plant, log everything

The digital twin holds the running state and maps detections to corrections,
harmonising the two control paths as in the supervisory layer of Figure 16.
"""
from __future__ import annotations
import numpy as np
import torch

from .config import MPC, BILSTM, MEAS
from .synthetic import DIWProcess
from .preprocess import MinMaxScaler
from .mpc import MPCSetpointSolver
from .yolo import MockDetector, apply_reactive_rule


class DigitalTwin:
    """Holds process state and coordinates reactive + predictive corrections."""

    def __init__(self, bilstm_model=None, scaler: MinMaxScaler | None = None):
        self.detector = MockDetector()
        self.mpc = MPCSetpointSolver()
        self.bilstm = bilstm_model
        self.scaler = scaler
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        if self.bilstm is not None:
            self.bilstm.to(self.device).eval()

    # ------------------------------------------------------------------ #
    def forecast_width(self, window_feats: np.ndarray) -> float | None:
        """
        Run the BiLSTM on a (window_len, 3) array of recent raw sensor values.
        Returns the 30 s-ahead width prediction in microns, or None if no
        model is loaded.
        """
        if self.bilstm is None or self.scaler is None:
            return None
        scaled = self.scaler.transform(window_feats).astype(np.float32)
        x = torch.tensor(scaled[None, :, :]).to(self.device)
        with torch.no_grad():
            raw = float(self.bilstm(x).cpu().item())
        # de-standardise to microns using the scaler stashed during training
        y_mean = getattr(self.bilstm, "_y_mean", 0.0)
        y_std = getattr(self.bilstm, "_y_std", 1.0)
        return raw * y_std + y_mean


def run_closed_loop(bilstm_model=None, scaler=None,
                    duration_s: float = 120.0, seed: int = 7,
                    predictive: bool = True):
    """
    Simulate one closed-loop print run.

    The plant is a DIWProcess instance. A deliberate humidity ramp drives a
    width excursion mid-run; the predictive path should pre-empt it while a
    purely reactive path can only respond after the bead has already drifted.

    Returns a log dict of time series for plotting / analysis.
    """
    dt = 1.0 / BILSTM.sample_hz                 # 0.1 s sensor tick (10 Hz)
    plant = DIWProcess(dt_s=dt, seed=seed)
    twin = DigitalTwin(bilstm_model, scaler)

    n = int(duration_s / dt)
    window = BILSTM.window_len                  # 300 samples = 30 s

    # pre-generate the environmental scenario (humidity/temp), but pressure &
    # width evolve under control feedback, so we step the plant manually.
    base = plant.generate_run(duration_s)
    humidity = base["humidity"]
    nozzle_temp = base["nozzle_temp"].copy()

    # control state
    pressure_offset = 0.0       # reactive pressure correction (kPa)
    temp_offset = 0.0           # reactive nozzle-temp correction (degC)

    log = {k: [] for k in (
        "t", "width", "pressure", "nozzle_temp", "humidity",
        "forecast", "print_speed", "substrate_temp", "defect", "mode",
    )}

    plant._width_state = plant.w0
    sensor_hist = np.zeros((n, 3), dtype=np.float32)   # press, temp, hum

    control_every = int(MPC.dt_s / dt)          # MPC runs at 1 Hz -> every 10 ticks

    for i in range(n):
        # --- assemble current sensor sample ---
        press = float(np.clip(base["pressure"][i] + pressure_offset,
                              *MEAS.extrusion_pressure_kpa))
        temp = float(np.clip(nozzle_temp[i] + temp_offset,
                             *MEAS.nozzle_temp_c))
        hum = float(humidity[i])
        sensor_hist[i] = [press, temp, hum]

        # --- step the true plant ---
        width = plant.step(press, temp, hum)

        # --- layer 1: reactive vision inspection every tick ---
        det = twin.detector.detect(width)
        rule = apply_reactive_rule(det["label"])
        pressure_offset += rule.get("d_pressure_kpa", 0.0) * 0.1   # gentle
        temp_offset += rule.get("d_nozzle_temp_c", 0.0) * 0.1

        # --- layers 2-3: predictive forecast + MPC at 1 Hz ---
        forecast = None
        if i % control_every == 0 and i >= window and predictive:
            win = sensor_hist[i - window:i]
            forecast = twin.forecast_width(win)
            sol = twin.mpc.solve(w_measured=width, width_forecast=forecast)
            speed = sol["print_speed_mm_s"]
            subtemp = sol["substrate_temp_c"]
            # close the loop: push the new setpoints into the plant so the
            # bead actually responds to print speed and substrate temperature.
            plant.set_actuators(speed, subtemp)
        else:
            speed = twin.mpc.speed
            subtemp = twin.mpc.subtemp

        # decay reactive offsets so they don't wind up
        pressure_offset *= 0.98
        temp_offset *= 0.98

        log["t"].append(i * dt)
        log["width"].append(width)
        log["pressure"].append(press)
        log["nozzle_temp"].append(temp)
        log["humidity"].append(hum)
        log["forecast"].append(forecast if forecast is not None else np.nan)
        log["print_speed"].append(speed)
        log["substrate_temp"].append(subtemp)
        log["defect"].append(det["label"])
        log["mode"].append("predictive" if predictive else "reactive")

    return {k: np.array(v) if k not in ("defect", "mode") else v
            for k, v in log.items()}
