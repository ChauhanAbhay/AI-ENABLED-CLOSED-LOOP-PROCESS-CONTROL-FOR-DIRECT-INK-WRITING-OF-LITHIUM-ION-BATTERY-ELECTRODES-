"""
Synthetic DIW process model.

A lightweight rheology-inspired simulator that produces multivariate sensor
streams (pressure, nozzle temperature, humidity) and the resulting filament
width. It reproduces the causal structure the report relies on:

  * humidity rise  -> LEADING indicator of viscosity / width drift
  * pressure creep -> LAGGING indicator (agglomeration onset)
  * temperature    -> small thermal drift, large viscosity effect at high solids
  * width responds to all three with a transport lag (~3-5 s), so a 30 s-ahead
    forecast is genuinely learnable rather than noise.

This is the data source for the BiLSTM (Chapter 5) and the "true plant" the
closed-loop simulation actuates against (Chapter 7).
"""
from __future__ import annotations
import numpy as np

from .config import MEAS, BILSTM, SEED


class DIWProcess:
    """First-order rheology surrogate for a single print run."""

    def __init__(self, dt_s: float = 0.1, seed: int = SEED):
        self.dt = dt_s
        self.rng = np.random.default_rng(seed)

        # --- nominal operating point (mid-range of Table 1) ---
        self.p0 = 90.0      # kPa   baseline extrusion pressure
        self.t0 = 30.0      # degC  baseline nozzle temperature
        self.h0 = 45.0      # %RH   baseline humidity
        self.w0 = MEAS.filament_width_nominal_um   # 250 um nominal width

        # --- sensitivity coefficients (width um per unit deviation) ---
        # tuned so a realistic env drift moves width by tens of microns
        self.k_press = 0.45     # um per kPa above baseline
        self.k_temp = 6.0       # um per degC (strong, high-solids regime)
        self.k_hum = 0.9        # um per %RH

        # --- transport lag: width follows a filtered driver signal ---
        self.tau_w = 4.0        # s, first-order width response time-constant
        self._width_state = self.w0

        # --- actuator coupling: MPC controls (print speed, substrate temp)
        #     shift the width set-point. These mirror the MPC internal-model
        #     gains in mpc.py so the controller's world-model is correct.
        self.speed_nom = 11.0   # mm/s reference operating speed
        self.subtemp_nom = 42.0 # degC reference substrate temperature
        self.g_speed = -3.5     # um per (mm/s) above nominal
        self.g_subtemp = -1.2   # um per degC above nominal
        self.actuator_bias = 0.0  # current width bias from control setpoints

        # --- humidity leads pressure: pressure creep follows humidity ---
        self.tau_p = 8.0        # s, pressure-creep lag behind humidity
        self._press_creep = 0.0

    # ------------------------------------------------------------------ #
    def set_actuators(self, print_speed_mm_s: float, substrate_temp_c: float):
        """Apply MPC setpoints; updates the steady-state width bias."""
        self.actuator_bias = (
            self.g_speed * (print_speed_mm_s - self.speed_nom)
            + self.g_subtemp * (substrate_temp_c - self.subtemp_nom)
        )

    def _driver(self, p, t, h):
        """Instantaneous width target from current sensor values + actuators."""
        return (
            self.w0
            + self.k_press * (p - self.p0)
            + self.k_temp * (t - self.t0)
            + self.k_hum * (h - self.h0)
            + self.actuator_bias
        )

    def step(self, p, t, h):
        """Advance one dt; return the realised (lagged + noisy) width."""
        target = self._driver(p, t, h)
        alpha = self.dt / (self.tau_w + self.dt)
        self._width_state += alpha * (target - self._width_state)
        noise = self.rng.normal(0.0, 1.2)        # measurement noise (um)
        return self._width_state + noise

    # ------------------------------------------------------------------ #
    def generate_run(self, duration_s: float = 120.0):
        """
        Generate one correlated multivariate run.

        Returns dict of arrays: t, pressure, nozzle_temp, humidity, width.
        The environmental scenario embeds a slow humidity ramp that LEADS a
        pressure creep that LEADS a width excursion — the exact structure the
        BiLSTM is meant to exploit.
        """
        n = int(duration_s / self.dt)
        t_axis = np.arange(n) * self.dt

        # --- humidity: slow random walk + a deliberate mid-run ramp ---
        hum = np.full(n, self.h0)
        walk = np.cumsum(self.rng.normal(0, 0.05, n))
        ramp = 12.0 / (1 + np.exp(-(t_axis - 55) / 6.0))   # leading rise ~55 s
        hum = np.clip(self.h0 + walk + ramp, *MEAS.humidity_rh)

        # --- nozzle temperature: sinusoidal heater cycling + noise ---
        temp = (
            self.t0
            + 0.8 * np.sin(2 * np.pi * t_axis / 25.0)       # heater cycle
            + self.rng.normal(0, 0.15, n)
        )
        temp = np.clip(temp, *MEAS.nozzle_temp_c)

        # --- pressure: baseline + creep that LAGS humidity + transients ---
        press = np.empty(n)
        creep = 0.0
        for i in range(n):
            # creep driven by humidity excess, with its own lag
            hum_excess = hum[i] - self.h0
            alpha_p = self.dt / (self.tau_p + self.dt)
            creep += alpha_p * (0.6 * hum_excess - creep)
            spike = 0.0
            if self.rng.random() < 0.002:                   # rare agglomeration
                spike = self.rng.uniform(3, 8)
            press[i] = np.clip(
                self.p0 + creep + spike + self.rng.normal(0, 0.4),
                *MEAS.extrusion_pressure_kpa,
            )

        # --- width: lagged response to all drivers ---
        self._width_state = self.w0
        width = np.empty(n)
        for i in range(n):
            width[i] = self.step(press[i], temp[i], hum[i])

        return {
            "t": t_axis,
            "pressure": press,
            "nozzle_temp": temp,
            "humidity": hum,
            "width": width,
        }


def make_dataset(n_runs: int = 10, duration_s: float = 120.0,
                 dt_s: float = 0.1, seed: int = SEED):
    """
    Concatenate several runs into one long multivariate series, mirroring the
    "1,200 synthetic samples at 0.1 s intervals" dataset in Section 5.1
    (here scaled up across runs for enough sliding windows to train on).
    """
    runs = []
    for r in range(n_runs):
        proc = DIWProcess(dt_s=dt_s, seed=seed + r)
        runs.append(proc.generate_run(duration_s))
    keys = ["pressure", "nozzle_temp", "humidity", "width"]
    merged = {k: np.concatenate([run[k] for run in runs]) for k in keys}
    merged["run_id"] = np.concatenate(
        [np.full(len(run["width"]), i) for i, run in enumerate(runs)]
    )
    return merged
