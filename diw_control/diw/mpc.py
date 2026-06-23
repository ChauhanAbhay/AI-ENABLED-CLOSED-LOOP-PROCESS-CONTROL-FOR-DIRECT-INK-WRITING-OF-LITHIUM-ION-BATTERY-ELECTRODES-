"""
MPC setpoint solver  (Section 6.1.3, 7.3).

A receding-horizon controller that decides print speed and substrate
temperature to keep filament width at nominal, given:

  * the current measured width (from the YOLO/profilometer path), and
  * a 30 s-ahead width-drift FORECAST from the BiLSTM, injected as a measured
    disturbance (feed-forward) so the controller acts BEFORE the deviation
    appears in the vision stream.

Internal model (linearised, the report uses an MLP/NN surrogate; here a
transparent linear surrogate keeps it inspectable):

    w_{k+1} = w_k + b_speed * du_speed + b_subtemp * du_subtemp + d_k

where d_k is the per-step disturbance implied by the BiLSTM forecast.

The optimisation is a small dense QP solved with scipy SLSQP each control
step (5 steps at 1 Hz -> trivially within the Jetson compute budget).
"""
from __future__ import annotations
import numpy as np
from scipy.optimize import minimize

from .config import MPC, MEAS


class MPCSetpointSolver:
    def __init__(self, cfg: MPC = MPC):
        self.cfg = cfg
        self.w_nom = MEAS.filament_width_nominal_um   # 250 um setpoint

        # internal-model gains: width sensitivity to control moves (um per unit)
        # negative for speed (faster print -> thinner filament),
        # negative for substrate temp (hotter bed -> faster dry, thinner bead).
        self.b_speed = -3.5      # um per (mm/s) step
        self.b_subtemp = -1.2    # um per (degC) step

        # current actuator state (mid-range start)
        self.speed = 11.0        # mm/s   within 2-20
        self.subtemp = 42.0      # degC   within 25-60

    # ------------------------------------------------------------------ #
    def _predict_widths(self, du_seq, w0, disturbance_seq):
        """Roll the internal model forward over the horizon."""
        H = self.cfg.horizon
        du = du_seq.reshape(H, self.cfg.n_controls)
        w = w0
        widths = np.empty(H)
        for k in range(H):
            w = w + self.b_speed * du[k, 0] + self.b_subtemp * du[k, 1] \
                  + disturbance_seq[k]
            widths[k] = w
        return widths

    def _cost(self, du_seq, w0, disturbance_seq):
        cfg = self.cfg
        widths = self._predict_widths(du_seq, w0, disturbance_seq)
        err = widths - self.w_nom
        # stage tracking + terminal weight
        j_track = cfg.w_width * np.sum(err[:-1] ** 2) \
            + cfg.w_terminal * err[-1] ** 2
        j_effort = cfg.w_effort * np.sum(du_seq ** 2)
        return j_track + j_effort

    # ------------------------------------------------------------------ #
    def solve(self, w_measured: float, width_forecast: float | None = None):
        """
        Compute the next control move.

        w_measured     : current filament width from sensors (um)
        width_forecast : BiLSTM 30 s-ahead width prediction (um) or None.
                         The gap between forecast and nominal is spread across
                         the horizon as an anticipated disturbance.

        Returns dict with new absolute setpoints and the applied deltas.
        """
        cfg = self.cfg
        H = cfg.horizon

        # disturbance feed-forward: if the BiLSTM expects drift, pre-load it
        if width_forecast is not None:
            total_dist = width_forecast - w_measured
            disturbance_seq = np.full(H, total_dist / H)
        else:
            disturbance_seq = np.zeros(H)

        # decision variables: du for each step, each control
        x0 = np.zeros(H * cfg.n_controls)

        # box bounds on per-step deltas (actuator rate limits)
        bounds = []
        for _ in range(H):
            bounds.append((-cfg.max_dspeed_mm_s, cfg.max_dspeed_mm_s))
            bounds.append((-cfg.max_dsubtemp_c, cfg.max_dsubtemp_c))

        # absolute-range constraints: cumulative speed / subtemp stay in range
        def make_abs_constraints():
            cons = []
            for k in range(H):
                # speed within [2,20]
                def sp_lo(z, k=k):
                    du = z.reshape(H, cfg.n_controls)
                    return self.speed + du[:k + 1, 0].sum() - MEAS.print_speed_mm_s[0]
                def sp_hi(z, k=k):
                    du = z.reshape(H, cfg.n_controls)
                    return MEAS.print_speed_mm_s[1] - (self.speed + du[:k + 1, 0].sum())
                def st_lo(z, k=k):
                    du = z.reshape(H, cfg.n_controls)
                    return self.subtemp + du[:k + 1, 1].sum() - MEAS.substrate_temp_c[0]
                def st_hi(z, k=k):
                    du = z.reshape(H, cfg.n_controls)
                    return MEAS.substrate_temp_c[1] - (self.subtemp + du[:k + 1, 1].sum())
                cons += [
                    {"type": "ineq", "fun": sp_lo},
                    {"type": "ineq", "fun": sp_hi},
                    {"type": "ineq", "fun": st_lo},
                    {"type": "ineq", "fun": st_hi},
                ]
            return cons

        res = minimize(
            self._cost, x0,
            args=(w_measured, disturbance_seq),
            method="SLSQP",
            bounds=bounds,
            constraints=make_abs_constraints(),
            options={"maxiter": 50, "ftol": 1e-4},
        )

        du = res.x.reshape(H, cfg.n_controls)
        # receding horizon: apply only the FIRST move
        dspeed, dsubtemp = du[0, 0], du[0, 1]
        self.speed = float(np.clip(self.speed + dspeed, *MEAS.print_speed_mm_s))
        self.subtemp = float(np.clip(self.subtemp + dsubtemp, *MEAS.substrate_temp_c))

        return {
            "print_speed_mm_s": self.speed,
            "substrate_temp_c": self.subtemp,
            "d_speed": float(dspeed),
            "d_subtemp": float(dsubtemp),
            "predicted_cost": float(res.fun),
            "solver_ok": bool(res.success),
        }
