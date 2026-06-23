"""
YOLOv8-nano visual inspector  (Section 6.1.1, 6.2, 7.2).

Two parts:

1. `two_stage_train()` — the REAL training routine, faithful to Appendix B
   Figure 17. It needs `ultralytics` + an image dataset on disk, so it is
   guarded behind an import check and is what you would run on the Jetson /
   training rig. Stage 1 pre-trains on real FDM data; Stage 2 fine-tunes on
   the hybrid dataset with the backbone frozen (layers 0-9).

2. `MockDetector` — a self-contained detector that returns defect labels from
   the simulated process state, so the closed-loop demo runs with no dataset
   and no GPU. It maps width deviation -> defect class the same way the real
   detector's bounding boxes would.

`REACTIVE_RULES` is Table 4 (reactive rule-based control).
"""
from __future__ import annotations
import numpy as np

from .config import YOLO, MEAS


# --------------------------------------------------------------------------- #
# Reactive rule-based control  (Table 4, Section 7.2)
# --------------------------------------------------------------------------- #
REACTIVE_RULES = {
    "under_extrusion": {
        "cause": "Low pressure or high viscosity",
        "action": {"d_pressure_kpa": +5.0, "d_nozzle_temp_c": +1.0},
    },
    "over_extrusion": {
        "cause": "High pressure or low print speed",
        "action": {"d_pressure_kpa": -5.0, "d_print_speed_mm_s": +1.0},
    },
    "edge_delamination": {
        "cause": "Poor substrate adhesion",
        "action": {"d_nozzle_temp_c": +1.5},
    },
    "coiling": {
        "cause": "Speed/pressure mismatch",
        "action": {"d_print_speed_mm_s": -1.0},
    },
    "line_discontinuity": {
        "cause": "Nozzle clog / flow interruption",
        "action": {"d_pressure_kpa": +3.0},
    },
    "normal": {"cause": "—", "action": {}},
}


def apply_reactive_rule(defect: str) -> dict:
    """Return the actuation delta dict for a detected defect class."""
    return REACTIVE_RULES.get(defect, REACTIVE_RULES["normal"])["action"]


# --------------------------------------------------------------------------- #
# Mock detector (runnable, no dataset needed)
# --------------------------------------------------------------------------- #
class MockDetector:
    """
    Stand-in for YOLOv8-nano inference. Classifies the current frame from the
    true filament width: a wide bead reads as over-extrusion, a thin bead as
    under-extrusion, beyond the +/-5% tolerance band (Table 1).
    """

    def __init__(self, conf: float = 0.85):
        self.conf = conf
        self.nominal = MEAS.filament_width_nominal_um
        self.tol = MEAS.filament_width_tol_frac

    def detect(self, true_width_um: float, rng=None) -> dict:
        dev = (true_width_um - self.nominal) / self.nominal
        if dev > self.tol:
            label = "over_extrusion"
        elif dev < -self.tol:
            label = "under_extrusion"
        else:
            label = "normal"
        # emulate a bounding box + confidence
        return {
            "label": label,
            "confidence": self.conf,
            "deviation_frac": dev,
            "bbox": [0.4, 0.45, 0.6, 0.55],   # normalised xyxy, centre of frame
        }


# --------------------------------------------------------------------------- #
# Real two-stage training  (Appendix B Fig 17) — runs only with ultralytics
# --------------------------------------------------------------------------- #
def two_stage_train(cfg: YOLO = YOLO):
    """
    Faithful reproduction of the report's two-stage training. Requires
    `pip install ultralytics` and the datasets referenced in cfg on disk.
    """
    try:
        from ultralytics import YOLO as UltralyticsYOLO
    except ImportError as e:
        raise ImportError(
            "ultralytics not installed. Install with `pip install ultralytics` "
            "and provide the FDM + hybrid datasets to run real training. "
            "Use MockDetector for the dataset-free closed-loop demo."
        ) from e

    model = UltralyticsYOLO(cfg.weights)

    # Stage 1: domain pre-training on real FDM extrusion images
    model.train(
        data=cfg.stage1_data,
        epochs=cfg.stage1_epochs,
        imgsz=cfg.imgsz,
        batch=cfg.batch,
        optimizer="Adam",
        lr0=cfg.stage1_lr,
        device=cfg.device,
        name="stage1_fdm",
    )

    # Stage 2: fine-tune on hybrid dataset, backbone frozen (layers 0-9)
    model.train(
        data=cfg.stage2_data,
        epochs=cfg.stage2_epochs,
        imgsz=cfg.imgsz,
        batch=cfg.batch,
        optimizer="Adam",
        lr0=cfg.stage2_lr,
        freeze=cfg.freeze_layers,
        device=cfg.device,
        name="stage2_hybrid_finetune",
    )
    return model
