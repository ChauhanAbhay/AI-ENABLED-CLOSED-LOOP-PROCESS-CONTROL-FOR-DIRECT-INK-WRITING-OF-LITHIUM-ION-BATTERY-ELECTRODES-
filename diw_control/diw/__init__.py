"""
diw_control — AI-enabled closed-loop process control for Direct Ink Writing
of lithium-ion battery electrodes.

Four-layer architecture (Singh, Ilobinso & Lucchina, 2026):
    Layer 1  YOLOv8-nano visual inspector      -> reactive defect correction
    Layer 2  BiLSTM rheological-drift predictor -> 30 s-ahead width forecast
    Layer 3  MPC setpoint solver               -> constrained actuation
    Layer 4  Digital twin / closed-loop bus    -> coordinates the above

This package is self-contained and runs on synthetic data calibrated to the
operational ranges in Chapter 3 of the report.
"""

__version__ = "1.0.0"

from . import config

__all__ = ["config", "__version__"]
