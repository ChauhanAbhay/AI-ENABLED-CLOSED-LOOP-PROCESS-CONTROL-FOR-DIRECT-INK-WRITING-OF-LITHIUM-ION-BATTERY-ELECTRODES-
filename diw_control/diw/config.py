"""
Central configuration.

Every constant here is traceable to a specific table/section in the report so
the code stays faithful to the design rather than inventing numbers.
"""
from dataclasses import dataclass, field


# --------------------------------------------------------------------------- #
# Measurand operating ranges  (Table 1, Section 3.2)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Measurands:
    extrusion_pressure_kpa: tuple = (20.0, 200.0)     # BiLSTM primary input
    nozzle_temp_c: tuple = (22.0, 40.0)               # BiLSTM input / MPC actuator
    print_speed_mm_s: tuple = (2.0, 20.0)             # MPC setpoint
    substrate_temp_c: tuple = (25.0, 60.0)            # MPC setpoint
    humidity_rh: tuple = (30.0, 60.0)                 # BiLSTM secondary input
    filament_width_nominal_um: float = 250.0          # YOLO target / BiLSTM target
    filament_width_tol_frac: float = 0.05             # +/-5% nominal (quality metric)
    coating_thickness_tol_um: float = 3.0             # digital-twin areal projection


# --------------------------------------------------------------------------- #
# Sensor / DAQ sample rates  (Table 2 + Section 4.2)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class SensorRates:
    pressure_hz: int = 1000        # Honeywell MLH
    accel_hz: int = 5000           # Kistler 8762A
    profilometer_hz: int = 800     # Keyence LJ-X8200
    camera_fps: int = 90           # Basler acA2040 (30 fps to YOLO)
    yolo_fps: int = 30
    temp_humidity_hz: int = 10     # PT100 array + Sensirion SHT45
    bilstm_input_hz: int = 10      # streams down-sampled to 10 Hz for the BiLSTM


# --------------------------------------------------------------------------- #
# BiLSTM hyperparameters  (Section 6.1.2 + Table 3 + Appendix B Fig 18)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class BiLSTMConfig:
    input_features: int = 3        # pressure, nozzle-temp, humidity
    hidden_units: int = 64         # two layers of 64 units each
    num_layers: int = 2
    dropout: float = 0.2
    lookback_s: int = 30           # 30 s look-back window
    horizon_s: int = 30            # predict 30 s ahead
    sample_hz: int = 10            # 10 Hz -> 300 samples / window
    learning_rate: float = 1e-3    # Adam
    epochs: int = 40
    batch_size: int = 32
    early_stop_patience: int = 8
    train_frac: float = 0.8        # 80/20 chronological split (no shuffle)

    @property
    def window_len(self) -> int:
        return self.lookback_s * self.sample_hz   # 300

    @property
    def horizon_len(self) -> int:
        return self.horizon_s * self.sample_hz    # 300


# --------------------------------------------------------------------------- #
# MPC setpoint solver  (Section 6.1.3 + 7.3)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class MPCConfig:
    horizon: int = 5               # 5 control intervals at 1 Hz (receding horizon)
    dt_s: float = 1.0              # 1 Hz control rate
    # control variables: print speed, substrate temperature
    n_controls: int = 2
    # cost weights
    w_width: float = 1.0           # tracking nominal filament width
    w_effort: float = 0.05         # penalty on control effort (delta-u)
    w_terminal: float = 2.0        # terminal width weight
    # actuator rate limits per step
    max_dspeed_mm_s: float = 2.0
    max_dsubtemp_c: float = 3.0


# --------------------------------------------------------------------------- #
# YOLOv8-nano two-stage training  (Section 6.2 + Table 3 + Appendix B Fig 17)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class YOLOConfig:
    weights: str = "yolov8n.pt"
    imgsz: int = 416
    batch: int = 8
    device: str = "cuda"
    # Stage 1: domain pre-training on real FDM
    stage1_epochs: int = 30
    stage1_lr: float = 1e-3
    stage1_data: str = "fdm_sample/data.yaml"
    # Stage 2: fine-tune on hybrid, backbone frozen (layers 0-9)
    stage2_epochs: int = 30
    stage2_lr: float = 1e-4
    stage2_data: str = "hybrid_dataset/data.yaml"
    freeze_layers: int = 10
    map50_target: float = 0.85
    classes: tuple = (
        "normal", "under_extrusion", "over_extrusion",
        "coiling", "edge_delamination", "line_discontinuity",
    )


# convenience singletons
MEAS = Measurands()
RATES = SensorRates()
BILSTM = BiLSTMConfig()
MPC = MPCConfig()
YOLO = YOLOConfig()

SEED = 42
