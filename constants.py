from dataclasses import dataclass

@dataclass
class Config:
    seq_len: int
    pred_len: int
    n_channels: int
    lr: int


H_LIST = [24, 48, 72, 96, 120, 144, 168, 192, 336, 504, 672, 720]
T_LIST = [96, 192, 336, 720]

VAL_SIZE = 0.25
TEST_SIZE = 0.2

# from tuning
class NLinearTuneResult:
    best_lr = 0.00017278732761833284
    best_batchsize = 32