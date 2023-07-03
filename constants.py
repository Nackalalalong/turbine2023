from dataclasses import dataclass

@dataclass
class Config:
    seq_len: int
    pred_len: int
    n_channels: int
    lr: int
    log_grad: bool = False


H_LIST = [24, 48, 72, 96, 120, 144, 168, 192, 336, 504, 672, 720]
T_LIST = [96, 192, 336, 720]

VAL_SIZE = 0.25
TEST_SIZE = 0.2

DATASETS = ['1d', '2d', '3d']
MODELS = ['nlinear', 'dlinear', 'jtft']

# from tuning
class NLinearTuneResult:
    best_lr = 0.000467785952240407
    best_batchsize = 64


class DLinearTuneResult:
    best_lr = 0.00036642856489512316
    best_batchsize = 32