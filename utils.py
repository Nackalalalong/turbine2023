from dataclasses import dataclass

@dataclass
class Config:
    seq_len: int
    pred_len: int
    n_channels: int