from dataclasses import dataclass


@dataclass
class Config:
    seq_len: int
    pred_len: int
    n_channels: int
    lr: int
    log_grad: bool = False


N_CHANNEL = 3

H_LIST = [24, 48, 72, 96, 120, 144, 168, 192, 336, 504, 672, 720]
T_LIST = [96, 192, 336, 720]

VAL_SIZE = 0.25
TEST_SIZE = 0.2

DATASETS = ['1d', '2d', '3d']

# from tuning
TUNE_RESULT = {
    'nlinear': {
        'lr': 0.000467785952240407,
        'batch_size': 64
    },
    'dlinear': {
        'lr': 0.00036642856489512316,
        'batch_size': 32
    },
    'tide-wo-a': {
        'lr': 0.00036490350684871407,
        'batch_size': 128,
        'hidden_dim': 64,
        'encoder_layer_num': 3,
        'decoder_layer_num': 1,
        'temporal_decoder_hidden': 32,
        'decoder_output_dim': 8,
        'dropout_rate': 0.2
    },
    'tide-w-a': {
        'lr': 0.00021678258939580435,
        'batch_size': 128,
        'hidden_dim': 128,
        'encoder_layer_num': 2,
        'decoder_layer_num': 2,
        'temporal_decoder_hidden': 64,
        'decoder_output_dim': 16,
        'dropout_rate': 0.1
    }
}
