import torch
import torch.nn as nn
from dataclasses import dataclass

torch.set_default_dtype(torch.float32)

from models.fdnet.convblock import ConvBlock
from models.fdnet.embed import DataEmbedding
from models.base import ModelBehavior
from constants import Config


class Decomposed_block(nn.Module):

    def __init__(self, enc_in, d_model, seq_kernel, dropout, attn_nums, ICOM, h_nums, label_len,
                 pred_len, timebed):
        super(Decomposed_block, self).__init__()
        self.enc_in = enc_in - timebed
        self.pred_len = pred_len
        self.embed = DataEmbedding(d_model, dropout)
        self.ICOM = ICOM
        if self.ICOM:
            pro_conv2d = [
                ConvBlock(d_model, d_model, seq_kernel, ICOM, pool=True, dropout=dropout)
                for _ in range(h_nums)
            ]
        else:
            pro_conv2d = [
                ConvBlock(d_model, d_model, seq_kernel, ICOM, pool=False, dropout=dropout)
                for _ in range(attn_nums)
            ]
        self.pro_conv2d = nn.ModuleList(pro_conv2d)

        self.F = nn.Flatten(start_dim=2)
        final_len = label_len // (2**h_nums) if ICOM else label_len
        self.FC = nn.Linear(final_len * d_model, pred_len)

        self.timebed = timebed
        if self.timebed:
            self.time_layer = nn.Linear(self.timebed, self.enc_in * d_model)

    def forward(self, x):
        if self.timebed:
            time = x[:, :, -self.timebed:, :]
            x = x[:, :, :-self.timebed, :]
            x = self.embed(x)
            B, C, S, V = x.shape
            time = self.time_layer(time[:, :, :, 0]).contiguous().view(B, S, V,
                                                                       C).permute(0, 3, 1, 2)
            x = x + time
        else:
            x = self.embed(x)

        x_2d = x.clone()
        for conv2d in self.pro_conv2d:
            x_2d = conv2d(x_2d)
        x_2d_out = self.F(x_2d.transpose(1, -1))
        x_out = self.FC(x_2d_out).transpose(1, 2)
        return x_out


@dataclass(kw_only=True)
class FDNetConfig(Config):
    seq_kernel: int = 3
    attn_nums: int = 3
    d_model: int = 64
    pyramid: int = 1
    ICOM: bool = False
    dropout: float = 0
    timebed: str = 'None'
    # seq_len
    label_len: int = None
    enc_in: int = None
    c_out: int = None

    def __post_init__(self):
        if self.label_len is None:
            self.label_len = self.seq_len
        
        if self.enc_in is None:
            self.enc_in = self.n_channels

        if self.c_out is None:
            self.c_out = self.n_channels


class FDNet(ModelBehavior):

    def __init__(self, config: FDNetConfig):
        super(FDNet, self).__init__(config)
        type_bed = {'None': 0, 'hour': 1, 'day': 1, 'year': 6, 'year_min': 7}
        timebed = int(type_bed[config.timebed])
        self.enc_in = config.enc_in
        self.timebed = timebed
        self.pyramid = config.pyramid

        self.label_len = config.label_len
        self.pred_len = config.pred_len
        self.c_out = config.c_out
        self.d_model = config.d_model



        FDNet_blocks = [Decomposed_block(config.enc_in, config.d_model, config.seq_kernel, config.dropout, config.attn_nums, config.ICOM, 1,
                                         config.label_len // (2 ** config.pyramid), config.pred_len, self.timebed)] + \
                       [Decomposed_block(config.enc_in, config.d_model, config.seq_kernel, config.dropout, config.attn_nums - i, config.ICOM, i + 1,
                                         config.label_len // (2 ** (config.pyramid - i)), config.pred_len, self.timebed)
                        for i in range(config.pyramid + 1)]
        self.FDNet_blocks = nn.ModuleList(FDNet_blocks)

    def forward(self, x_enc):
        enc_input = x_enc[:, :self.label_len, :]
        enc_input_list = [enc_input[:, -self.label_len // (2**self.pyramid):, :]]

        enc_out = 0
        num_output = 0
        for i in range(self.pyramid):
            enc_input_list.append(
                enc_input[:, -self.label_len // (2**(self.pyramid - i - 1)):-self.label_len //
                          (2**(self.pyramid - i)), :])
        for curr_input, FD_b in zip(enc_input_list, self.FDNet_blocks):
            enc_out += FD_b(curr_input.unsqueeze(-1))
            num_output += 1

        return enc_out / num_output