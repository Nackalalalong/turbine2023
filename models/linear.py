# main.py
import torch
import torch.nn as nn

from models.base import ModelBehavior
from constants import Config

torch.set_default_dtype(torch.float32)

class NLinear(ModelBehavior):
    """
    adapts from https://github.com/cure-lab/LTSF-Linear/blob/main/models/NLinear.py
    """

    def __init__(self, config: Config):
        super(NLinear, self).__init__(config)

        self.linear = nn.ModuleList()
        for _ in range(self.n_channels):
            self.linear.append(nn.Linear(self.seq_len, self.pred_len, dtype=torch.float64))

    def forward(self, x):
        # x: [Batch, Input length, Channel]
        seq_last = x[:, -1:, :].detach()
        x = x - seq_last

        output = torch.zeros([x.size(0), self.pred_len, x.size(2)], dtype=x.dtype).to(x.device)
        for i in range(self.n_channels):
            output[:, :, i] = self.linear[i](x[:, :, i])
        x = output

        x = x + seq_last
        return x  # [Batch, Output length, Channel]


####### DLinear ##############
# from https://github.com/cure-lab/LTSF-Linear/blob/main/models/DLinear.py


class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """

    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # padding on the both ends of time series
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x


class series_decomp(nn.Module):
    """
    Series decomposition block
    """

    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean


class DLinear(ModelBehavior):
    """
    Decomposition-Linear
    """

    def __init__(self, config: Config):
        super(DLinear, self).__init__(config)

        # Decompsition Kernel Size
        self.kernel_size = 25
        self.decompsition = series_decomp(self.kernel_size)

        self.Linear_Seasonal = nn.ModuleList()
        self.Linear_Trend = nn.ModuleList()

        for i in range(self.n_channels):
            self.Linear_Seasonal.append(nn.Linear(self.seq_len, self.pred_len, dtype=torch.float64))
            self.Linear_Trend.append(nn.Linear(self.seq_len, self.pred_len, dtype=torch.float64))

    def forward(self, x):
        # x: [Batch, Input length, Channel]
        seasonal_init, trend_init = self.decompsition(x)
        seasonal_init, trend_init = seasonal_init.permute(0, 2, 1), trend_init.permute(0, 2, 1)

        seasonal_output = torch.zeros(
            [seasonal_init.size(0), seasonal_init.size(1), self.pred_len],
            dtype=seasonal_init.dtype).to(seasonal_init.device)
        trend_output = torch.zeros(
            [trend_init.size(0), trend_init.size(1), self.pred_len],
            dtype=trend_init.dtype).to(trend_init.device)
        for i in range(self.n_channels):
            seasonal_output[:, i, :] = self.Linear_Seasonal[i](seasonal_init[:, i, :])
            trend_output[:, i, :] = self.Linear_Trend[i](trend_init[:, i, :])

        x = seasonal_output + trend_output
        return x.permute(0, 2, 1)  # to [Batch, Output length, Channel]
