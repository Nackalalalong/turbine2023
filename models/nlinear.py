# main.py
import torch
import torch.nn as nn
from dataclasses import dataclass

from models.base import ModelBehavior
from constants import Config

torch.set_default_dtype(torch.float32)


@dataclass(kw_only=True)
class NLinearConfig(Config):
    individual: bool = True


class NLinear(ModelBehavior):
    """
    adapts from https://github.com/cure-lab/LTSF-Linear/blob/main/models/NLinear.py
    """

    def __init__(self, config: NLinearConfig):
        super(NLinear, self).__init__(config)
        
        self.individual = config.individual

        if self.individual:
            self.Linear = nn.ModuleList()
            for i in range(self.channels):
                self.Linear.append(nn.Linear(self.seq_len,self.pred_len))
        else:
            self.Linear = nn.Linear(self.seq_len, self.pred_len)

    def forward(self, x):
        # x: [Batch, Input length, Channel]
        seq_last = x[:,-1:,:].detach()
        x = x - seq_last
        if self.individual:
            output = torch.zeros([x.size(0),self.pred_len,x.size(2)],dtype=x.dtype).to(x.device)
            for i in range(self.channels):
                output[:,:,i] = self.Linear[i](x[:,:,i])
            x = output
        else:
            x = self.Linear(x.permute(0,2,1)).permute(0,2,1)
        x = x + seq_last
        return x # [Batch, Output length, Channel]