# main.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
from torch.optim.lr_scheduler import StepLR

from utils import Config


DEVICE = 'cuda'


class NLinear(L.LightningModule):
    """
    adapts from https://github.com/cure-lab/LTSF-Linear/blob/main/models/NLinear.py
    """
    def __init__(self, config: Config):
        super(NLinear, self).__init__()

        self.seq_len = config.seq_len
        self.pred_len = config.pred_len
        self.n_channels = config.n_channels

        self.linear = nn.ModuleList()
        for _ in range(self.n_channels):
            self.linear.append(nn.Linear(self.seq_len, self.pred_len, dtype=torch.float64))

        self.validation_step_losses = []
        self.training_step_losses = []

    def forward(self, x):
        # x: [Batch, Input length, Channel]
        seq_last = x[:,-1:,:].detach()
        x = x - seq_last

        output = torch.zeros([x.size(0),self.pred_len,x.size(2)],dtype=x.dtype).to(x.device)
        for i in range(self.n_channels):
            output[:,:,i] = self.linear[i](x[:,:,i])
        x = output

        x = x + seq_last
        return x # [Batch, Output length, Channel]
    
    def extract_batch(self, batch):
        x, y = batch
        x = x.cuda()
        y = y.cuda()

        return x,y

    def training_step(self, batch, batch_idx):
        x,y = self.extract_batch(batch)

        y_hat = self.forward(x)
        loss = F.mse_loss(y_hat, y)
        self.training_step_losses.append(loss.item())
        self.log('train_loss_step', loss, prog_bar=True)

        return loss
    
    def on_train_epoch_end(self):
        losses = self.training_step_losses
        avg_loss = sum(losses) / len(losses)
        self.logger.experiment.add_scalars('loss', {'train': avg_loss}, self.current_epoch) 
        self.log('train_loss', avg_loss, prog_bar=True, logger=False)
        self.training_step_losses.clear()
    
    def validation_step(self, batch, batch_idx):
        x,y = self.extract_batch(batch)
        logits = self.forward(x)
        loss = F.mse_loss(logits, y)
        self.validation_step_losses.append(loss.item())

    def on_validation_epoch_end(self):
        losses = self.validation_step_losses
        avg_loss = sum(losses) / len(losses)
        self.logger.experiment.add_scalars('loss', {'val': avg_loss}, self.current_epoch) 
        self.log('val_loss', avg_loss, prog_bar=True, logger=False)
        self.validation_step_losses.clear()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=5e-3)
        scheduler = StepLR(optimizer, step_size=5, gamma=0.8)
        return {'optimizer': optimizer, 'lr_scheduler': scheduler }
