# main.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
from torch.optim.lr_scheduler import StepLR

from constants import Config


class LinearTrainBehavior(L.LightningModule):

    def __init__(self, config: Config):
        super(LinearTrainBehavior, self).__init__()

        self.seq_len = config.seq_len
        self.pred_len = config.pred_len
        self.n_channels = config.n_channels
        self.lr = config.lr 
        self.log_grad = config.log_grad

        self.validation_step_losses = []
        self.training_step_losses = []
        self.test_step_losses = []

    def extract_batch(self, batch):
        x, y = batch

        return x.cuda() , y.cuda()

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

    def test_step(self, batch, batch_idx):
        x,y = self.extract_batch(batch)

        y_hat = self.forward(x)
        loss = F.mse_loss(y_hat, y)
        self.test_step_losses.append(loss.item())

    def on_test_epoch_end(self) -> None:
        losses = self.test_step_losses
        avg_loss = sum(losses) / len(losses)
        self.log('test_loss', avg_loss, prog_bar=True)
        self.test_step_losses.clear()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = StepLR(optimizer, step_size=5, gamma=0.8)
        return {'optimizer': optimizer, 'lr_scheduler': scheduler }
    
    def on_after_backward(self):
        if self.log_grad:
            global_step = self.global_step
            for name, param in self.named_parameters():
                self.logger.experiment.add_histogram(name, param, global_step)
                if param.requires_grad:
                    self.logger.experiment.add_histogram(f"{name}_grad", param.grad, global_step)
    
    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        x,y = self.extract_batch(batch)

        return self.forward(x)


class NLinear(LinearTrainBehavior):
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
        seq_last = x[:,-1:,:].detach()
        x = x - seq_last

        output = torch.zeros([x.size(0),self.pred_len,x.size(2)],dtype=x.dtype).to(x.device)
        for i in range(self.n_channels):
            output[:,:,i] = self.linear[i](x[:,:,i])
        x = output

        x = x + seq_last
        return x # [Batch, Output length, Channel]
    

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

class DLinear(LinearTrainBehavior):
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
            self.Linear_Seasonal.append(nn.Linear(self.seq_len,self.pred_len, dtype=torch.float64))
            self.Linear_Trend.append(nn.Linear(self.seq_len,self.pred_len, dtype=torch.float64))

    def forward(self, x):
        # x: [Batch, Input length, Channel]
        seasonal_init, trend_init = self.decompsition(x)
        seasonal_init, trend_init = seasonal_init.permute(0,2,1), trend_init.permute(0,2,1)

        seasonal_output = torch.zeros([seasonal_init.size(0),seasonal_init.size(1),self.pred_len],dtype=seasonal_init.dtype).to(seasonal_init.device)
        trend_output = torch.zeros([trend_init.size(0),trend_init.size(1),self.pred_len],dtype=trend_init.dtype).to(trend_init.device)
        for i in range(self.n_channels):
            seasonal_output[:,i,:] = self.Linear_Seasonal[i](seasonal_init[:,i,:])
            trend_output[:,i,:] = self.Linear_Trend[i](trend_init[:,i,:])

        x = seasonal_output + trend_output
        return x.permute(0,2,1) # to [Batch, Output length, Channel]