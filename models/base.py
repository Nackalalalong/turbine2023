import torch
import torch.nn.functional as F
import lightning as L
from torch.optim.lr_scheduler import StepLR

from constants import Config


class ModelBehavior(L.LightningModule):

    def __init__(self, config: Config):
        super(ModelBehavior, self).__init__()

        self.config = config

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

        return x.cuda(), y.cuda()

    def training_step(self, batch, batch_idx):
        x, y = self.extract_batch(batch)

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
        x, y = self.extract_batch(batch)
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
        x, y = self.extract_batch(batch)

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
        return {'optimizer': optimizer, 'lr_scheduler': scheduler}

    def on_after_backward(self):
        if self.log_grad:
            global_step = self.global_step
            for name, param in self.named_parameters():
                self.logger.experiment.add_histogram(name, param, global_step)
                if param.requires_grad:
                    self.logger.experiment.add_histogram(f"{name}_grad", param.grad, global_step)

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        x, y = self.extract_batch(batch)

        return self.forward(x)