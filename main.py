import lightning as L
from lightning.pytorch.callbacks import TQDMProgressBar
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.loggers import TensorBoardLogger

from data import prepare_3d_dataloaders
from constants import H_LIST, T_LIST
from models.linear import NLinear
from constants import Config, NLinearTuneResult

seq_len = H_LIST[0]
pred_len = T_LIST[0]

n_channels = 3
max_epochs = 10


if __name__ == '__main__':

    train_loader, val_loader, test_loader, scaler = prepare_3d_dataloaders(
        batch_size=NLinearTuneResult.best_batchsize, 
        seq_len=seq_len, 
        pred_len=pred_len,
        n_channels=n_channels
    )

    nlinear = NLinear(Config(
        seq_len=seq_len, 
        pred_len=pred_len, 
        n_channels=n_channels, 
        lr=NLinearTuneResult.best_lr
    ))
    nlinear.cuda()

    early_stop = EarlyStopping(monitor="val_loss", mode="min", patience=3)
    logger = TensorBoardLogger(save_dir='dev', name='NLinear')
    trainer = L.Trainer(
        gradient_clip_val=1,
        max_epochs=max_epochs,
        callbacks=[early_stop],
        logger=logger
    )
    trainer.fit(nlinear, train_loader, val_loader)