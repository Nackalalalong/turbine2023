import lightning as L
from lightning.pytorch.callbacks import TQDMProgressBar
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.loggers import TensorBoardLogger

from data import H_list, T_list, prepare_3d_dataloaders
from models import NLinear
from utils import Config

seq_len = H_list[len(H_list)//2]
pred_len = T_list[len(T_list)//2]

n_channels = 3

train_loader, val_loader, test_loader, scaler = prepare_3d_dataloaders(seq_len=seq_len, pred_len=pred_len,n_channels=n_channels)

nlinear = NLinear(Config(seq_len=seq_len, pred_len=pred_len, n_channels=n_channels))
nlinear.cuda()

early_stop = EarlyStopping(monitor="val_loss", mode="min", patience=5)
logger = TensorBoardLogger(save_dir='tuning', name='NLinear')
trainer = L.Trainer(
    max_epochs=100,
    callbacks=[TQDMProgressBar(refresh_rate=10)],
    logger=logger
)
trainer.fit(nlinear, train_loader, val_loader)