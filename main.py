import lightning as L
from lightning.pytorch.callbacks import TQDMProgressBar
from lightning.pytorch.loggers import TensorBoardLogger

from data import prepare_3d_dataloaders
from models import NLinear
from utils import Config

seq_len = 24
pred_len = 96

train_loader, val_loader, test_loader = prepare_3d_dataloaders()

nlinear = NLinear(Config(seq_len=seq_len, pred_len=pred_len, n_channels=3))
nlinear.cuda()

logger = TensorBoardLogger(save_dir='tuning', name='NLinear')
trainer = L.Trainer(
    max_epochs=20,
    callbacks=[TQDMProgressBar(refresh_rate=10)],
    logger=logger
)
trainer.fit(nlinear, train_loader, val_loader)