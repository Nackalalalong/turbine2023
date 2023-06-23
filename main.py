import lightning as L
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
import typer

from data import prepare_3d_dataloaders
from constants import H_LIST, T_LIST
from models.linear import NLinear
from constants import Config, NLinearTuneResult


app = typer.Typer()


seq_len = H_LIST[0]
pred_len = T_LIST[0]

n_channels = 3


@app.command(name='nlinear')
def train_nlinear(max_epochs: int = 10):
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
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',  # Monitor the validation loss
        mode='min',  # Minimum value indicates better performance
        filename='nlinear-{epoch:02d}-{val_loss:.4f}',  # File name pattern
        save_top_k=1  # Save only the best model based on validation loss
    )
    
    trainer = L.Trainer(
        gradient_clip_val=1,
        max_epochs=max_epochs,
        callbacks=[checkpoint_callback, early_stop],
        logger=logger
    )
    trainer.fit(nlinear, train_loader, val_loader)

if __name__ == '__main__':

    app()