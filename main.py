import lightning as L
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint
import typer

from data import prepare_3d_dataloaders
from constants import H_LIST, T_LIST
from models.linear import NLinear, DLinear
from constants import Config, NLinearTuneResult


app = typer.Typer(pretty_exceptions_enable=False)


seq_len = H_LIST[0]
pred_len = T_LIST[0]

n_channels = 3


def train(
        model: L.LightningModule,  
        max_epochs: int,
        batch_size: int,
        name: str
        ):
    train_loader, val_loader, test_loader, scaler = prepare_3d_dataloaders(
        batch_size=batch_size, 
        seq_len=seq_len, 
        pred_len=pred_len,
        n_channels=n_channels
    )

    model.cuda()

    early_stop = EarlyStopping(monitor="val_loss", mode="min", patience=3)
    logger = TensorBoardLogger(save_dir='dev', name=name)

    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        mode='min',
        filename=name + '-{epoch:02d}-{val_loss:.4f}',
        save_top_k=1
    )
    
    trainer = L.Trainer(
        max_epochs=max_epochs,
        callbacks=[checkpoint_callback, early_stop],
        logger=logger
    )

    trainer.fit(model, train_loader, val_loader)
    


@app.command(name='nlinear')
def train_nlinear(max_epochs: int = 10):

    nlinear = NLinear(Config(
        seq_len=seq_len, 
        pred_len=pred_len, 
        n_channels=n_channels, 
        lr=NLinearTuneResult.best_lr
    ))

    train(nlinear, max_epochs=max_epochs, batch_size=NLinearTuneResult.best_batchsize, name='NLinear')


@app.command(name='dlinear')
def train_dlinear(max_epochs: int = 10):

    dlinear = DLinear(Config(
        seq_len=seq_len, 
        pred_len=pred_len, 
        n_channels=n_channels, 
        lr=1e-3
    ))

    train(dlinear, max_epochs=max_epochs, batch_size=8, name='DLinear')


if __name__ == '__main__':

    app()