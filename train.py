import lightning as L
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint
import typer
import itertools
import warnings
warnings.filterwarnings('ignore')

from data import prepare_dataloaders
from constants import H_LIST, T_LIST
from models.linear import NLinear, DLinear
from constants import Config, NLinearTuneResult, DLinearTuneResult


app = typer.Typer(pretty_exceptions_enable=False)

n_channels = 3

def train(
        model: L.LightningModule,  
        data: str,
        max_epochs: int,
        batch_size: int,
        name: str,
        seq_len: int,
        pred_len: int,
        version: str = None
        ):
    train_loader, val_loader, test_loader, scaler = prepare_dataloaders(
        data=data,
        batch_size=batch_size, 
        seq_len=seq_len, 
        pred_len=pred_len,
        n_channels=n_channels
    )

    model.cuda()

    early_stop = EarlyStopping(monitor="val_loss", mode="min", patience=3)
    logger = TensorBoardLogger(save_dir='exp', name=name, version=version)

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


@app.command()
def main(
    model: str = 'nlinear',
    data: str = '1d',
    seq_len: int = H_LIST[0],
    pred_len: int = T_LIST[0],
    long_run: bool = False, 
    max_epochs: int = 100,

):
    if model not in ['all', 'nlinear','dlinear','jtft']:
        raise 'invalid model'
    
    if data not in ['1d','2d','3d']:
        raise 'invalid data'
    
    lr = None
    batch_size = None
    ModelClass = None
    if model == 'nlinear':
        lr = NLinearTuneResult.best_lr
        batch_size = NLinearTuneResult.best_batchsize
        ModelClass = NLinear
    elif model == 'dlinear':
        lr = DLinearTuneResult.best_lr
        batch_size = DLinearTuneResult.best_batchsize
        ModelClass = DLinear
    
    if long_run:
        for H,T in itertools.product(H_LIST, T_LIST):
            config = Config(
                seq_len=H,
                pred_len=T,
                n_channels=n_channels,
                lr=lr
            )
            
            train(
                ModelClass(config),
                data=data,
                max_epochs=max_epochs,
                batch_size=batch_size,
                name=model,
                seq_len=H,
                pred_len=T,
                version=f'H{H}-T{T}'
            )

    else:
        config = Config(
            seq_len=seq_len,
            pred_len=pred_len,
            n_channels=n_channels,
            lr=lr
        )
        
        train(
            ModelClass(config),
            data=data,
            max_epochs=max_epochs,
            batch_size=batch_size,
            name=model,
            seq_len=seq_len,
            pred_len=pred_len
        )

if __name__ == '__main__':

    app()