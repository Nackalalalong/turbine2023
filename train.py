import concurrent.futures
import itertools
import os
from datetime import datetime

import lightning as L
import torch
import typer
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.loggers import TensorBoardLogger

torch.set_float32_matmul_precision('high')

import logging

logging.getLogger("lightning.pytorch.utilities.rank_zero").setLevel(logging.WARNING)
logging.getLogger("lightning.pytorch.accelerators.cuda").setLevel(logging.WARNING)

import warnings

warnings.filterwarnings('ignore')

from constants import (DATASETS, H_LIST, MODELS, T_LIST, Config,
                       DLinearTuneResult, NLinearTuneResult)
from data import prepare_dataloaders
from models.linear import DLinear, NLinear

app = typer.Typer(pretty_exceptions_enable=False)

n_channels = 3

def train(
        ModelClass,  
        data: str,
        max_epochs: int,
        batch_size: int,
        name: str,
        seq_len: int,
        pred_len: int,
        n_channels: int,
        lr: int,
        tensorboard_save_dir: str,
        version: str = None,
        enable_progress_bar: bool = True,
        enable_model_summary: bool = True,
        skip_done: bool = False,
        ):
    train_loader, val_loader, test_loader, scaler = prepare_dataloaders(
        data=data,
        batch_size=batch_size, 
        seq_len=seq_len, 
        pred_len=pred_len,
        n_channels=n_channels
    )

    config = Config(
        seq_len=seq_len,
        pred_len=pred_len,
        n_channels=n_channels,
        lr=lr
    )
        
    if skip_done:
        version_dir = os.path.join(tensorboard_save_dir, name, version)
        if os.path.exists(version_dir):
            return True

    model = ModelClass(config)
    model.cuda()

    early_stop = EarlyStopping(monitor="val_loss", mode="min", patience=3)
    logger = TensorBoardLogger(save_dir=tensorboard_save_dir, name=name, version=version)

    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        mode='min',
        filename=name + '-{epoch:02d}-{val_loss:.4f}',
        save_top_k=1
    )
    
    trainer = L.Trainer(
        max_epochs=max_epochs,
        callbacks=[checkpoint_callback, early_stop],
        logger=logger,
        enable_progress_bar=enable_progress_bar,
        enable_model_summary=enable_model_summary,
    )

    trainer.fit(model, train_loader, val_loader)


def translate_data(data: str):
    if data != 'all':
        return [data]

    return DATASETS


@app.command()
def main(
    model: str = 'nlinear',
    data: str = '1d',
    seq_len: int = H_LIST[0],
    pred_len: int = T_LIST[0],
    long_run: bool = False, 
    max_epochs: int = 20,
    parallel: bool = False,
    skip_done: bool = False,
    max_workers: int = 4,
    tensorboard_save_dir: str = 'exp'
):
    if model not in ['all'] + MODELS:
        raise 'invalid model'
    
    if data not in ['all'] + DATASETS:
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
        dataset_names = translate_data(data)
        args_list = itertools.product(dataset_names, H_LIST, T_LIST)

        if parallel:
            with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
                futures = {}

                for dataset_name, H, T in args_list:
                    kwargs = dict(
                        ModelClass=ModelClass,
                        data=dataset_name,
                        max_epochs=max_epochs,
                        batch_size=batch_size,
                        name=model,
                        seq_len=H,
                        pred_len=T,
                        n_channels=n_channels,
                        lr=lr,
                        version=f'{dataset_name}/H{H}-T{T}',
                        enable_progress_bar=False,
                        enable_model_summary=False,
                        skip_done=skip_done,
                        tensorboard_save_dir=tensorboard_save_dir
                    )

                    future = executor.submit(
                        train, 
                        **kwargs
                    )

                    futures[future] = kwargs

                for future in concurrent.futures.as_completed(futures):
                    e = future.exception()
                    if e:
                        kwargs = futures[future]
                        with open('exp_error.txt','a') as f:
                            f.write(f"{datetime.now().isoformat()}\t{model}\t{kwargs['data']}\t{kwargs['seq_len']}\t{kwargs['pred_len']}\t{str(e)}\n")
                
        else:
            for dataset_name,H,T in itertools.product(dataset_names, H_LIST, T_LIST):

                train(
                    ModelClass,
                    data=dataset_name,
                    max_epochs=max_epochs,
                    batch_size=batch_size,
                    name=model,
                    seq_len=H,
                    pred_len=T,
                    n_channels=n_channels,
                    lr=lr,
                    version=f'{dataset_name}/H{H}-T{T}',
                    skip_done=skip_done,
                    tensorboard_save_dir=tensorboard_save_dir
                )


    else:
        config = Config(
            seq_len=seq_len,
            pred_len=pred_len,
            n_channels=n_channels,
            lr=lr
        )

        for dataset_name in translate_data(data):      
            train(
                ModelClass,
                data=dataset_name,
                max_epochs=max_epochs,
                batch_size=batch_size,
                name=model,
                n_channels=n_channels,
                lr=lr,
                seq_len=seq_len,
                pred_len=pred_len,
                tensorboard_save_dir=tensorboard_save_dir
            )

if __name__ == '__main__':

    app()