import concurrent.futures
import itertools
import os
from datetime import datetime
import json
import shutil

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
from utils import read_event_values

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
        eval_after_train: bool = False
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

    version_dir = os.path.join(tensorboard_save_dir, name, version)
    with open(os.path.join(version_dir, 'config.json'), 'w') as f:
        config_dict = config.__dict__
        config_dict['batch_size'] = batch_size
        json.dump(config_dict, f)
        
    done = False
    if skip_done:
        val_loss_dir = os.path.join(version_dir, 'loss_val')
        if os.path.exists(val_loss_dir):
            event_filename = os.listdir(val_loss_dir)[0]
            event_filepath = os.path.join(val_loss_dir, event_filename)
            values = read_event_values(event_filepath)

            # there is an extra step 0
            if len(values) >= max_epochs + 1:
                done = True

    if ( skip_done and done ) and not eval_after_train:
        return
    
    model: L.LightningModule = ModelClass(config)
    model.cuda()    

    if not (skip_done and done):

        print('training', version_dir, '...')

        if os.path.exists(version_dir):
            shutil.rmtree(version_dir)

        # early_stop = EarlyStopping(monitor="val_loss", mode="min", patience=3)
        logger = TensorBoardLogger(save_dir=tensorboard_save_dir, name=name, version=version)

        checkpoint_callback = ModelCheckpoint(
            monitor='val_loss',
            mode='min',
            filename=name + '-{epoch:02d}-{val_loss:.4f}',
            save_top_k=1
        )

        trainer = L.Trainer(
            max_epochs=max_epochs,
            callbacks=[checkpoint_callback],
            logger=logger,
            enable_progress_bar=enable_progress_bar,
            enable_model_summary=enable_model_summary,
        )
        trainer.fit(model, train_loader, val_loader)
    
    del train_loader
    del val_loader

    if eval_after_train:
        print('evaluating', version_dir, '...')
        checkpoint_dir = os.path.join(version_dir, 'checkpoints')
        checkpoint_filename = os.listdir(checkpoint_dir)[0]
        checkpoint_filepath = os.path.join(checkpoint_dir, checkpoint_filename)
        model = model.load_from_checkpoint(checkpoint_filepath, config=config)
        model.freeze()
        model.eval()

        trainer = L.Trainer(
            enable_progress_bar=enable_progress_bar,
            enable_model_summary=enable_model_summary,
        )

        result = trainer.test(model, dataloaders=(test_loader), verbose=False)[0]

        with open(os.path.join(version_dir, 'eval.json'), 'w') as f:
            json.dump(result, f)

    del test_loader
    del trainer
    del model

    torch.cuda.empty_cache()


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
    tensorboard_save_dir: str = 'exp',
    eval_after_train: bool = False
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
                        tensorboard_save_dir=tensorboard_save_dir,
                        eval_after_train=eval_after_train
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
                    enable_progress_bar=False,
                    enable_model_summary=False,
                    version=f'{dataset_name}/H{H}-T{T}',
                    skip_done=skip_done,
                    tensorboard_save_dir=tensorboard_save_dir,
                    eval_after_train=eval_after_train
                )


    else:

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
                tensorboard_save_dir=tensorboard_save_dir,
                eval_after_train=eval_after_train,
                version=f'{dataset_name}/H{seq_len}-T{pred_len}',
                skip_done=skip_done
            )

if __name__ == '__main__':

    app()