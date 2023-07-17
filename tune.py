import lightning as L
from lightning.pytorch.loggers import TensorBoardLogger
from ray.tune.integration.pytorch_lightning import TuneReportCallback
from ray import tune
import os
import typer

from data import prepare_dataloaders
from models.linear import NLinear, DLinear
from models.tide import TiDE, TiDEConfig
from constants import Config, H_LIST, T_LIST

app = typer.Typer()


class _TuneReportCallback(TuneReportCallback, L.Callback):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


seq_len = H_LIST[len(H_LIST) // 2]
pred_len = T_LIST[len(T_LIST) // 2]

n_channels = 3
max_epochs = 10

tensorboard_dir = os.path.abspath('./tuning')


@app.command(name='nlinear')
def tune_NLinear():

    def train_nlinear(config):

        batch_size = config['batch_size']
        lr = config['lr']

        train_loader, val_loader, test_loader, scaler = prepare_dataloaders('3d',
                                                                            batch_size=batch_size,
                                                                            seq_len=seq_len,
                                                                            pred_len=pred_len,
                                                                            n_channels=n_channels)

        nlinear = NLinear(Config(seq_len=seq_len, pred_len=pred_len, n_channels=n_channels, lr=lr))
        nlinear.cuda()

        logger = TensorBoardLogger(save_dir=tensorboard_dir, name='NLinear')

        metrics = {"val_loss": "val_loss", 'train_loss': 'train_loss'}
        tune_cb = _TuneReportCallback(metrics, on="validation_end")

        trainer = L.Trainer(max_epochs=max_epochs,
                            callbacks=[tune_cb],
                            logger=logger,
                            enable_progress_bar=False)
        trainer.fit(nlinear, train_loader, val_loader)

    num_samples = 15

    tune_config = {
        "lr": tune.loguniform(1e-4, 1e-1),
        "batch_size": tune.choice([8, 16, 32, 64]),
    }

    trainable = tune.with_parameters(train_nlinear)

    analysis = tune.run(trainable,
                        resources_per_trial={
                            "cpu": 1,
                            "gpu": 1
                        },
                        metric="val_loss",
                        mode="min",
                        config=tune_config,
                        num_samples=num_samples,
                        name="tune_linear")

    print(analysis)


@app.command(name='dlinear')
def tune_DLinear():

    def train_dlinear(config):

        batch_size = config['batch_size']
        lr = config['lr']

        train_loader, val_loader, test_loader, scaler = prepare_dataloaders('3d',
                                                                            batch_size=batch_size,
                                                                            seq_len=seq_len,
                                                                            pred_len=pred_len,
                                                                            n_channels=n_channels)

        model = DLinear(Config(seq_len=seq_len, pred_len=pred_len, n_channels=n_channels, lr=lr))
        model.cuda()

        logger = TensorBoardLogger(save_dir=tensorboard_dir, name='DLinear')

        metrics = {"val_loss": "val_loss", 'train_loss': 'train_loss'}
        tune_cb = _TuneReportCallback(metrics, on="validation_end")

        trainer = L.Trainer(max_epochs=max_epochs,
                            callbacks=[tune_cb],
                            logger=logger,
                            enable_progress_bar=False)
        trainer.fit(model, train_loader, val_loader)

    num_samples = 15

    tune_config = {
        "lr": tune.loguniform(1e-4, 1e-1),
        "batch_size": tune.choice([8, 16, 32, 64]),
    }

    trainable = tune.with_parameters(train_dlinear)

    analysis = tune.run(trainable,
                        resources_per_trial={
                            "cpu": 1,
                            "gpu": 1
                        },
                        metric="val_loss",
                        mode="min",
                        config=tune_config,
                        num_samples=num_samples,
                        name="tune_dlinear")

    print(analysis)


@app.command(name='tide-wo-a')
def tune_TiDE():

    def train(config):

        batch_size = config['batch_size']
        lr = config['lr']

        train_loader, val_loader, test_loader, scaler = prepare_dataloaders('3d',
                                                                            batch_size=batch_size,
                                                                            seq_len=seq_len,
                                                                            pred_len=pred_len,
                                                                            n_channels=n_channels)

        model_config = dict(config)
        del model_config['lr']
        del model_config['batch_size']

        model = TiDE(
            TiDEConfig(seq_len=seq_len,
                       pred_len=pred_len,
                       n_channels=n_channels,
                       lr=lr,
                       **model_config))
        model.cuda()

        logger = TensorBoardLogger(save_dir=tensorboard_dir, name='tide-wo-a')

        metrics = {"val_loss": "val_loss", 'train_loss': 'train_loss'}
        tune_cb = _TuneReportCallback(metrics, on="validation_end")

        trainer = L.Trainer(max_epochs=max_epochs,
                            callbacks=[tune_cb],
                            logger=logger,
                            enable_progress_bar=False)
        trainer.fit(model, train_loader, val_loader)

    num_samples = 20

    tune_config = {
        "lr": tune.loguniform(1e-5, 1e-2),
        "batch_size": tune.choice([32, 64, 128]),
        'hidden_dim': tune.choice([
            64,
            128,
            256,
            512,
        ]),
        'encoder_layer_num': tune.choice([1, 2, 3]),
        'decoder_layer_num': tune.choice([1, 2, 3]),
        'temporal_decoder_hidden': tune.choice([32, 64, 128]),
        'decoder_output_dim': tune.choice([4, 8, 16, 32]),
        'dropout_rate': tune.choice([0, 0.1, 0.2, 0.3, 0.5])
    }

    trainable = tune.with_parameters(train)

    analysis = tune.run(trainable,
                        resources_per_trial={
                            "cpu": 1,
                            "gpu": 1
                        },
                        metric="val_loss",
                        mode="min",
                        config=tune_config,
                        num_samples=num_samples,
                        name="tune_tide-wo-a")

    print(analysis)


@app.command(name='tide-w-a')
def tune_TiDE():

    def train(config):

        batch_size = config['batch_size']
        lr = config['lr']

        train_loader, val_loader, test_loader, scaler = prepare_dataloaders('3d',
                                                                            batch_size=batch_size,
                                                                            seq_len=seq_len,
                                                                            pred_len=pred_len,
                                                                            n_channels=n_channels)

        model_config = dict(config)
        del model_config['lr']
        del model_config['batch_size']

        model = TiDE(
            TiDEConfig(seq_len=seq_len,
                       pred_len=pred_len,
                       n_channels=n_channels,
                       lr=lr,
                       diameter=3,
                       **model_config))
        model.cuda()

        logger = TensorBoardLogger(save_dir=tensorboard_dir, name='tide-w-a')

        metrics = {"val_loss": "val_loss", 'train_loss': 'train_loss'}
        tune_cb = _TuneReportCallback(metrics, on="validation_end")

        trainer = L.Trainer(max_epochs=max_epochs,
                            callbacks=[tune_cb],
                            logger=logger,
                            enable_progress_bar=False)
        trainer.fit(model, train_loader, val_loader)

    num_samples = 20

    tune_config = {
        "lr": tune.loguniform(1e-5, 1e-2),
        "batch_size": tune.choice([32, 64, 128]),
        'hidden_dim': tune.choice([
            64,
            128,
            256,
            512,
        ]),
        'encoder_layer_num': tune.choice([1, 2, 3]),
        'decoder_layer_num': tune.choice([1, 2, 3]),
        'temporal_decoder_hidden': tune.choice([32, 64, 128]),
        'decoder_output_dim': tune.choice([4, 8, 16, 32]),
        'dropout_rate': tune.choice([0, 0.1, 0.2, 0.3, 0.5])
    }

    trainable = tune.with_parameters(train)

    analysis = tune.run(trainable,
                        resources_per_trial={
                            "cpu": 1,
                            "gpu": 1
                        },
                        metric="val_loss",
                        mode="min",
                        config=tune_config,
                        num_samples=num_samples,
                        name="tune_tide-w-a")

    print(analysis)


if __name__ == '__main__':

    app()
