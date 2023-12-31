import lightning as L
from lightning.pytorch.loggers import TensorBoardLogger
from ray.tune.integration.pytorch_lightning import TuneReportCallback
from ray import tune
import os
import typer

from data import prepare_dataloaders
from constants import H_LIST, T_LIST
from utils.exp import build_model

app = typer.Typer()


class _TuneReportCallback(TuneReportCallback, L.Callback):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


seq_len = H_LIST[len(H_LIST) // 2]
pred_len = T_LIST[len(T_LIST) // 2]

n_channels = 3

tensorboard_dir = os.path.abspath('./tuning')


@app.command()
def main(model: str = 'nlinear-i',
         epochs: int = 10,
         num_samples: int = 10,
         data: str = '3d'):

    def train(config):
        batch_size = config['batch_size']

        train_loader, val_loader, test_loader, scaler = prepare_dataloaders(data=data,
                                                                            batch_size=batch_size,
                                                                            seq_len=seq_len,
                                                                            pred_len=pred_len,
                                                                            n_channels=n_channels)
        config['seq_len'] = seq_len
        config['pred_len'] = pred_len
        config['n_channels'] = n_channels
        _model = build_model(model, data, **config)
        _model.cuda()

        logger = TensorBoardLogger(save_dir=tensorboard_dir, name=model)

        metrics = {"val_loss": "val_loss", 'train_loss': 'train_loss'}
        tune_cb = _TuneReportCallback(metrics, on="validation_end")

        trainer = L.Trainer(epochs=epochs,
                            callbacks=[tune_cb],
                            logger=logger,
                            enable_progress_bar=False)
        trainer.fit(_model, train_loader, val_loader)

    tune_config = None
    if 'nlinear' in model:
        tune_config = {
            "lr": tune.loguniform(1e-4, 1e-1),
            "batch_size": tune.choice([16, 32, 64]),
        }
    elif 'dlinear' in model:
        tune_config = {
            "lr": tune.loguniform(1e-4, 1e-1),
            "batch_size": tune.choice([16, 32, 64]),
        }
    elif 'tide' in model:
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
    elif model == 'gcformer':
        tune_config = {
            "lr": tune.loguniform(1e-5, 1e-2),
            "batch_size": tune.choice([16, 32, 64]),
            'n_heads': tune.choice([8, 16, 32]),
            'd_model': tune.choice([64, 128, 256]),
            'd_ff': tune.choice([64, 128, 256, 512]),
            'patch_len': tune.choice([8, 16, 32]),
            'stride': tune.choice([8, 16, 32]),
            'dropout': tune.uniform(0, 0.5),
            'fc_dropout': tune.uniform(0, 0.5),
            'global_bias': tune.uniform(0, 0.5),
            'local_bias': tune.uniform(0, 0.5),
            'h_channel': tune.choice([32, 64]),
        }
    elif model == 'fdnet':
        tune_config = {
            "lr": tune.loguniform(1e-5, 1e-2),
            "batch_size": tune.choice([16, 32, 64]),
            'seq_kernel': tune.choice([3,5]),
            'attn_nums': tune.choice([2,3,5,7]),
            'd_model': tune.choice([8,16,32,64]),
            'pyramid': tune.choice([0,1,2,3,4]),
            'dropout': tune.uniform(0, 0.3),
            'ICOM': tune.choice([True,False])
        }
    else:
        raise "invalid model name"

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
                        name=model)

    print(analysis)


if __name__ == '__main__':

    app()
