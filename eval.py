import typer
import os
import json
import lightning as L
import torch
import numpy as np
import matplotlib.pyplot as plt

from models.linear import NLinear, DLinear
from utils import extract_H_T
from constants import Config, DATASETS
from data import prepare_dataloaders


app = typer.Typer(pretty_exceptions_enable=False)


def build_model(model_name: str, config: Config) -> L.LightningModule:
    if model_name == 'nlinear':
        return NLinear(config=config)
    elif model_name == 'dlinear':
        return DLinear(config=config)


def average_predictions(predictions: list, batch_size: int, only_first=False) -> np.ndarray:
    preds = []
    for batch_index,batch in enumerate(predictions):
        for pred_sequence_index, pred_sequence in enumerate(batch):
            for time_step_index, time_step in enumerate(pred_sequence):
                pred_index = (batch_index * batch_size) + pred_sequence_index + time_step_index
                if pred_index >= len(preds):
                    preds.append([])
                preds[pred_index].append(time_step)

    avg_preds = []
    for pred in preds:
        if only_first:
            temp = np.array(pred)[0,:]
        else:
            temp = np.mean(pred, axis=0)
        avg_preds.append(temp)

    return np.array(avg_preds)


def plot_reconstructed(preds: np.ndarray, targets: np.ndarray):
    fig, axes = plt.subplots(ncols=3, nrows=1, figsize=(16,4))
    titles = ['velocity', 'thrust', 'torqe']
    for i in range(3):
        min_val = min(np.min(targets[:,i]), np.min(preds[:,i]))
        max_val = max(np.max(targets[:,i]), np.max(preds[:,i]))

        axes[i].scatter(targets[:,i], preds[:,i])

        temp = np.arange(min_val,max_val,0.1)
        axes[i].plot(temp, temp, c='r')

        axes[i].title.set_text(titles[i])

    plt.show()



@app.command()
def main(
        tensorbaord_save_dir: str = 'exp'
        ):
    for data in DATASETS:
        for model_name in os.listdir(tensorbaord_save_dir):
            model_dir = os.path.join(tensorbaord_save_dir,model_name)

            data_dir = os.path.join(model_dir, data)

            for ht_name in os.listdir(data_dir):
                ht_dir = os.path.join(data_dir, ht_name)
                H,T = extract_H_T(ht_name)
                checkpoint_path = os.listdir(os.path.join(ht_dir, 'checkpoints'))[0]

                with open(os.path.join(ht_dir, 'config.json')) as f:
                    config_dict = json.load(f)
                batch_size = config_dict['batch_size']
                del config_dict['batch_size']
                config = Config(**config_dict)

                _, _, test_loader, scaler = prepare_dataloaders(
                    data, 
                    batch_size=batch_size,
                    seq_len=H,
                    pred_len=T,
                    n_channels=config.n_channels
                )

                model = build_model(model_name, config)
                model = model.load_from_checkpoint(checkpoint_path, config=config)
                model.freeze()
                model.eval()

                trainer = L.Trainer()

                predictions = trainer.predict(model, test_loader)
                
                preds = average_predictions(predictions, batch_size)
                batch_y_list = []
                for batch in test_loader:
                    x,y = batch
                    batch_y_list.append(y)

                targets = average_predictions(batch_y_list, batch_size, only_first=True)

                preds = scaler.inverse_transform(preds)
                targets = scaler.inverse_transform(targets)

                plt.scatter(targets, preds)
                plt.savefig("test_eval.png")

                exit(0)




if __name__ == '__main__':
    app()