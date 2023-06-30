import typer
from os import listdir
from os.path import join
import json
import lightning as L
import numpy as np
import matplotlib.pyplot as plt
import pickle
from tqdm import tqdm

from models.linear import NLinear, DLinear
from utils import extract_H_T, create_dirs_if_not_exist
from constants import Config, DATASETS
from data import prepare_dataloaders

import logging

logging.getLogger("lightning.pytorch.utilities.rank_zero").setLevel(logging.WARNING)
logging.getLogger("lightning.pytorch.accelerators.cuda").setLevel(logging.WARNING)

import warnings

warnings.filterwarnings('ignore')


app = typer.Typer(pretty_exceptions_enable=False)


def build_model(model_name: str, config: Config) -> L.LightningModule:
    if model_name == 'nlinear':
        return NLinear(config=config)
    elif model_name == 'dlinear':
        return DLinear(config=config)


def average_predictions(predictions: list, batch_size: int) -> np.ndarray:
    preds = []
    for batch_index,batch in enumerate(predictions):
        for pred_sequence_index, pred_sequence in enumerate(batch):
            for time_step_index, time_step in enumerate(pred_sequence):
                pred_index = (batch_index * batch_size) + pred_sequence_index + time_step_index
                if pred_index >= len(preds):
                    preds.append([])
                preds[pred_index].append(time_step)

    firsts_preds = []
    means_preds = []
    for pred in preds:
        firsts = np.array(pred)[0,:]
        firsts_preds.append(firsts)

        means = np.mean(pred, axis=0)
        means_preds.append(means)

    return means_preds, firsts_preds


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


def count_total_loop(root_dir: str):
    i = 0
    for model_name in listdir(root_dir):
        model_dir = join(root_dir, model_name)
        for data in DATASETS:
            data_dir = join(model_dir, data)
            for ht_name in listdir(data_dir):
                i += 1
    
    return i


@app.command()
def main(
        tensorbaord_save_dir: str = 'exp'
        ):
    

    total_loop = count_total_loop(tensorbaord_save_dir)
    

    out_dir = 'reconstructed/'
    create_dirs_if_not_exist(out_dir)
    
    values_dir = join(out_dir, 'values')
    create_dirs_if_not_exist(values_dir)

    individuals_dir = join(out_dir, 'individuals')
    create_dirs_if_not_exist(individuals_dir)

    pbar = tqdm(total=total_loop)

    for model_name in listdir(tensorbaord_save_dir):
        model_dir = join(tensorbaord_save_dir,model_name)


        for data in DATASETS:
            data_dir = join(model_dir, data)

            targets = None
            means_preds_list = []
            firsts_preds_list = []

            for ht_name in listdir(data_dir):
                pbar.set_description(f'{model_name}, {data}, {ht_name}')

                ht_dir = join(data_dir, ht_name)
                H,T = extract_H_T(ht_name)
                checkpoint_name = listdir(join(ht_dir, 'checkpoints'))[0]
                checkpoint_path = join(ht_dir, 'checkpoints', checkpoint_name)

                with open(join(ht_dir, 'config.json')) as f:
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

                trainer = L.Trainer(
                    enable_progress_bar=False,
                    enable_model_summary=False,
                )

                predictions = trainer.predict(model, test_loader)
                
                means_preds, first_preds = average_predictions(predictions, batch_size)
                means_preds = scaler.inverse_transform(means_preds)
                means_preds_list.append(means_preds)

                first_preds = scaler.inverse_transform(first_preds)
                firsts_preds_list.append(first_preds)

                means_preds_obj_dir = join(values_dir, model_name, data, "means")
                create_dirs_if_not_exist(means_preds_obj_dir)
                means_preds_obj_path = join(means_preds_obj_dir, ht_name + '.pickle')
                with open(means_preds_obj_path, 'wb') as f:
                    pickle.dump(means_preds, f)
                    
                firsts_preds_obj_dir = join(values_dir, model_name, data, "firsts")
                create_dirs_if_not_exist(firsts_preds_obj_dir)
                firsts_preds_obj_path = join(firsts_preds_obj_dir, ht_name + '.pickle')
                with open(firsts_preds_obj_path, 'wb') as f:
                    pickle.dump(first_preds, f)      

                if targets is None:
                    batch_y_list = []
                    for batch in test_loader:
                        x,y = batch
                        batch_y_list.append(y)

                    _, targets = average_predictions(batch_y_list, batch_size)
                    targets = scaler.inverse_transform(targets)

                # plot individual
                indiv_means_preds_dir = join(individuals_dir, model_name, data, 'means')
                create_dirs_if_not_exist(indiv_means_preds_dir)
                indiv_means_preds_img_path = join(indiv_means_preds_dir, ht_name + '.png')
                plot_reconstructed(means_preds, targets)
                plt.savefig(indiv_means_preds_img_path)

                indiv_firsts_preds_dir = join(individuals_dir, model_name, data, 'firsts')
                create_dirs_if_not_exist(indiv_firsts_preds_dir)
                indiv_firsts_preds_img_path = join(indiv_firsts_preds_dir, ht_name + '.png')
                plot_reconstructed(first_preds, targets)
                plt.savefig(indiv_firsts_preds_img_path)

                pbar.update(1)


            targets_path = join(values_dir, model_name, data, 'targets.pickle')
            with open(targets_path, 'wb') as f:
                pickle.dump(targets, f)

            preds = np.mean(means_preds_list, axis=0)
            plot_reconstructed(preds, targets)
            img_name = f"{model_name}_{data}_means.png"
            img_path = join(out_dir, img_name)
            plt.savefig(img_path)


            preds = np.mean(firsts_preds_list, axis=0)
            plot_reconstructed(preds, targets)
            img_name = f"{model_name}_{data}_firsts.png"
            img_path = join(out_dir, img_name)
            plt.savefig(img_path)
            
            




if __name__ == '__main__':
    app()