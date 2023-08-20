import typer
import os
from os import listdir
from os.path import join
import json
import lightning as L
import numpy as np
import matplotlib.pyplot as plt
import pickle
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
import pickle
import pandas as pd

from utils.exp import extract_H_T, create_dirs_if_not_exist, build_model, map_model_name
from constants import Config, DATASETS, H_LIST, T_LIST
from data import prepare_dataloaders

import logging

logging.getLogger("lightning.pytorch.utilities.rank_zero").setLevel(logging.WARNING)
logging.getLogger("lightning.pytorch.accelerators.cuda").setLevel(logging.WARNING)

import warnings

warnings.filterwarnings('ignore')

import matplotlib.font_manager as fm

# Define the font family name
font_family = "cmr10"

# Find the font file for the Computer Modern font
font_path = fm.findfont(fm.FontProperties(family=font_family))

# Set the default font family
plt.rcParams["font.family"] = font_family
plt.rcParams["font.sans-serif"] = [font_family]


app = typer.Typer(pretty_exceptions_enable=False)


def load_model_for_eval(ht_dir: str, model_name: str, data: str):

    checkpoint_name = listdir(join(ht_dir, 'checkpoints'))[0]
    checkpoint_path = join(ht_dir, 'checkpoints', checkpoint_name)

    with open(join(ht_dir, 'config.json')) as f:
        config_dict = json.load(f)

    model = build_model(model_name, data, **config_dict)
    model = model.load_from_checkpoint(checkpoint_path, config=model.config)
    model = model.float()  # change weight dtype to float32
    model.cuda()
    model.freeze()
    model.eval()

    return model


def make_targets(test_loader: DataLoader):
    targets_firsts = []
    targets_means = []

    batch_length = len(test_loader)
    for batch_index, batch in enumerate(test_loader):
        _, batch_y = batch
        batch_size = len(batch_y)
        for sequence_index, sequence in enumerate(batch_y):
            targets_firsts.append(sequence[0, :])
            targets_means.append(sequence[0, :])

            if batch_index == batch_length - 1 and sequence_index == batch_size - 1:
                for j in range(1, len(sequence)):
                    targets_means.append(sequence[j, :])

    return targets_means, targets_firsts


def average_predictions(predictions: list[torch.TensorType]) -> np.ndarray:
    batch_size = len(predictions[0])
    preds = []
    firsts_preds = []
    for batch_index, batch in enumerate(predictions):
        for pred_sequence_index, pred_sequence in enumerate(batch):
            for time_step_index, time_step in enumerate(pred_sequence):
                if time_step_index == 0:
                    firsts_preds.append(time_step)
                pred_index = (batch_index * batch_size) + pred_sequence_index + time_step_index
                if pred_index >= len(preds):
                    preds.append([])
                preds[pred_index].append(time_step)

    means_preds = []
    for pred in preds:

        means = np.mean(pred, axis=0)
        means_preds.append(means)

    firsts_preds = np.array(firsts_preds)
    means_preds = np.array(means_preds)

    return means_preds, firsts_preds


def plot_reconstructed(preds: np.ndarray, targets: np.ndarray):
    fig, axes = plt.subplots(ncols=3, nrows=1, figsize=(16, 4))
    titles = ['velocity', 'thrust', 'torqe']
    for i in range(3):
        min_val = min(np.min(targets[:, i]), np.min(preds[:, i]))
        max_val = max(np.max(targets[:, i]), np.max(preds[:, i]))

        axes[i].scatter(targets[:, i], preds[:, i])

        temp = np.arange(min_val, max_val, 0.1)
        axes[i].plot(temp, temp, c='r')

        axes[i].title.set_text(titles[i])


def save_pickle(dir: str, filename: str, obj: object):
    create_dirs_if_not_exist(dir)
    filepath = join(dir, filename)
    with open(filepath, 'wb') as f:
        pickle.dump(obj, f)

cache_collection = dict()

def get_mses(H: int, T: int, model_name: str, data: str, data_dir: str, cache_dir: str):

        cache_dir = join(cache_dir, 'cache')
        cache_file = join(cache_dir, 'cache.pickle')

        if cache_file in cache_collection:
            cache = cache_collection[cache_file]
        else:
            if os.path.exists(cache_file):
                with open(cache_file, 'rb') as f:
                    cache = pickle.load(f)
            else:
                cache = dict()
            cache_collection[cache_file] = cache

        cache_key = f'{model_name}_{data}_H{H}_T{T}'
        if cache_key in cache:
            mses = cache[cache_key]
        else:
            ht_name = f"H{H}-T{T}"
            ht_dir = join(data_dir, ht_name)
            model = load_model_for_eval(ht_dir, model_name, data)

            _, _, test_loader, _ = prepare_dataloaders(data,
                                                    batch_size=32,
                                                    seq_len=H,
                                                    pred_len=T,
                                                    n_channels=3)

            trainer = L.Trainer(
                enable_progress_bar=False,
                enable_model_summary=False,
            )

            predictions = trainer.predict(model, test_loader)
            if model_name == 'gcformer':
                predictions = [e[0] for e in predictions]
            predictions = torch.vstack(predictions)

            test_targets = []
            for batch in test_loader:
                _, y_batch = batch
                test_targets.append(y_batch)
            test_targets = torch.vstack(test_targets)

            assert predictions.shape == test_targets.shape

            mses = torch.mean((predictions - test_targets) ** 2, axis=0)

            cache[cache_key] = mses
            with open(cache_file, 'wb') as f:
                pickle.dump(cache, f)

            del model 
            del trainer
            del predictions
            del test_loader

            torch.cuda.empty_cache()

        return mses


@app.command()
def resconstruct(tensorbaord_save_dir: str = 'exp'):

    def iterate_exp_combinations(tensorbaord_save_dir: str, data: str = 'all'):

        if data == 'all':
            datas = DATASETS
        else:
            datas = [data]

        for data in datas:
            for model_name in listdir(tensorbaord_save_dir):
                model_dir = join(tensorbaord_save_dir, model_name)
                data_dir = join(model_dir, data)
                for ht_name in listdir(data_dir):
                    ht_dir = join(data_dir, ht_name)
                    yield data, data_dir, model_name, model_dir, ht_name, ht_dir

    total_loop = len(list(iterate_exp_combinations(tensorbaord_save_dir)))

    out_dir = 'reconstructed/'
    create_dirs_if_not_exist(out_dir)

    values_dir = join(out_dir, 'values')
    create_dirs_if_not_exist(values_dir)

    individuals_dir = join(out_dir, 'individuals')
    create_dirs_if_not_exist(individuals_dir)

    pbar = tqdm(total=total_loop)

    for data, data_dir, model_name, model_dir, ht_name, ht_dir in iterate_exp_combinations(
            tensorbaord_save_dir, data='1d'):
        pbar.set_description(f'{model_name}, {data}, {ht_name}')
        H, T = extract_H_T(ht_name)
        checkpoint_name = listdir(join(ht_dir, 'checkpoints'))[0]
        checkpoint_path = join(ht_dir, 'checkpoints', checkpoint_name)

        with open(join(ht_dir, 'config.json')) as f:
            config_dict = json.load(f)
        batch_size = config_dict['batch_size']
        del config_dict['batch_size']
        config = Config(**config_dict)

        model = build_model(model_name, config)
        model = model.load_from_checkpoint(checkpoint_path, config=config)
        model.cuda()
        model.freeze()
        model.eval()

        _, _, test_loader, scaler = prepare_dataloaders(data,
                                                        batch_size=batch_size,
                                                        seq_len=H,
                                                        pred_len=T,
                                                        n_channels=config.n_channels)

        trainer = L.Trainer(
            enable_progress_bar=False,
            enable_model_summary=False,
        )

        predictions = trainer.predict(model, test_loader)

        means_preds, first_preds = average_predictions(predictions)
        means_preds = scaler.inverse_transform(means_preds)
        first_preds = scaler.inverse_transform(first_preds)

        targets_means, targets_firsts = make_targets(test_loader)
        targets_means = scaler.inverse_transform(targets_means)
        targets_firsts = scaler.inverse_transform(targets_firsts)

        # plot individual
        title = f"{model_name} - {data} - {ht_name}"

        indiv_means_preds_dir = join(individuals_dir, model_name, data, 'means')
        create_dirs_if_not_exist(indiv_means_preds_dir)
        indiv_means_preds_img_path = join(indiv_means_preds_dir, ht_name + '.png')
        plot_reconstructed(means_preds, targets_means)
        plt.suptitle(title)
        plt.tight_layout()
        plt.savefig(indiv_means_preds_img_path)

        indiv_firsts_preds_dir = join(individuals_dir, model_name, data, 'firsts')
        create_dirs_if_not_exist(indiv_firsts_preds_dir)
        indiv_firsts_preds_img_path = join(indiv_firsts_preds_dir, ht_name + '.png')
        plot_reconstructed(first_preds, targets_firsts)
        plt.suptitle(title)
        plt.tight_layout()
        plt.savefig(indiv_firsts_preds_img_path)

        pbar.update(1)


@app.command()
def analyse_how_far(tensorboard_save_dir: str = 'exp', level: str = 'overall', log_x: bool = False):

    def do_level_T():

        def iterate_exp_combinations(tensorbaord_save_dir: str):
            for data in DATASETS:
                for model_name in listdir(tensorbaord_save_dir):
                    model_dir = join(tensorbaord_save_dir, model_name)
                    data_dir = join(model_dir, data)

                    yield data, data_dir, model_name, model_dir

        def count_total():
            c = len(list(iterate_exp_combinations(tensorboard_save_dir)))
            return c * (len(T_LIST))

        pbar = tqdm(total=count_total())
        for data, data_dir, model_name, model_dir in iterate_exp_combinations(tensorboard_save_dir):
            for T in T_LIST:
                pbar.set_description(f"{model_name} - {data} - T{T}")
                fig, axes = plt.subplots(ncols=3, nrows=1, figsize=(18, 6), sharey=True)

                n_H = len(H_LIST)
                colors = plt.cm.viridis(np.linspace(0, 1, n_H))
                for H_index, H in enumerate(H_LIST):
                    mses = get_mses(H, T, model_name, data, data_dir, out_dir)
                    for i, feature in enumerate(['velocity', 'thrust', 'torqe']):
                        axes[i].plot(range(T),
                                     mses[:, i],
                                     label=str(H),
                                     color=colors[n_H - H_index - 1],
                                     linewidth=3,
                                     alpha=0.75)
                        axes[i].title.set_text(feature)

                handles, labels = axes[-1].get_legend_handles_labels()
                fig.legend(handles, labels, loc='upper right', title='H')

                fig.supxlabel('time step in T')
                fig.supylabel('mse')

                title = f"{map_model_name(model_name)} - {data} - T{T}"
                plt.suptitle(title)
                plt.tight_layout()
                fig.subplots_adjust(right=0.95)

                img_dir = join(out_dir, 'level_T', model_name, data)
                create_dirs_if_not_exist(img_dir)
                img_path = join(img_dir, f"T{T}.png")
                plt.savefig(img_path)

                pbar.update(1)

    def do_level_data():

        def iterate_exp_combinations(tensorbaord_save_dir: str):
            for data in DATASETS:
                for model_name in listdir(tensorbaord_save_dir):
                    model_dir = join(tensorbaord_save_dir, model_name)
                    data_dir = join(model_dir, data)

                    yield data, data_dir, model_name, model_dir

        def count_total():
            c = len(list(iterate_exp_combinations(tensorboard_save_dir)))
            return c

        pbar = tqdm(total=count_total())
        for data, data_dir, model_name, model_dir in iterate_exp_combinations(tensorboard_save_dir):
            n_T = len(T_LIST)
            fig, axes = plt.subplots(ncols=3, nrows=n_T, figsize=(18, 18), sharey=True)
            pbar.set_description(f"{model_name} - {data}")

            for T_index, T in enumerate(T_LIST):
                n_H = len(H_LIST)
                colors = plt.cm.viridis(np.linspace(0, 1, n_H))
                for H_index, H in enumerate(H_LIST):
                    mses = get_mses(H, T, model_name, data, data_dir, out_dir)
                    for i, feature in enumerate(['velocity', 'thrust', 'torqe']):
                        if log_x:
                            axes[T_index, i].set_xscale('log')
                        axes[T_index, i].plot(range(T),
                                              mses[:, i],
                                              label=str(H),
                                              color=colors[n_H - H_index - 1],
                                              linewidth=3,
                                              alpha=0.75)
                        if T_index == 0:
                            axes[T_index, i].title.set_text(feature)
                        if i == 0:
                            axes[T_index, i].set_ylabel(f"T{T}")

            handles, labels = axes[0, -1].get_legend_handles_labels()
            fig.legend(handles, labels, loc='upper right', title='H')

            fig.supxlabel('time step in T')
            fig.supylabel('mse')

            title = f"{model_name} - {data}"
            plt.suptitle(title)
            plt.tight_layout(pad=3)
            fig.subplots_adjust(right=0.95)

            img_dir = join(out_dir, 'level_data', model_name)
            create_dirs_if_not_exist(img_dir)
            if log_x:
                img_path = join(img_dir, f"logx_{data}.png")
            else:
                img_path = join(img_dir, f"{data}.png")
            plt.savefig(img_path)

            pbar.update(1)

    def do_overall():

        model_names = ['nlinear', 'nlinear-ni', 'dlinear', 'dlinear-ni', 'tide-wo-a', 'tide-w-a', 'gcformer', 'fdnet']

        n_rows = 2
        n_cols = len(model_names) // n_rows
        fig, axes = plt.subplots(n_rows, n_cols, sharey=True, figsize=(4*n_cols, 4*n_rows))
        i = 0
        for model_name in model_names:
            print(model_name)
            t_to_sum = dict()
            t_to_n = dict()

            model_dir = join(tensorboard_save_dir, model_name)
            for data in DATASETS:
                data_dir = join(model_dir, data)

                for T_index, T in enumerate(T_LIST):
                    print(data, T)
                    for H_index, H in enumerate(H_LIST):
                        mses = get_mses(H, T, model_name, data, data_dir, out_dir)

                        for t in range(len(mses)):
                            n_channels = mses.shape[1]
                            if t not in t_to_sum:
                                t_to_sum[t] = torch.sum(mses[t])
                                t_to_n[t] = n_channels
                            else:
                                n = t_to_n[t]
                                t_to_sum[t] = t_to_sum[t] + torch.sum(mses[t])
                                t_to_n[t] = n + n_channels
                        del mses

            avg_mses = []
            for t in range(T_LIST[-1]):
                avg_mses.append(t_to_sum[t]/t_to_n[t])

            r = i // n_cols
            c = i % n_cols

            axes[r,c].plot(np.arange(T_LIST[-1]) + 1, avg_mses)
            axes[r,c].set_xlabel(map_model_name(model_name), fontsize=15)
            if c == 0:
                axes[r,c].set_ylabel('mse')
            if log_x:
                axes[r,c].set_xscale('log')
            i += 1
    
        # fig.supxlabel('model')
        # fig.supylabel('mse')

        # plt.suptitle('Overall MSE of time step from 1 to 720')
        plt.tight_layout()

        plt.savefig(join(out_dir, 'overall.png'))

    out_dir = 'how_far/'
    create_dirs_if_not_exist(out_dir)

    if level == 'T':
        do_level_T()
    elif level == 'data':
        do_level_data()
    elif level == 'overall':
        do_overall()
    else:
        raise "invalid level"


@app.command()
def feature_rmse(tensorboard_save_dir: str = 'exp', out_dir='rmse_feature'):

    df_cols = [
        (['data1D']*3) + (['data2D']*3) + (['data3D']*3),
        ['velocity', 'thrust','torque'] * 3
    ]

    model_names = ['nlinear', 'nlinear-ni', 'dlinear', 'dlinear-ni', 'tide-wo-a', 'tide-w-a', 'gcformer', 'fdnet']

    for model_name in model_names:
        col_values = []
        for data in DATASETS:
            model_dir = join(tensorboard_save_dir, model_name)
            data_dir = join(model_dir, data)

            all_mses = None

            for T_index, T in enumerate(T_LIST):
                n_H = len(H_LIST)
                for H_index, H in enumerate(H_LIST):
                    mses = get_mses(H, T, model_name, data, data_dir, 'how_far')
                    mses = np.array(mses)
                    if all_mses is None:
                        all_mses = mses
                    else:
                        all_mses = np.vstack([all_mses, mses])

            all_rmses = np.sqrt(mses)
            avg_rmses = np.mean(all_rmses, axis=0)

            assert len(avg_rmses) == 3
            assert len(avg_rmses.shape) == 1

            col_values.extend(list(avg_rmses))

        col_values = ['%.3f' % e for e in col_values]
        df_cols.append(col_values)

    create_dirs_if_not_exist(out_dir)
    out_path = join(out_dir, 'rmse.csv')

    columns = ['dataset','feature'] + model_names
    df = pd.DataFrame({columns[i]: df_cols[i] for i in range(len(columns))})
    df.to_csv(out_path, index=False)

if __name__ == '__main__':
    app()
