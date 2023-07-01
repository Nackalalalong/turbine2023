from typing import Tuple, List
import os
from tensorboard.backend.event_processing import event_accumulator
import json
import itertools

from constants import H_LIST, T_LIST


def extract_H_T(H_T_dirname: str) -> Tuple[int,int]:
    h,t = H_T_dirname.split('-')

    return int(h[1:]), int(t[1:])


def read_event_values(event_path: str) -> List[float]:
    ea = event_accumulator.EventAccumulator(event_path)
    ea.Reload()

    values = [e.value for e in ea.Scalars('loss')]

    return values


def read_train_val_test_loss(ht_dir_path: str) -> Tuple[List[float], List[float], float]:
    loss_train_dir = os.path.join(ht_dir_path, 'loss_train')
    loss_val_dir = os.path.join(ht_dir_path, 'loss_val')

    train_event_filename = os.listdir(loss_train_dir)[0]
    val_event_filename = os.listdir(loss_val_dir)[0]

    train_event_filepath = os.path.join(loss_train_dir, train_event_filename)
    val_event_filepath = os.path.join(loss_val_dir, val_event_filename)

    eval_json_file = os.path.join(ht_dir_path, 'eval.json')
    with open(eval_json_file) as f:
        test_loss = json.load(f)['test_loss']

    return read_event_values(train_event_filepath), read_event_values(val_event_filepath), test_loss


def create_dirs_if_not_exist(dir: str):
    if not os.path.exists(dir):
        os.makedirs(dir)


def get_h_t_combination():
    names = []
    for h,t in itertools.product(H_LIST, T_LIST):
        names.append(f"H{h}-T{t}")

    return names