from typing import Tuple, List
import os
from tensorboard.backend.event_processing import event_accumulator
import json
import itertools
import copy

from constants import H_LIST, T_LIST, TUNE_RESULT
from models.nlinear import NLinear, NLinearConfig
from models.dlinear import DLinear, DLinearConfig
from models.base import ModelBehavior
from models.tide import TiDE, TiDEConfig
from models.gcformer import GCFormer, GCFormerConfig
from models.fdnet import FDNet, FDNetConfig


def extract_H_T(H_T_dirname: str) -> Tuple[int, int]:
    h, t = H_T_dirname.split('-')

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
    for h, t in itertools.product(H_LIST, T_LIST):
        names.append(f"H{h}-T{t}")

    return names


def get_model_class(model_name: str):
    if 'dlinear' in model_name:
        return DLinear
    if 'nlinear' in model_name:
        return NLinear
    if model_name == 'gcformer':
        return GCFormer
    if 'tide' in model_name:
        return TiDE
    if model_name == 'fdnet':
        return FDNet
    raise "invalid model name"


def get_config_class(model_name: str):
    if 'dlinear' in model_name:
        return DLinearConfig
    if 'nlinear' in model_name:
        return NLinearConfig
    if model_name == 'gcformer':
        return GCFormerConfig
    if 'tide' in model_name:
        return TiDEConfig
    if model_name == 'fdnet':
        return FDNetConfig
    raise "invalid model name"


def get_tune_result(model_name: str):
    return copy.deepcopy(TUNE_RESULT[model_name])


def build_model(model_name: str, data: str = None, **config_kwargs) -> ModelBehavior:
    ModelClass = get_model_class(model_name)
    ConfigClass = get_config_class(model_name)

    if model_name == 'tide-w-a':
        config_kwargs['has_attribute'] = True

    elif model_name == 'nlinear-ni':
        config_kwargs['individual'] = False

    elif model_name == 'nlinear':
        config_kwargs['individual'] = True

    elif model_name == 'dlinear-ni':
        config_kwargs['individual'] = False

    elif model_name == 'dlinear':
        config_kwargs['individual'] = True

    config = ConfigClass(**config_kwargs)
    model = ModelClass(config)

    if model_name == 'nlinear-ni' or model_name == 'dlinear-ni':
        assert model.config.individual == False
        assert model.individual == False
    elif model_name == 'nlinear' or model_name == 'dlinear':
        assert model.config.individual == True
        assert model.individual == True
    elif model_name == 'tide-w-a':
        assert model.config.has_attribute == True

    return model
    

def map_model_name(name: str) -> str:
    return {
        'nlinear': 'NLinear-i',
        'nlinear-ni': 'NLinear-ni',
        'dlinear': 'DLinear-i',
        'dlinear-ni': 'DLinear-ni',
        'tide-wo-a': 'TiDE-wo-a',
        'tide-w-a': 'TiDE-w-a',
        'gcformer': 'GCFormer',
        'fdnet': 'FDNet'
    }[name]