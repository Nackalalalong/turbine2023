from os.path import abspath, join

import torch
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

torch.set_default_dtype(torch.float32)

from constants import TEST_SIZE, VAL_SIZE

DATA_DIR = abspath('./data')


class CustomDataset(Dataset):

    def __init__(self, X: np.ndarray, y: np.ndarray, cuda=False) -> None:
        assert len(X) == len(y)

        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y)

        if cuda:
            self.X = self.X.cuda()
            self.y = self.y.cuda()

        self.length = len(X)

    def __getitem__(self, index) -> torch.TensorType:
        return self.X[index, :], self.y[index, :]

    def __len__(self):
        return self.length


def read_data(path: str) -> pd.DataFrame:
    df = pd.read_table(path, sep='\t')
    df.columns = [e.lower() for e in df.columns]

    return df


def read_1d() -> pd.DataFrame:
    path = join(DATA_DIR, 'Data1D.txt')

    return read_data(path)


def read_2d() -> pd.DataFrame:
    path = join(DATA_DIR, 'Data2D.txt')

    return read_data(path)


def read_3d() -> pd.DataFrame:
    path = join(DATA_DIR, 'Data3D.txt')

    return read_data(path)


def read(data: str) -> pd.DataFrame:
    if data == '3d':
        df = read_3d()
    elif data == '2d':
        df = read_2d()
    elif data == '1d':
        df = read_1d()
    else:
        raise Exception('invalid data')

    return df


def read_3d2d() -> pd.DataFrame:
    df3d = read_3d()
    df3d.columns = [e + "_3d" for e in df3d.columns]
    df2d = read_2d()
    df2d.columns = [e + "_2d" for e in df2d.columns]

    min_len = min(len(df3d), len(df2d))
    df = pd.concat([df3d, df2d], axis=1).dropna()

    assert len(df) == min_len
    assert len(df) == 50280

    return df


def read_3d2d1d() -> pd.DataFrame:
    df3d2d = read_3d2d()
    df1d = read_1d()
    df1d.columns = [e + "_1d" for e in df1d.columns]

    min_len = min(len(df3d2d), len(df1d))

    df = pd.concat([df3d2d, df1d], axis=1).dropna()

    assert len(df) == min_len
    assert len(df) == 50280

    return df


def scale_data(values: np.ndarray):
    assert len(values.shape) == 2

    cutoff = int(len(values) * (1 - TEST_SIZE))
    cutoff = int(cutoff * (1 - VAL_SIZE))
    assert abs(cutoff / len(values) - (1 - TEST_SIZE) * (1 - VAL_SIZE)) < 1e-3

    scaler = StandardScaler()
    scaler.fit(values[:cutoff, :])

    return scaler.transform(values), scaler


def make_X_y(values: np.ndarray, seq_len: int, pred_len: int):
    assert len(values.shape) == 2

    ys = []
    Xs = []

    i = 0
    while True:
        y_index_start = i + seq_len
        y_index_end = y_index_start + pred_len

        if y_index_end > values.shape[0]:
            break

        X = values[i:y_index_start, :]
        y = values[y_index_start:y_index_end, :]

        if len(ys) > 0:
            assert len(y) == len(ys[-1])
            assert len(X) == len(Xs[-1])

        ys.append(y)
        Xs.append(X)

        i += 1

    assert len(Xs) == len(ys)
    assert len(ys) == len(values) - pred_len - seq_len + 1

    return np.array(Xs), np.array(ys)


def train_test_split(X: np.ndarray, y: np.ndarray, test_size: int):
    assert len(X) == len(y)

    cutoff = int(len(X) * (1 - test_size))
    assert abs((len(X) - cutoff) / len(X) - test_size) < 1e-3

    return X[:cutoff, :], X[cutoff:, :], y[:cutoff, :], y[
        cutoff:,
    ]


def train_val_test_split(X: np.ndarray, y: np.ndarray):
    assert len(X) == len(y)

    train_X, test_X, train_y, test_y = train_test_split(X, y, VAL_SIZE)
    train_X, val_X, train_y, val_y = train_test_split(train_X, train_y, TEST_SIZE)

    return train_X, val_X, test_X, train_y, val_y, test_y


def _prepare_dataloaders(df: pd.DataFrame,
                         batch_size: int,
                         seq_len: int,
                         pred_len: int,
                         n_channels: int,
                         cuda: bool = False):
    assert len(df.values.shape) == 2

    values, scaler = scale_data(df.values)
    values = values.astype(np.float32)
    X, y = make_X_y(values, seq_len, pred_len)

    train_X, val_X, test_X, train_y, val_y, test_y = train_val_test_split(X, y)

    assert n_channels == train_X.shape[2]
    assert n_channels == val_X.shape[2]
    assert n_channels == test_X.shape[2]
    assert n_channels == train_y.shape[2]
    assert n_channels == val_y.shape[2]
    assert n_channels == test_y.shape[2]

    train_loader = DataLoader(CustomDataset(train_X, train_y, cuda),
                              shuffle=False,
                              batch_size=batch_size)
    val_loader = DataLoader(CustomDataset(val_X, val_y, cuda), shuffle=False, batch_size=batch_size)
    test_loader = DataLoader(CustomDataset(test_X, test_y, cuda),
                             shuffle=False,
                             batch_size=batch_size)

    return train_loader, val_loader, test_loader, scaler


def _prepare_dataloaders_all_in_one(batch_size: int,
                                    seq_len: int,
                                    pred_len: int,
                                    n_channels: int,
                                    cuda: bool = False,
                                    with_attr: bool = False):

    all_train_X, all_val_X, all_test_X, all_train_y, all_val_y, all_test_y, all_attr_train, all_attr_val, all_attr_test = None, None, None, None, None, None, None, None, None

    for data in ['1d', '2d', '3d']:
        values, scaler = scale_data(read(data).values)
        values = values.astype(np.float32)
        X, y = make_X_y(values, seq_len, pred_len)
        train_X, val_X, test_X, train_y, val_y, test_y = train_val_test_split(X, y)
        all_train_X = train_X.copy() if all_train_X is None else np.vstack([all_train_X, train_X])
        all_val_X = val_X.copy() if all_val_X is None else np.vstack([all_val_X, val_X])
        all_test_X = test_X.copy() if all_test_X is None else np.vstack([all_test_X, test_X])
        all_train_y = train_y.copy() if all_train_y is None else np.vstack([all_train_y, train_y])
        all_val_y = val_y.copy() if all_val_y is None else np.vstack([all_val_y, val_y])
        all_test_y = test_y.copy() if all_test_y is None else np.vstack([all_test_y, test_y])

        diameter = int(data[0])
        attr_train = np.ones((train_X.shape[0], train_X.shape[1], 1), dtype=np.float32) * diameter
        attr_val = np.ones((val_X.shape[0], val_X.shape[1], 1), dtype=np.float32) * diameter
        attr_test = np.ones((test_X.shape[0], test_X.shape[1], 1), dtype=np.float32) * diameter
        all_attr_train = attr_train if all_attr_train is None else np.vstack(
            [all_attr_train, attr_train])
        all_attr_val = attr_val if all_attr_val is None else np.vstack([all_attr_val, attr_val])
        all_attr_test = attr_test if all_attr_test is None else np.vstack(
            [all_attr_test, attr_test])

    if with_attr:
        all_train_X = np.concatenate([all_train_X, all_attr_train], axis=2)
        all_val_X = np.concatenate([all_val_X, all_attr_val], axis=2)
        all_test_X = np.concatenate([all_test_X, all_attr_test], axis=2)

    train_loader = DataLoader(CustomDataset(all_train_X, all_train_y, cuda),
                              shuffle=False,
                              batch_size=batch_size)
    val_loader = DataLoader(CustomDataset(all_val_X, all_val_y, cuda),
                            shuffle=False,
                            batch_size=batch_size)
    test_loader = DataLoader(CustomDataset(all_test_X, all_test_y, cuda),
                             shuffle=False,
                             batch_size=batch_size)

    return train_loader, val_loader, test_loader, None  # for scaler


def prepare_dataloaders(data: str,
                        batch_size: int,
                        seq_len: int,
                        pred_len: int,
                        n_channels: int,
                        cuda: bool = False):
    if n_channels is None:
        raise "n_channels must be number"

    if data == 'all-in-one' or data == 'all-in-one-w-a':
        with_attr = data == 'all-in-one-w-a'
        return _prepare_dataloaders_all_in_one(batch_size=batch_size,
                                               seq_len=seq_len,
                                               pred_len=pred_len,
                                               n_channels=n_channels,
                                               cuda=cuda,
                                               with_attr=with_attr)

    return _prepare_dataloaders(df,
                                batch_size=batch_size,
                                seq_len=seq_len,
                                pred_len=pred_len,
                                n_channels=n_channels,
                                cuda=cuda)


if __name__ == '__main__':
    # tests

    df = read_3d2d()

    df = read_3d2d1d()

    df = pd.DataFrame({'a': [1, 2, 3, 4, 5], 'b': [6, 7, 8, 9, 10]})
    X, y = make_X_y(df.values, 2, 2)
    assert np.array_equal(y, np.array([[[3, 8], [4, 9]], [[4, 9], [5, 10]]]))

    assert np.array_equal(X, np.array([[[1, 6], [2, 7]], [[2, 7], [3, 8]]]))

    prepare_dataloaders('1d', 8, 8, 8, 3)
    prepare_dataloaders('2d', 8, 8, 8, 3)
    prepare_dataloaders('3d', 8, 8, 8, 3)
