import pandas as pd
from os.path import join, abspath
import numpy as np
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler

from constants import H_LIST, T_LIST, VAL_SIZE, TEST_SIZE

DATA_DIR = abspath('./data')


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

def scale_data(values: np.array):
    assert len(values.shape) == 2

    cutoff = int(len(values) * (1 - TEST_SIZE))
    cutoff = int(cutoff * (1 - VAL_SIZE))
    assert abs(cutoff/len(values) - (1-TEST_SIZE)*(1-VAL_SIZE)) < 1e-3

    scaler = StandardScaler()
    scaler.fit(values[:cutoff,:])

    return scaler.transform(values), scaler

def make_X_y(values: np.array, seq_len: int, pred_len: int):
    assert len(values.shape) == 2

    ys = []
    Xs = []

    i = 0
    while True:
        x_index_start = i + pred_len
        x_index_end = x_index_start + seq_len

        if x_index_end > values.shape[0]:
            break

        y = values[i:x_index_start,:]
        X = values[x_index_start:x_index_end,:]

        if len(ys) > 0:
            assert len(y) == len(ys[-1])
            assert len(X) == len(Xs[-1])

        ys.append(y)
        Xs.append(X)

        i += 1

    assert len(Xs) == len(ys)
    assert len(ys) == len(values) - pred_len - seq_len + 1

    return np.array(Xs), np.array(ys)

def train_test_split(X: np.array, y: np.array, test_size: int):
    assert len(X) == len(y)

    cutoff = int(len(X) * (1 - test_size))
    assert abs((len(X) - cutoff)/len(X) - test_size) < 1e-3

    return X[:cutoff, :], X[cutoff:,:], y[:cutoff,:], y[cutoff:,]

def train_val_test_split(X, y):
    assert len(X) == len(y)

    train_X, test_X, train_y, test_y = train_test_split(X, y, VAL_SIZE)
    train_X, val_X, train_y, val_y = train_test_split(train_X, train_y, TEST_SIZE)

    return train_X, val_X, test_X, train_y, val_y, test_y


def prepare_3d_dataloaders(batch_size: int = 8, seq_len: int = H_LIST[0], pred_len: int = T_LIST[0], n_channels: int = None):
    if n_channels is None:
        raise "n_channels must be number"
    print('preparing data3d...')
    df = read_3d()
    values, scaler = scale_data(df.values)
    X, y = make_X_y(values, seq_len, pred_len)

    train_X, val_X, test_X, train_y, val_y, test_y = train_val_test_split(X, y)

    assert n_channels == train_X.shape[2]
    assert n_channels == val_X.shape[2]
    assert n_channels == test_X.shape[2]
    assert n_channels == train_y.shape[2]
    assert n_channels == val_y.shape[2]
    assert n_channels == test_y.shape[2]

    train_loader = DataLoader(list(zip(train_X, train_y)), shuffle=False, batch_size=batch_size)
    val_loader = DataLoader(list(zip(val_X, val_y)), shuffle=False, batch_size=batch_size)
    test_loader = DataLoader(list(zip(test_X, test_y)), shuffle=False, batch_size=batch_size)
    print('done preparing data3d')

    return train_loader, val_loader, test_loader, scaler

if __name__ == '__main__':
    df = read_3d2d()

    df = read_3d2d1d()

    df = pd.DataFrame({'a': [1,2,3,4,5], 'b': [6,7,8,9,10]})
    X, y = make_X_y(df, 2, 2)
    assert np.array_equal(X, np.array([
        [[ 3, 8],
         [ 4, 9]],
        [[ 4, 9],
         [ 5, 10]]
    ]))

    assert np.array_equal(y, np.array([
        [[1, 6],
         [2, 7]],
        [[2, 7],
         [3, 8]]
    ]))


    prepare_3d_dataloaders()
