import pandas as pd
from os.path import join


data_dir = 'data'

def read_data(path: str) -> pd.DataFrame:
    df = pd.read_table(path, sep='\t')

    return df

def read_1d() -> pd.DataFrame:
    path = join(data_dir, 'Data1D.txt')

    return read_data(path)