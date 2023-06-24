import typer
import os


from models.linear import NLinear, DLinear
from utils import extract_H_T


app = typer.Typer(pretty_exceptions_enable=False)

@app.command()
def main(
        tensorbaord_save_dir: str = 'exp'
        ):
    for model in os.listdir(tensorbaord_save_dir):
        model_dir = os.path.join(tensorbaord_save_dir,model)
        for data in os.listdir(model_dir):
            data_dir = os.path.join(model_dir, data)
            for ht_name in os.listdir(data_dir):
                ht_dir = os.path.join(data_dir, ht_name)
                H,T = extract_H_T(ht_name)
                checkpoint_path = os.listdir(os.path.join(ht_dir, 'checkpoints'))[0]

