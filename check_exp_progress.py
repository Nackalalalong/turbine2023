from os import listdir, path
import typer
from constants import H_LIST, T_LIST, DATASETS

from utils.exp import read_event_values

app = typer.Typer()


@app.command()
def main(dir: str = 'exp', max_epochs: int = 20):

    total_exps = len(H_LIST) * len(T_LIST) * len(DATASETS)

    models = listdir(dir)

    for model in models:
        model_dir = path.join(dir, model)

        print('*' * 20)
        done_count = 0
        for data in listdir(model_dir):
            data_dir = path.join(model_dir, data)

            for version in listdir(data_dir):
                version_dir = path.join(data_dir, version)

                val_loss_dir = path.join(version_dir, 'loss_val')
                if path.exists(val_loss_dir):
                    event_filename = listdir(val_loss_dir)[0]
                    event_filepath = path.join(val_loss_dir, event_filename)
                    values = read_event_values(event_filepath)

                    # there is an extra step 0
                    if len(values) >= max_epochs + 1:
                        done_count += 1
                    else:
                        print('unfinished', event_filepath)

        print('model:', model)
        print(f'progress: {done_count}/{total_exps}')


if __name__ == "__main__":
    app()
