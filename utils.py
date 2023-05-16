import random
import os
import pandas as pd

def random_split_list(wind_speeds, test_perc, valid_perc):
    #TODO change name of all things with "wind_speeds" to make it generic
    test_size = int(len(wind_speeds) * test_perc)
    test_wind_speeds = random.sample(wind_speeds, test_size)

    train_wind_speeds = [x for x in wind_speeds if x not in test_wind_speeds]
    valid_wind_speeds = []
    if valid_perc > 0:
        valid_size = int(len(wind_speeds) * valid_perc)
        valid_wind_speeds = random.sample(train_wind_speeds, valid_size)
        train_wind_speeds = [x for x in train_wind_speeds if x not in valid_wind_speeds]

    assert len(train_wind_speeds) + len(test_wind_speeds) + len(valid_wind_speeds) == len(wind_speeds)
    assert set(train_wind_speeds).isdisjoint(set(valid_wind_speeds)) \
        and set(valid_wind_speeds).isdisjoint(set(test_wind_speeds)) \
        and set(train_wind_speeds).isdisjoint(set(test_wind_speeds))
    
    if len(valid_wind_speeds) > 0:
        return train_wind_speeds, test_wind_speeds, valid_wind_speeds
    return train_wind_speeds, test_wind_speeds

"""Data utils"""
def get_dir_name(x_start_factor: int, x_end_factor: int,
                    y_start_factor: int, y_end_factor: int,
                    grid_step_factor: float) -> str:
    dir = f"data/discr_factors_x{x_start_factor}_{x_end_factor}_y{y_start_factor}_{y_end_factor}_step{grid_step_factor}"
    if not os.path.exists(dir):
        os.makedirs(dir)
        print(f"Created the folder: {dir}")
    return dir

def get_filepath(x_start_factor: int, x_end_factor: int,
                    y_start_factor: int, y_end_factor: int,
                    grid_step_factor: float, wind_speed: int) -> str:
    dir = get_dir_name(x_start_factor, x_end_factor, y_start_factor, y_end_factor, grid_step_factor)
    filename = f"ws_{wind_speed}.csv"
    return os.path.join(dir, filename)

def get_discr_factors(dir_name: str) -> tuple[int, int, int, int, float]:
    import re
    def strip_letters(s: str) -> str:
        return re.sub(r"[^\d.-]+", '', s)

    factors = [strip_letters(f) for f in dir_name.strip("/").split("_")[2:]]
    x_start_factor, x_end_factor, y_start_factor, y_end_factor, grid_factor = \
        int(factors[0]), int(factors[1]), int(factors[2]), int(factors[3]), float(factors[4])
    return x_start_factor, x_end_factor, y_start_factor, y_end_factor, grid_factor

def load_csv(data_folder: str, wind_speed: int, include_ws_column: bool = True) -> pd.DataFrame:
    filepath = os.path.join(data_folder, f"ws_{wind_speed}.csv")
    df = pd.read_csv(filepath)
    if include_ws_column:
        df["ws"] = wind_speed
    return df