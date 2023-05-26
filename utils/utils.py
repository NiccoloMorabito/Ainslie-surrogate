import random
import os
import pandas as pd
import xarray as xr
import numpy as np

def my_arange(start, end, step, include_end : bool = False) -> np.ndarray:
    """
    Function for np.arange(start, stop, step) without creating problems with the float representations
    """
    factor = 10
    while factor*step < 1:
        factor *= 10
    if include_end:
        end+=step
    return np.arange(start*factor, end*factor, step*factor) / factor

def random_split_list(input_list : list, perc_part2: float, perc_part3: float = 0) -> tuple:
    """
    Splits the input_list in 2 or 3 parts randomly according to the passed percentages
    """
    assert type(input_list) == list and len(input_list) > 1, "Unvalid input list"
    assert 0 < perc_part2 < 1, "Unvalid percentage for the second part"
    assert 0 <= perc_part3 < 1, "Unvalid percentage for the third part"
    part2_size = int(len(input_list) * perc_part2)
    part2 = random.sample(input_list, part2_size)

    part1 = [x for x in input_list if x not in part2]
    part3 = []
    if perc_part3 > 0:
        part3_size = int(len(input_list) * perc_part3)
        part3 = random.sample(part1, part3_size)
        part1 = [x for x in part1 if x not in part3]

    assert len(part1) + len(part2) + len(part3) == len(input_list)
    assert set(part1).isdisjoint(set(part3)) \
        and set(part3).isdisjoint(set(part2)) \
        and set(part1).isdisjoint(set(part2))
    
    if len(part3) > 0:
        return part1, part2, part3
    return part1, part2

def get_wake_coordinates_from_discr_factors(
        x_start_factor, x_end_factor, y_start_factor, y_end_factor, grid_factor):
    x_range = np.arange(x_start_factor, x_end_factor, grid_factor)
    y_range = np.arange(y_start_factor, y_end_factor, grid_factor)
    return x_range, y_range
