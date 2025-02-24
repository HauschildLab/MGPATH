"""
@author : Tien Nguyen
@date   : 2024-Nov-09
@update : 2025-Feb-24
"""
from typing import Union
from typing import Dict
from typing import Tuple

import pickle


def load_pickle_file(
    pkl_file: str,
) -> Union[Dict, Tuple, int, float]:
    with open(pkl_file, 'rb') as file:
        data = pickle.load(file)
    return data
