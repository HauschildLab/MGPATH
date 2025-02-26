# coding=utf-8
"""
@author : Tien Nguyen
@date   : 2024-Nov-09
@update : 2025-Feb-24
"""
from typing import Union
from typing import Dict
from typing import Tuple

import pickle

import pandas
import numpy

def load_pickle_file(
    pkl_file: str,
) -> Union[Dict, Tuple, int, float]:
    with open(pkl_file, 'rb') as file:
        data = pickle.load(file)
    return data

def read_text_prompt_file(
    text_prompt_path: str
):
    return numpy.array(pandas.read_csv(text_prompt_path,\
                                                header=None)).squeeze()
