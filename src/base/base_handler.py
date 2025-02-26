# coding=utf-8
"""
@author : Tien Nguyen
@date   : 2024-Nov-09
@update : 2025-Feb-26
"""
from typing import Tuple

import pandas

from torch.utils.data import sampler
from torch.utils.data import DataLoader

class BaseHandler(object):

    def __init__(
        self,
        configs: dict
    ):
        self.seed = configs['seed']
        self.configs = configs

    def load_slide_ids(
        self,
        csv_path: str
    ) -> Tuple[list, list]:
        """
        @desc:
            - the function is implemented to parse the fold csv file.
        @params:
            - csv_path: str
                + the path to the fold csv file.
        """
        fold_df = pandas.read_csv(csv_path, sep=",")
        slide_ids = fold_df['slide_id'].tolist()
        labels = fold_df['label'].tolist()
        return slide_ids, labels

    def define_data_loader(
        self,
        data_handler,
        batch_size=1
    ) -> DataLoader:
        """
        @desc:
            1) the function is implemented to create a Pytorch data loader.
        """
        kwargs = {
            'num_workers': 4,
            'pin_memory': False
        }
        data_loader = DataLoader(data_handler, batch_size=batch_size,\
                            sampler=sampler.SequentialSampler(data_handler))
        return data_loader
