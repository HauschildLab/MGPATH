# coding=utf-8
"""
@author : Tien Nguyen
@date   : 2024-Nov-09
@update : 2025-Feb-24
"""
import pandas
from torch.utils.data import Dataset

from base import BaseHandler

class DatasetHandler(Dataset):
    def __init__(
        self,
        seed       : int,
        labels     : list,
        slide_ids  : list,
        label_dict : dict,
    ) -> None:
        """
        @params:
            1) labels: the list of labels corresponding to the elements
                                                            of the `slide_ids`
            2) slide_ids: the list of slide ids (samples, but not patient ids).
            3) label_dict: a Python dictionary that maps the label to the
                                                        corresponding integer.
        """
        self.seed = seed
        self.label_dict = label_dict
        self.labels = labels
        self.slide_ids = slide_ids
        self.num_classes = len(set(self.label_dict.values()))

    def __len__(
        self
    ) -> int:
        return len(self.slide_ids)
