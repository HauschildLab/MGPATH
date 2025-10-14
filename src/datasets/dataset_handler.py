"""
@author : Tien Nguyen
@date   : 2024-Aug-24
@desc   : -) The class DatasetHandler is implemented to work with
                                                            augmented dataset.
          -) Currently, it only works with training proportion.
          -) This class takes an object of
                                    the class Generic_MIL_Dataset as input.
@method :
         +) An object of the class Generic_MIL_Dataset has the following
                                                                attributes:
            -) self.data_dir_s: the path to the directory containing the
                                                    low resolution patches.
            -) self.data_dir_l: the path to the directory containing the
                                                    high resolution patches.
            -) self.slide_data: a Pandas DataFrame containing the
                                                    information of the slides.
         +) The class DatasetHandler is designed to compact the augmented
             slides with the self.slide_data.
             -) Create a new Python List `slide_data_s` containing
                 the information of the low resolution patches which
                                    are extracted from the self.slide_data.
             -) Create a new Python List `slide_data_l` containing
                    the information of the high resolution patches which
                                    are extracted from the self.slide_data.
             -) `slide_data_s` and `slide_data_l` must have the full
                                                file path to the patches.
             -) The `slide_data_s` and `slide_data_l` must have the same length.
             -) An element of `slide_data_s` is associated to an element
                                        of `slide_data_l` at the same index.
"""
from typing import List
from typing import Tuple
from typing import Dict

import os

import h5py
import pickle

import numpy

import torch
from torch.utils.data import Dataset

from datasets.dataset_generic import Generic_MIL_Dataset


def read_pkl_file(
    pkl_file: str
):
    with open(pkl_file, 'rb') as file:
        data = pickle.load(file)
    return data


class DatasetHandler(Dataset):
    def __init__(
        self,
        low_aug_patches_dir: str,
        high_aug_patches_dir: str,
        low_aug_graph_dir: str,
        high_aug_graph_dir: str,
        dataset: Generic_MIL_Dataset
    ) -> None:
        self.low_aug_patches_dir = low_aug_patches_dir
        self.high_aug_patches_dir = high_aug_patches_dir
        self.data_dir_s = dataset.data_dir_s
        self.data_dir_l = dataset.data_dir_l
        self.slide_data = dataset.slide_data
        slide_data_s, slide_data_l = self.compact_data()
        self.slide_data_s = slide_data_s
        self.slide_data_l = slide_data_l
        self.low_aug_graph_dir = low_aug_graph_dir
        self.high_aug_graph_dir = high_aug_graph_dir


    def __getitem__(
        self,
        index: int
    ) -> Tuple:
        """
        @desc:
            - This method is designed to return the features and the coordinates
                of the low and high resolution patches at the index.
            - Assert that the label of the low and high resolution patches
                are the same.
        """
        scale_s_file = self.slide_data_s[index]['h5_file']
        scale_l_file = self.slide_data_l[index]['h5_file']
        graph_s_file = os.path.basename(scale_s_file).replace(".h5", ".pickle")
        graph_s_file = os.path.join(self.low_aug_graph_dir, graph_s_file)
        graph_l_file = os.path.basename(scale_l_file).replace(".h5", ".pickle")
        graph_l_file = os.path.join(self.high_aug_graph_dir, graph_l_file)
        label_s = self.slide_data_s[index]['label']
        label_l = self.slide_data_l[index]['label']
        assert label_s == label_l
        scale_s = h5py.File(scale_s_file, 'r')
        scale_l = h5py.File(scale_l_file, 'r')
        features_s = torch.from_numpy(numpy.array(scale_s['features']))
        coords_s = torch.from_numpy(numpy.array(scale_s['coords']))
        features_l = torch.from_numpy(numpy.array(scale_l['features']))
        coords_l = torch.from_numpy(numpy.array(scale_l['coords']))
        graph_scale_s = read_pkl_file(graph_s_file)
        graph_scale_l = read_pkl_file(graph_l_file)
        nodes_s = graph_scale_s['nodes']
        edges_s = graph_scale_s['edges']
        nodes_l = graph_scale_l['nodes']
        edges_l = graph_scale_l['edges'] 
        return features_s, coords_s, nodes_s, edges_s, features_l, coords_l, nodes_l, edges_l, label_s


    def __len__(
        self
    ) -> int:
        return len(self.slide_data_s)


    def compact_data(
        self
    ) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
        """
        @desc:
            - This method is designed to compact the augmented slides
                with the self.slide_data.
        """
        slide_data_s = []
        slide_data_l = []
        for index, row in self.slide_data.iterrows():
            slide_id = row['slide_id']
            label = row['label']
            h5_file = f"{slide_id}.h5"
            scale_s = os.path.join(self.data_dir_s, h5_file)
            aug_scale_s = os.path.join(self.low_aug_patches_dir, h5_file)
            scale_l = os.path.join(self.data_dir_l, h5_file)
            aug_scale_l = os.path.join(self.high_aug_patches_dir, h5_file)
            slide_data_s.append({
                'slide_id': slide_id,
                'label': label,
                'h5_file': scale_s
            })
            slide_data_l.append({
                'slide_id': slide_id,
                'label': label,
                'h5_file': scale_l
            })
            slide_data_s.append({
                'slide_id': slide_id,
                'label': label,
                'h5_file': aug_scale_s
            })
            slide_data_l.append({
                'slide_id': slide_id,
                'label': label,
                'h5_file': aug_scale_l
            })
        return slide_data_s, slide_data_l
