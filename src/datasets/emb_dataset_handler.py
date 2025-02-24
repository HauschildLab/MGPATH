"""
@author : Tien Nguyen
@date   : 2024-Nov-09
@update : 2025-Feb-24
"""
from typing import Tuple

import os
import h5py

import numpy

import utils
from datasets import DatasetHandler

class EMBDatasetHandler(DatasetHandler):
    """
    @desc:
        1) This class is a subclass of DatasetHandler.
        2) This class is designed to handle embedding features of 
            Whole Slide Images (WSIs), which are extracted from vision 
            encoders or vision foundation models. These embedding features 
            are stored in H5 files.
    """

    def __init__(
        self,
        seed: int,
        csv_path: str,
        label_dict: dict,
        data_dir_s: str,
        data_graph_dir_s: str,
        data_dir_l: str,
        data_graph_dir_l: str,
    ):
        """
        @desc:
            1) there are two kinds of embedding features which are extracted
               from a WSI spatial features, and local features.
            2) the proposed model requires a WSI has to have 2 magnifications
               5x (low resolution) and 10x (high resolution).
            3) therefore, there are 2 spatial features corresponding to 2
                magnifications.
            4) there are 4 directories in total:
                a)data_dir_s: the folder contains local features of 5x
                b)data_graph_dir_s: the folder contains graph features of 5x
                c)data_dir_l: the folder contains local features of 10x
                d)data_graph_dir_l: the folder contains graph features of 10x
        """
        super(EMBDatasetHandler, self).__init__(
            seed=seed,
            csv_path=csv_path,
            label_dict=label_dict,
        )
        self.data_dir_s = data_dir_s
        self.data_graph_dir_s = data_graph_dir_s
        self.data_dir_l = data_dir_l
        self.data_graph_dir_l = data_graph_dir_l

    def __getitem__(
        self, 
        index: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, \
                                            torch.Tensor, torch.Tensor, int]:
        slide_id = self.slide_ids[index]
        label = self.labels[index]
        label = self.label_dict[label]
        features_s, features_l = self.load_local_patch_features(slide_id)
        nodes_s, edges_s, nodes_l, edges_l = self.load_spatial_features(\
                                                                    slide_id)
        return features_s, features_l, nodes_s, edges_s, nodes_l, edges_l, label

    def load_local_patch_features(
        self,
        slide_id: str
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        scale_s_h5_file = os.path.join(self.data_dir_s,\
                                                    '{}.h5'.format(slide_id))
        scale_l_h5_file = os.path.join(self.data_dir_l,\
                                                    '{}.h5'.format(slide_id))
        scale_s = h5py.File(scale_s_h5_file, 'r')
        scale_l = h5py.File(scale_l_h5_file, 'r')
        features_s = torch.from_numpy(numpy.array(scale_s['features']))
        features_l = torch.from_numpy(numpy.array(scale_l['features']))
        return features_s, features_l

    def load_spatial_features(
        self,
        slide_id: str
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        graph_scale_s_pkl_file = os.path.join(self.data_graph_dir_s,\
                                                        f"{slide_id}.pickle")
        graph_scale_s = load_pickle_file(graph_scale_s_pkl_file)
        graph_scale_l_pkl_file = os.path.join(self.data_graph_dir_l,\
                                                        f"{slide_id}.pickle")
        graph_scale_l = load_pickle_file(graph_scale_l_pkl_file)
        nodes_s = graph_scale_s['nodes']
        edges_s = graph_scale_s['edges']
        nodes_l = graph_scale_l['nodes']
        edges_l = graph_scale_l['edges']
        return nodes_s, edges_s, nodes_l, edges_l
