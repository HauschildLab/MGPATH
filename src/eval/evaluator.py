# coding=utf-8
"""
@author : Tien Nguyen
@date   : 2024-Nov-09
@update : 2025-Feb-25
"""
from typing import List
import os

from torch.utils.data import DataLoader

from base import BaseHandler
from models import MILModel
from datasets import EMBDatasetHandler


class Evaluator(BaseHandler):
    """
    @desc:
        - the class is implemented to evaluate the performance of the model
          in K-fold cross-validation.
    """
    def __init__(
        self,
        configs: dict
    ) -> None:
        super(Evaluator, self).__init__(configs=configs)

    def evaluate(
        self,
        k_start    : int,
        k_end      : int,
        ckpt_dir   : str,
        splits_dir : str
    ):
        """
        @desc:
            - the function is implemented to evaluate the performance
            of the model in K-fold cross-validation.
        @params:
            - k_start: int
                + the start index of the fold.
            - k_end: int
                + the end index of the fold.
            - ckpt_dir: str
                + the directory to save the checkpoint.
            - splits_dir: str
                + the directory to save the slide ids for each fold.
        """
        ckpt_paths = self.load_checkpoint_paths(k_start=k_start,
                                                k_end=k_end, ckpt_dir=ckpt_dir)
        data_splits = self.load_splits_data(ckpt_paths=ckpt_paths,\
                                                        splits_dir=splits_dir)
        for idx in range(len(ckpt_paths)):
            checkpoint = self.load_checkpoint(ckpt_paths[idx])
            test_slide_ids = data_splits[idx]['test_slide_ids']
            test_labels = data_splits[idx]['test_labels']
            test_data_handler = self.create_data_handler(labels=test_labels,\
                                slide_ids=test_slide_ids, configs=self.configs)
            test_data_loader = self.define_data_loader(\
                                                data_handler=test_data_handler)
            model = self.load_model(ckpt_path=ckpt_paths[idx])


    def inference(
        self,
        mode        : str,
        model       : MILModel,
        device,
        data_loader : DataLoader 
    ):
        for idx, sample in enumerate(data_loader):
            features_s, features_l, nodes_s,\
                                    edges_s, nodes_l, edges_l, label = sample
            label = label.to(device)
            features_s, nodes_s, edges_s, = features_s.to(device),\
                                        nodes_s.to(device), edges_s.to(device)
            features_l, nodes_l, edges_l = features_l.to(device),\
                                        nodes_l.to(device), edges_l.to(device)
            with torch.no_grad():
                model(features_s, nodes_s, edges_s, features_l, nodes_l, edges_l, label)




    def load_model(
        self,
        ckpt_path: str
    ):
        config = {}
        config['input_size'] = self.config['input_size']
        config['num_classes'] = 2 #testing for TCGA-NSCLC
        config['ratio_graph'] = 0.2
        config['freeze_textEn'] = True
        config['typeGNN'] = self.config['gat_conv']
        model = MILModel(config=config, num_classes=config['num_classes'])
        return model


    def create_data_handler(
        self,
        labels     : list,
        slide_ids  : list,
        configs    : dict
    ) -> EMBDatasetHandler:
        data_handler = EMBDatasetHandler(\
                            seed=self.seed,\
                            labels=labels,\
                            slide_ids=slide_ids,\
                            label_dict=self.configs['labels'],\
                            data_dir_s=self.configs['scale_s_dir'],\
                            data_graph_dir_s=self.configs['graph_s_dir'],\
                            data_dir_l=self.configs['scale_l_dir'],\
                            data_graph_dir_l=self.configs['graph_l_dir']
        )
        return data_handler


    def load_splits_data(
        self,
        ckpt_paths : List[str],
        splits_dir : str
    ) -> List[dict]:
        data_splits = []
        for idx in range(len(ckpt_paths)):
            split_data = {}
            for split_type in ['train', 'val', 'test']:
                csv_path = os.path.join(splits_dir,\
                                        f"fold_{idx}_{split_type}_splits.csv")
                slide_ids, labels = self.load_slide_ids(csv_path)
                split_data[f"{split_type}_slide_ids"] = slide_ids
                split_data[f"{split_type}_labels"] = labels
            data_splits.append(split_data)
        return data_splits


    def load_checkpoint_paths(
        self,
        k_start  : int,
        k_end    : int,
        ckpt_dir : str
    ) -> list:
        """
        @desc:
            - the function is implemented to load the checkpoint paths
                                      of the model in K-fold cross-validation.
            - the function returns a list of checkpoint paths.
            - the checkpoint path is in the format: fold_{fold}_checkpoint.pt
            - the fold index is in the range `[k_start, k_end]`.
            - the function loads all the checkpoints in the directory
                                                                    `ckpt_dir`.
        """
        ckpt_paths = []
        folds = range(k_start, k_end + 1)
        for fold in folds:
            ckpt_path = os.path.join(ckpt_dir, f"fold_{fold}_checkpoint.pt")
            ckpt_paths.append(ckpt_path)
        return ckpt_paths
