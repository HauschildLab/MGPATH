# coding=utf-8
"""
@author : Tien Nguyen
@date   : 2024-Nov-09
@update : 2025-Feb-25
"""
from typing import List
import os

from tqdm import tqdm

import pandas

import torch
from torch.utils.data import DataLoader

from base import BaseHandler
from models import MILModel
from metrics import Meter
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
    ) -> dict:
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
        summary = {}
        ckpt_paths = self.load_checkpoint_paths(k_start=k_start,
                                                k_end=k_end, ckpt_dir=ckpt_dir)
        data_splits = self.load_splits_data(ckpt_paths=ckpt_paths,\
                                                        splits_dir=splits_dir)
        for idx in range(len(ckpt_paths)):
            test_slide_ids = data_splits[idx]['test_slide_ids']
            test_labels = data_splits[idx]['test_labels']
            test_data_handler = self.create_data_handler(labels=test_labels,\
                                slide_ids=test_slide_ids, configs=self.configs)
            test_data_loader = self.define_data_loader(\
                                                data_handler=test_data_handler)
            model = self.load_model(ckpt_path=ckpt_paths[idx])
            results = self.inference(mode=f'fold_{idx}', model=model,\
                                         device=self.configs['device'],\
                                                 data_loader=test_data_loader)
            summary.update(results)
        avg_acc = sum(fold['ACC'] for fold in results.values()) / len(results)
        avg_auc = sum(fold['AUC'] for fold in results.values()) / len(results)
        avg_f1 = sum(fold['F1'] for fold in results.values()) / len(results)
        summary_df = pandas.DataFrame.from_dict(summary, orient='index')
        average_values = {'ACC': avg_acc, 'AUC': avg_auc, 'F1': avg_f1}
        summary_df.loc['Average'] = average_values
        summary_df.to_csv(os.path.join(self.configs['output_dir'], 'results.csv'), index=True)
        return summary

    def inference(
        self,
        mode        : str,
        model       : MILModel,
        device      : torch.device,
        data_loader : DataLoader 
    ) -> dict:
        results = {}
        model = model.to(device)
        model.eval()
        meter = Meter(num_classes=self.configs['num_classes'])
        for idx, sample in tqdm(enumerate(data_loader)):
            features_s, features_l, nodes_s,\
                                    edges_s, nodes_l, edges_l, label = sample
            label = label.to(device)
            features_s, nodes_s, edges_s, = features_s.to(device),\
                                        nodes_s.to(device), edges_s.to(device)
            features_l, nodes_l, edges_l = features_l.to(device),\
                                        nodes_l.to(device), edges_l.to(device)
            with torch.no_grad():
                Y_prob, Y_hat, loss = model(features_s, nodes_s, edges_s,\
                                           features_l, nodes_l, edges_l, label)
                meter.add(Y_prob, Y_hat, label)
        accuracy, auc, f1 = meter.compute()
        results[mode] = {
            'ACC' : float(accuracy.detach().cpu().numpy()),
            'AUC' : float(auc.detach().cpu().numpy()),
            'F1'  : float(f1.detach().cpu().numpy())
        }
        return results



    def load_model(
        self,
        ckpt_path: str
    ):
        config = {}
        config['input_size'] = self.configs['input_size']
        config['num_classes'] = self.configs['num_classes']
        config['ratio_graph'] = self.configs['ratio_graph']
        config['freeze_textEn'] = self.configs['free_text_encoder']
        config['alignment'] = self.configs['alignment'] 
        config['text_prompt'] = self.configs['text_prompt']
        config['typeGNN'] = 'gat_conv'
        model = MILModel(config=config, num_classes=config['num_classes'])
        ckpt = torch.load(ckpt_path)
        model.load_state_dict(ckpt, strict=True)
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
                            label_dict=configs['labels'],\
                            data_dir_s=configs['location']['scale_s_dir'],\
                            data_graph_dir_s=configs['location']['graph_s_dir'],\
                            data_dir_l=configs['location']['scale_l_dir'],\
                            data_graph_dir_l=configs['location']['graph_l_dir']
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

