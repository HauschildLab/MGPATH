"""
@author : Tien Nguyen
@date   : 2024-Nov-09
@update : 2025-Feb-24
"""
import os
import argparse

import numpy
import pandas

import torch

import utils
from eval import Evaluator

def eval(
    args: argparse.Namespace,
    configs: dict,
):
    args = vars(args)
    configs.update(args)
   
    device = torch.device("cuda")
    configs['device'] = device
    evaluator = Evaluator(configs)
    evaluator.evaluate(k_start=configs['k_start'], k_end=configs['k_end'],\
                                   ckpt_dir=configs['checkpoint_dir'],\
                                             splits_dir=configs['splits_dir'])

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluation Script')
    parser.add_argument('--seed', type=int, default=2024, help='seed number')
    parser.add_argument('--k_start', type=int, default=0, help='start fold')
    parser.add_argument('--k_end', type=int, default=5, help='end fold')
    parser.add_argument('--task', type=str,\
                        help='task name, such as task_tcga_lung_subtyping')
    parser.add_argument("--config", type=str, default=None,\
            help='path to the config yaml file, such as yaml/tcga_lung.yml')
    parser.add_argument("--checkpoint_dir", type=str, default=None,\
            help='path to the checkpoint directory."\
                            "The checkpoint directory should contain"\
                                    " the model checkpoints for each fold')
    parser.add_argument("--splits_dir", type=str, default=None,\
                            help='path to the splits directory containing'\
                                                ' the splits for each fold')
    parser.add_argument("--input_size", type=int, default=1024,\
                            help='the dimension of the embedding features')
    parser.add_argument("--free_text_encoder", action='store_true',\
                                     default=True, help='freeze text encoder.')
    parser.add_argument('--ratio_graph', type=float, default=0.2,\
                                          help='the ratio of spatial features')
    parser.add_argument('--alignment', type=str, default=None,\
                               help='path to checkpoint of the PLIP alignment')
    parser.add_argument("--output_dir", type=str, default=None)


    args = parser.parse_args()
    utils.create_dir(args.output_dir)
    
    configs = utils.read_yaml_file(args.config)
    configs['text_prompt'] = utils.read_text_prompt_file(\
                                                  configs['text_prompt'])
    eval(args, configs)

